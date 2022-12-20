import torch
import torch.nn.functional as functional


class PerspectiveResampler(object):
    def __init__(self, max_depth=100, min_depth=0.01, min_sampling_res=3):
        self.min_disp = 1 / max_depth
        self.max_disp = 1 / min_depth

        self.min_sampling_res = min_sampling_res

        if self.min_sampling_res < 3:
            raise ValueError(
                'Bilinear sampling needs at least a 2x2 image to sample from. '
                'Increase --min_sampling_res to at least 3.'
            )

    def _homogeneous_grid_like(self, ref):
        n, c, h, w = ref.shape

        grid_x = torch.linspace(0, w - 1, w, device=ref.device)
        grid_y = torch.linspace(0, h - 1, h, device=ref.device)

        grid_x = grid_x.view(1, 1, -1, 1).expand(n, h, w, 1)
        grid_y = grid_y.view(1, -1, 1, 1).expand(n, h, w, 1)

        grid_w = torch.ones(n, h, w, 1, device=ref.device)

        grid = torch.cat((grid_x, grid_y, grid_w), 3)

        return grid

    def _to_pointcloud(self, depth, cam_inv):
        # Generate a grid that resembles the coordinates
        # of the projection plane
        grid = self._homogeneous_grid_like(depth)

        # Use the inverse camera matrix to generate
        # a vector that points from the camera focal
        # point to the pixel position on the projection plane.
        # Use unsqueeze and squeeze to coax pytorch into
        # performing a matrix - vector product.
        pointcloud = (cam_inv @ grid.unsqueeze(-1)).squeeze(-1)

        # Transpose the depth from shape (n, 1, h, w) to (n, h, w, 1)
        # to match the pointcloud shape (n, h, w, 3) and
        # multiply the projection plane vectors by the depth.
        pointcloud = pointcloud * depth.permute(0, 2, 3, 1)

        # Make the pointcloud coordinates homogeneous
        # by extending each vector with a one.
        pointcloud = torch.cat(
            (pointcloud, torch.ones_like(pointcloud[:,:,:,:1])),
            3
        )

        return pointcloud

    def _surface_normal(self, pointcloud):
        # Pointcloud has shape (n, h, w, 4).
        # Calculate the vectors pointing from a point associated with
        # a pixel to the points associated with the pixels down and to the right.
        dy = pointcloud[:,1:,:-1,:3] - pointcloud[:,:-1,:-1,:3]
        dx = pointcloud[:,:-1,1:,:3] - pointcloud[:,:-1,:-1,:3]

        # Calculate the normal vector for the plane spanned
        # by the vectors above.
        n = torch.cross(dx, dy,  3)
        n = n / (n.norm(2, 3, True) + 1e-8)

        return n

    def _change_perspective(self, pointcloud, cam_to_cam):
        """ Use the translation/rotation matrix to move
        a pointcloud between reference frames
        """

        # Use unsqueeze and squeeze to coax pytorch into
        # performing a matrix - vector product.
        pointcloud = (cam_to_cam @ pointcloud.unsqueeze(-1)).squeeze(-1)

        return pointcloud

    def _to_sample_grid(self, pointcloud, cam):
        # Project the pointcloud onto a projection
        # plane using the camera matrix
        grid = (cam @ pointcloud.unsqueeze(-1)).squeeze(-1)

        # Now grid has shape (n, h, w, 3).
        # Each pixel contains a 3-dimensional homogeneous coordinate.
        # To get to x,y coordinates the first two elements of
        # the homogeneous coordinates have to be divided by the third.
        grid = grid[:,:,:,:2] / (grid[:,:,:,2:3] + 1e-7)

        # At this point grid contains sampling coordinates in pixels
        # but grid_sample works with sampling coordinates in the range -1,1
        # so some rescaling has to be applied.
        h, w = grid.shape[1:3]
        dim = torch.tensor((w - 1, h - 1), dtype=grid.dtype, device=grid.device)
        grid = 2 * grid / dim - 1

        return grid

    def _shape_cam(self, inv_cam, cam):
        # We only need the top 3x4 of the camera matrix for projection
        # from pointcloud to homogeneous grid positions
        cam = cam[:,:3,:]

        # We only need the top 3x3 of the inverse matrix
        # for projection to pointcloud
        inv_cam = inv_cam[:,:3,:3]

        # Take the camera matrix from shape (N, 3, 4) to
        # (N, 1, 1, 3, 4) to match the pointcloud shape (N, H, W, 4).
        cam = cam.unsqueeze(1).unsqueeze(2)

        # Take the inverse camera matrix from shape (N, 3, 3)
        # to (N, 1, 1, 3, 3) to match the grid shape (N, H, W, 3).
        inv_cam = inv_cam.unsqueeze(1).unsqueeze(2)

        return inv_cam, cam

    def scale_disp(self, disp):
        return self.min_disp + (self.max_disp - self.min_disp) * disp

    def warp_images(self, inputs, outputs, outputs_masked):
        """Generate the warped (reprojected) color images for a minibatch.
        """

        predictions = dict()
        resolutions = tuple(frozenset(k[1] for k in outputs if k[0] == 'disp'))
        frame_ids = tuple(frozenset(k[2] for k in outputs if k[0] == 'cam_T_cam'))

        inv_cam, cam = self._shape_cam(inputs["inv_K", 0], inputs["K", 0])

        disps = tuple(
            functional.interpolate(
                outputs["disp", res], scale_factor=2**res,
                mode="bilinear", align_corners=False
            )
            for res in resolutions
        )

        depths = tuple(
            1 / self.scale_disp(disp)
            for disp in disps
        )

        # Take the pixel position in the target image and
        # the estimated depth to generate a point cloud of
        # target image pixels.
        pointclouds_source = tuple(
            self._to_pointcloud(depth, inv_cam)
            for depth in depths
        )

        # Calculate per-pixel surface normals
        surface_normals = tuple(
            self._surface_normal(pointcloud)
            for pointcloud in pointclouds_source
        )

        for frame_id in frame_ids:
            img_source = inputs["color", frame_id, 0]
            cam_to_cam = outputs["cam_T_cam", 0, frame_id]

            # Transfer the estimated pointclouds from one frame-of
            # reference to the other
            pointclouds_target = tuple(
                self._change_perspective(pointcloud, cam_to_cam)
                for pointcloud in pointclouds_source
            )

            # Using the projection matrix, map this point cloud
            # to expected pixel coordinates in the source image.
            grids = tuple(
                self._to_sample_grid(pointcloud, cam)
                for pointcloud in pointclouds_target
            )

            # Construct warped target images by sampling from the source image
            for res, grid in zip(resolutions, grids):
                # TODO check which align corners behaviour is desired, default is false, but originally it used to
                #  be true, also for segmentation warping, define grid_sample for 1.1.0 version and for newest version
                img_pred = torch.nn.functional.grid_sample(img_source, grid, padding_mode="border")
                predictions["sample", frame_id, res] = grid
                predictions["color", frame_id, res] = img_pred

            # sample the warped segmentation image
            if outputs_masked is not None:
                shape = outputs_masked[("segmentation", frame_id, 0)].shape
                seg_source = outputs_masked[("segmentation", frame_id, 0)].reshape(
                    (shape[0], 1, shape[1], shape[2]))
                seg_pred = torch.nn.functional.grid_sample(seg_source.float(), grids[0], padding_mode="border",
                                                           mode='nearest').reshape((shape[0], shape[1], shape[2]))
                outputs_masked["segmentation_warped", frame_id, 0] = seg_pred

        for res, depth, surface_normal in zip(resolutions, depths, surface_normals):
            predictions["depth", 0, res] = depth
            predictions["normals_pointcloud", res] = surface_normal

        return predictions
