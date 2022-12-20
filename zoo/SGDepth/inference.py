import time
from models.sgdepth import SGDepth
import torch
from arguments import InferenceEvaluationArguments
import cv2
import os
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import glob as glob


DEBUG = False  # if this flag is set the images are displayed before being saved


class Inference:
    """Inference without harness or dataloader"""

    def __init__(self):
        self.model_path = opt.model_path
        self.image_dir = opt.image_path
        self.image_path = opt.image_path
        self.num_classes = 20
        self.depth_min = opt.model_depth_min
        self.depth_max = opt.model_depth_max
        self.output_path = opt.output_path
        self.output_format = opt.output_format
        self.all_time = []
        # try:
        #     self.checkpoint_path = os.environ['IFN_DIR_CHECKPOINT']
        # except KeyError:
        #     print('No IFN_DIR_CHECKPOINT defined.')

        self.labels = (('CLS_ROAD', (128, 64, 128)),
                       ('CLS_SIDEWALK', (244, 35, 232)),
                       ('CLS_BUILDING', (70, 70, 70)),
                       ('CLS_WALL', (102, 102, 156)),
                       ('CLS_FENCE', (190, 153, 153)),
                       ('CLS_POLE', (153, 153, 153)),
                       ('CLS_TRLIGHT', (250, 170, 30)),
                       ('CLS_TRSIGN', (220, 220, 0)),
                       ('CLS_VEGT', (107, 142, 35)),
                       ('CLS_TERR', (152, 251, 152)),
                       ('CLS_SKY', (70, 130, 180)),
                       ('CLS_PERSON', (220, 20, 60)),
                       ('CLS_RIDER', (255, 0, 0)),
                       ('CLS_CAR', (0, 0, 142)),
                       ('CLS_TRUCK', (0, 0, 70)),
                       ('CLS_BUS', (0, 60, 100)),
                       ('CLS_TRAIN', (0, 80, 100)),
                       ('CLS_MCYCLE', (0, 0, 230)),
                       ('CLS_BCYCLE', (119, 11, 32)),
                       )

    def init_model(self):
        print("Init Model...")
        sgdepth = SGDepth

        with torch.no_grad():
            # init 'empty' model
            self.model = sgdepth(
                opt.model_split_pos, opt.model_num_layers, opt.train_depth_grad_scale,
                opt.train_segmentation_grad_scale,
                # opt.train_domain_grad_scale,
                opt.train_weights_init, opt.model_depth_resolutions, opt.model_num_layers_pose,
                # opt.model_num_domains,
                # opt.train_loss_weighting_strategy,
                # opt.train_grad_scale_weighting_strategy,
                # opt.train_gradnorm_alpha,
                # opt.train_uncertainty_eta_depth,
                # opt.train_uncertainty_eta_seg,
                # opt.model_shared_encoder_batchnorm_momentum
            )

            # load weights (copied from state manager)
            state = self.model.state_dict()
            to_load = torch.load(self.model_path)
            for (k, v) in to_load.items():
                if k not in state:
                    print(f"    - WARNING: Model file contains unknown key {k} ({list(v.shape)})")

            for (k, v) in state.items():
                if k not in to_load:
                    print(f"    - WARNING: Model file does not contain key {k} ({list(v.shape)})")

                else:
                    state[k] = to_load[k]

            self.model.load_state_dict(state)
            self.model = self.model.eval().cuda()  # for inference model should be in eval mode and on gpu

    def load_image(self):
        print("Load Image: " + self.image_path)

        self.image = Image.open(self.image_path)  # open PIL image
        self.image_o_width, self.image_o_height = self.image.size

        resize = transforms.Resize(
            (opt.inference_resize_height, opt.inference_resize_width))
        image = resize(self.image)  # resize to argument size

        #center_crop = transforms.CenterCrop((opt.inference_crop_height, opt.inference_crop_width))
        #image = center_crop(image)  # crop to input size

        to_tensor = transforms.ToTensor()  # transform to tensor

        self.input_image = to_tensor(image)  # save tensor image to self.input_image for saving later
        image = self.normalize(self.input_image)

        image = image.unsqueeze(0).float().cuda()

        # simulate structure of batch:
        image_dict = {('color_aug', 0, 0): image}  # dict
        image_dict[('color', 0, 0)] = image
        image_dict['domain'] = ['cityscapes_val_seg', ]
        image_dict['purposes'] = [['segmentation', ], ['depth', ]]
        image_dict['num_classes'] = torch.tensor([self.num_classes])
        image_dict['domain_idx'] = torch.tensor(0)
        self.batch = (image_dict,)  # batch tuple


    def normalize(self, tensor):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        normalize = transforms.Normalize(mean, std)
        tensor = normalize(tensor)

        return tensor


    def inference(self):
        self.init_model()
        print('Saving images to' + str(self.output_path) + ' in ' + str(self.output_format) + '\n \n')

        for image_path in glob.glob(self.image_dir + '/*'):
            self.image_path = image_path  # for output

            # load image and transform it in necessary batch format
            self.load_image()

            start = time.time()
            with torch.no_grad():
                output = self.model(self.batch) # forward pictures

            self.all_time.append(time.time() - start)
            start = 0

            disps_pred = output[0]["disp", 0] # depth results
            segs_pred = output[0]['segmentation_logits', 0] # seg results

            segs_pred = segs_pred.exp().cpu()
            segs_pred = segs_pred.numpy()  # transform preds to np array
            segs_pred = segs_pred.argmax(1)  # get the highest score for classes per pixel

            self.save_pred_to_disk(segs_pred, disps_pred) # saves results

        print("Done with all pictures in: " + str(self.output_path))
        print("\nAverage forward time for processing one Image (the first one excluded): ", np.average(self.all_time[1::]))

    def save_pred_to_disk(self, segs_pred, depth_pred):
        ## Segmentation visualization
        segs_pred = segs_pred[0]
        o_size = segs_pred.shape

        # init of seg image
        seg_img_array = np.zeros((3, segs_pred.shape[0], segs_pred.shape[1]))

        # create a color image from the classes for every pixel todo: probably a lot faster if vectorized with numpy
        i = 0
        while i < segs_pred.shape[0]:  # for row
            n = 0
            while n < segs_pred.shape[1]:  # for column
                lab = 0
                while lab < self.num_classes:  # for classes
                    if segs_pred[i, n] == lab:
                        # write colors to pixel
                        seg_img_array[0, i, n] = self.labels[lab][1][0]
                        seg_img_array[1, i, n] = self.labels[lab][1][1]
                        seg_img_array[2, i, n] = self.labels[lab][1][2]
                        break
                    lab += 1
                n += 1
            i += 1

        # scale the color values to 0-1 for proper visualization of OpenCV
        seg_img = seg_img_array.transpose(1, 2, 0).astype(np.uint8)
        seg_img = seg_img[:, :, ::-1 ]

        if DEBUG:
            cv2.imshow('segmentation', seg_img)
            cv2.waitKey()

        # Depth Visualization
        depth_pred = np.array(depth_pred[0][0].cpu())  # depth predictions to numpy and CPU

        depth_pred = self.scale_depth(depth_pred)  # Depthmap in meters
        depth_pred = depth_pred * (255 / depth_pred.max())  # Normalize Depth to 255 = max depth
        depth_pred = np.clip(depth_pred, 0, 255)  # Clip to 255 for safety
        depth_pred = depth_pred.astype(np.uint8)  # Cast to uint8 for openCV to display

        depth_img = cv2.applyColorMap(depth_pred, cv2.COLORMAP_PLASMA)  # Use PLASMA Colormap like in the Paper

        if DEBUG:
            cv2.imshow('depth', depth_img)
            cv2.waitKey()


        # Color_img
        color_img = np.array(self.image)
        # color_img = color_img.transpose((1, 2, 0))
        color_img = color_img[: ,: , ::-1]

        if DEBUG:
            cv2.imshow('color', color_img)
            cv2.waitKey()

        # resize depth and seg
        depth_img = cv2.resize(depth_img, (self.image_o_width, self.image_o_height))
        seg_img = cv2.resize(seg_img, (self.image_o_width, self.image_o_height), interpolation=cv2.INTER_NEAREST)

        # Concetenate all 3 pictures together
        conc_img = np.concatenate((color_img, seg_img, depth_img), axis=0)

        if DEBUG:
            cv2.imshow('conc', conc_img)
            cv2.waitKey()

        img_head, img_tail = os.path.split(self.image_path)
        img_name = img_tail.split('.')[0]
        print('Saving...')
        cv2.imwrite(str(self.output_path +'/' + img_name + self.output_format), conc_img)


    def scale_depth(self, disp):
        min_disp = 1 / self.depth_max
        max_disp = 1 / self.depth_min
        return min_disp + (max_disp - min_disp) * disp


if __name__ == "__main__":
    opt = InferenceEvaluationArguments().parse()

    infer = Inference()
    infer.inference()
