import argparse
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()


        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default="kitti_data")
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(PROJECT_DIR, "tmp"))
        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 default='full_res18_192x640',
                                 help="the name of the folder to save the model in",
                                 )

        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark", "test"],
                                 default="eigen_zhou")
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test"])
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--reprojection",
                                 default=1.0,
                                 type=float)
        self.parser.add_argument("--pretrained",
                                 type=int,
                                 default=1,
                                 help='use ImageNet pretrained weight for ResNet encoder')
        # OPTIMIZATION options

        # Please use two gpus to set total batch size as 12, or use one gpu and set change option into 12
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size for a single gpu",
                                 default=6)

        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1.5e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)

        # 0.3 for training the full model. set this option as 1.0 for training (only SGT) or (only SGT).
        self.parser.add_argument("--semantic_distil",
                                 type=float,
                                 default=0.3,
                                 help='weight factor of CE loss for training semantic segmentation')

        self.parser.add_argument("--auto_mask",
                                 action="store_true",
                                 default=True)

        self.parser.add_argument("--min_reprojection",
                                 action='store_true',
                                 default=True)
        self.parser.add_argument("--lr_decay",
                                 nargs='+',
                                 type=int,
                                 default=[10, 15])
        self.parser.add_argument("--decay_rate",
                                 type=float, default=0.1)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")

        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        # EVALUATION options
        self.parser.add_argument("--eval_stereo", help="if set evaluates in stereo mode", action="store_true")
        self.parser.add_argument("--eval_mono", help="if set evaluates in mono mode", action="store_true")
        self.parser.add_argument("--disable_median_scaling", help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor", help="if set multiplies predictions by this number",
                                 type=float, default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 choices=[
                                     "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10"],
                                 help="which split to run eval on")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")

        self.parser.add_argument("--local_rank", type=int, default=0)

        # Semantics-guided Triplet Loss options
        self.parser.add_argument("--sgt", type=float, default=0.1, help='weight factor for sgt loss')
        self.parser.add_argument("--sgt_layers", nargs='+', type=int, default=[3, 2, 1],
                                 help='layer configurations for sgt loss')
        self.parser.add_argument("--sgt_margin", type=float, default=0.3, help='margin for sgt loss')
        self.parser.add_argument("--sgt_kernel_size", type=int, nargs='+', default=[5, 5, 5],
                                 help='kernel size (local patch size) for sgt loss')

        # Corss-task Multi-embedding Module options
        self.parser.add_argument("--no_cma", action='store_true', default=False, help='disable cma module')
        self.parser.add_argument("--num_head", type=int, default=4, help='number of embeddings H for cma module')
        self.parser.add_argument("--head_ratio", type=float, default=2, help='embedding dimension ratio for cma module')
        self.parser.add_argument("--cma_layers", nargs="+", type=int, default=[3, 2, 1],
                                 help='layer configurations for cma module')

        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
