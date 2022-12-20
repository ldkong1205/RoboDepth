import os
import argparse
from model import Model

file_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser(description="EPCDepth args")

# data
parser.add_argument("--data_path", type=str, help="path to the data")
parser.add_argument("--img_height", type=int, help="input image height", default=320)
parser.add_argument("--img_width", type=int, help="input image width", default=1024)

# device
parser.add_argument("--gpu_id", type=str, default='0')
parser.add_argument("--no_cuda", help="if set disables CUDA", action="store_true")

# training
parser.add_argument("--start_epoch", type=int, help="start epoch while train", default=0)
parser.add_argument("--logs_dir", type=str, help="path to save the logs", default=os.path.join(file_dir, "logs"))
parser.add_argument("--models_dir", type=str, help="path to save the checkpoints", default=os.path.join(file_dir, "models"))
parser.add_argument("--split", type=str, help="which training split to use", choices=["eigen_full"], default="eigen_full")
parser.add_argument("--num_layers", type=int, help="number of resnet layers", default=18, choices=[18, 34, 50, 101, 152])
parser.add_argument("--batch_size", type=int, help="batch size", default=8)
parser.add_argument("--epochs", type=int, help="number of epochs", default=20)
parser.add_argument("--num_workers", type=int, help="number of dataloader workers", default=4)
parser.add_argument("--learning_rate", type=float, help="learning rate", default=1e-4)
parser.add_argument("--scheduler_step_size", type=int, help="step size of the scheduler", default=15)
parser.add_argument("--disparity_smoothness", type=float, help="disparity smoothness weight", default=0)
parser.add_argument("--spp_loss", type=float, default=1)
parser.add_argument("--disable_automasking", help="if set, doesn't do auto-masking", action="store_true")
parser.add_argument("--use_data_graft", action="store_true")
parser.add_argument("--use_spp_distillate", action="store_true")
parser.add_argument("--use_full_scale", action="store_true")
parser.add_argument("--use_depth_hint", help="if use depth hint", action="store_true")
parser.add_argument("--pretrained", help="if use pretrained encoder", action="store_true")

# validate
parser.add_argument("--output_scale", type=int, help="output disparity scale, -1 for mean", default=0, choices=[0, 1, 2, -1])
parser.add_argument("--resume", type=str, help="path of models to resume", metavar="PATH", default="")
parser.add_argument("--min_depth", type=float, help="minimum depth", default=0.1)
parser.add_argument("--max_depth", type=float, help="maximum depth", default=100.0)
parser.add_argument("--val_split", type=str, default="eigen", choices=["eigen"])
parser.add_argument("--val", action="store_true")
parser.add_argument("--post_process", action="store_true")
parser.add_argument("--vis", action="store_true", help="visualization")
parser.add_argument("--disps_path", type=str, help="path to the disparity")


args = parser.parse_args()

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    model = Model(args)
    model.main()
