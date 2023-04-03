import numpy as np 
import os, pdb 
import shutil 
from tqdm import tqdm 
from glob import glob 
import argparse 
import copy


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="kitti_raw")
parser.add_argument("--data_path", type=str, default="/home/szha2609/data/kitti/kitti_raw")
parser.add_argument("--noise_type", type=str, required=True, choices=["velo", "gyro", "acc", "both"])
parser.add_argument("--noise_std", type=float, default=1.0, help="Gaussian standard error of the noises to replace the learnt ys")
parser.add_argument("--remove", action="store_true", help="if set, remove all generated noisy IMU files")


def add_noise_kitti(opt):
    """Add IMU to KITTI matched IMU data

    Args:
        opt (dict): --dataset --data_path
    """
    Id3 = np.eye(3)
    gyro_std = [1.0, 1.0, 1.0]
    # gyro_std = [x * np.pi / 180 for x in gyro_std]
    # acc_std = [0.05, 0.05, 0.05]
    acc_std = [1.0, 1.0, 1.0]
    velo_std = [1.0, 1.0, 1.0]
    
    scenes = [x for x in glob(os.path.join(opt.data_path, "./*")) if "2011" in x.split("/")[-1]]
    for scene in scenes:
        seqs = [x for x in glob(os.path.join(scene, "./*")) if "2011" in x.split("/")[-1]]
        for seq in seqs:
            print("=============================")
            print("=> processing {}...".format(seq))
            imu_file = os.path.join(seq, "matched_oxts/matched_oxts.txt")
            if not os.path.isfile(imu_file):
                print("=> ERROR: No matched_oxts.txt found. Skip...")
                continue
            outputs = []
            with open(imu_file, 'r') as f:
                for line in f.readlines():
                    if "nan" in line or "none" in line:
                        outputs.append(line)
                        continue 
                    oxts = []
                    # gyro/acc_RotErrs are the same for IMUs between two images
                    # while N(0, std) ys_gyro/acc are generated for each IMU
                    Id3 = np.eye(3)
                    gyro_RotErr = 0.05 * np.random.normal(0, 1, (3, 3))
                    acc_RotErr = 0.05 * np.random.normal(0, 1, (3, 3))
                    for datum in line.split('|'):
                        datum = copy.deepcopy(datum).strip().split(' ')
                        w_xyz = np.array([float(x) for x in datum[17:20]])
                        a_xyz = np.array([float(x) for x in datum[11:14]])
                        v_flu = np.array([float(x) for x in datum[8:11]])
                        # Add noise to gyro
                        if opt.noise_type in ["gyro", "both"]:
                            ys_gyro = np.random.normal(0, opt.noise_std, w_xyz.shape)
                            w_xyz = (Id3 + gyro_RotErr) @ w_xyz + gyro_std * ys_gyro
                            
                        # Add noise to acc
                        if opt.noise_type in ["acc", "both"]:
                            ys_acc = np.random.normal(0, opt.noise_std, a_xyz.shape)
                            a_xyz = (Id3 + acc_RotErr) @ a_xyz + acc_std * ys_acc
                        
                        # Add noise to velo
                        if opt.noise_type in ["velo"]:
                            ys_v = np.random.normal(0, opt.noise_std, v_flu.shape)
                            v_flu = v_flu + velo_std * ys_v
                        
                        datum[17:20] = [str(x) for x in w_xyz]
                        datum[11:14] = [str(x) for x in a_xyz]
                        datum[8:11]  = [str(x) for x in v_flu]
                        oxts.append(' '.join(datum))
                    outputs.append(' | '.join(oxts) + '\n')
                    
            noisy_imu_file = os.path.join(seq, "matched_oxts/matched_oxts_{}_{}.txt".format(opt.noise_type, opt.noise_std))
            with open(noisy_imu_file, 'w') as fo:
                for line in outputs:
                    fo.write(line)


def remove(opt):
    scenes = [x for x in glob(os.path.join(opt.data_path, "./*")) if "2011" in x.split("/")[-1]]
    for scene in scenes:
        seqs = [x for x in glob(os.path.join(scene, "./*")) if "2011" in x.split("/")[-1]]
        for seq in seqs:
            print("=============================")
            print("=> removing {}...".format(seq))
            imu_files = glob(os.path.join(seq, "matched_oxts/matched_oxts_*"))
            for f_ in imu_files:
                if os.path.isfile(f_):
                    os.remove(f_)
                    

if __name__ == "__main__":
    opt = parser.parse_args()
    if opt.remove:
        remove(opt)
    else:
        if opt.dataset == "kitti_raw":
            add_noise_kitti(opt)
        else:
            raise ValueError()
        