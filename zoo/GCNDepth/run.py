import os

if __name__ == '__main__':

    os.system('python -m torch.distributed.launch --master_port=9900 --nproc_per_node=1 train.py --config ./config/cfg_kitti_fm.py --work_dir results')
    
