import os
import shutil
import PIL
from PIL import Image
import numpy as np
from tqdm import tqdm


def check_imu():
    """
    check the timestamp match between raw_oxts and sync_oxts
    -> also done in match_imu() where more filtering are performed
    -> results are written to check_kitti_imu.log
    """
    msgs = []
    seqs = []
    for mode in ["train", "val"]:
        with open("kitti_extract/{}.txt".format(mode), 'r') as f:
            for line in f.readlines():
                if line.strip() not in seqs:
                    seqs.append(line.strip()) # e.g. "2011_09_30_drive_0028"
    with open("test_scenes.txt", 'r') as f:
            for line in f.readlines():
                if line.strip() not in seqs:
                    seqs.append(line.strip()) # e.g. "2011_09_30_drive_0028"
                    
    for seq in seqs:
        msgs.append('=========================')
        msgs.append('processing seq {}'.format(seq))
        print('=========================')
        print('processing seq {}'.format(seq))

        scene = "_".join(seq.split("_")[:3]) # "2011_09_30"
        
        raw_dir = "kitti_extract/{}/{}_extract".format(scene, seq)
        sync_dir  = "kitti/kitti_raw/{}/{}_sync".format(scene, seq)
        raw_time = []
        sync_time = []
        with open('{}/oxts/timestamps.txt'.format(raw_dir), mode='r') as f:
            for _line in tqdm(f.readlines()):
                raw_time.append(_line[:-1])
        with open('{}/oxts/timestamps.txt'.format(sync_dir), mode='r') as f:
            for _line in tqdm(f.readlines()):
                sync_time.append(_line[:-1])
        
        missing_num = 0
        for _i, _sync in tqdm(enumerate(sync_time)):
            if _sync not in raw_time:
                msgs.append('Not found: [{}] {}'.format(_i, _sync))
                missing_num += 1
        print('--------------------------------')
        print('total number of timestamps: {}'.format(len(sync_time)))
        print('the number of not found: {}'.format(missing_num))
        msgs.append('--------------------------------')
        msgs.append('total number of timestamps: {}'.format(len(sync_time)))
        msgs.append('the number of not found: {}'.format(missing_num))

    with open('check_kitti_imu.log', mode='w') as f:
        for _msg in msgs:
            print(_msg)
            f.write('{}\n'.format(_msg))
    

def match_imu():
    """
    match imu from raw_oxts to sync_oxts in sync/00/matched_oxts
    -> results are written to match_kitti_imu.log
    """
    msgs = []
    seqs = []
    for mode in ["train", "val"]:
        with open("kitti_extract/{}.txt".format(mode), 'r') as f:
            for line in f.readlines():
                if line.strip() not in seqs:
                    seqs.append(line.strip()) # e.g. "2011_09_30_drive_0028"
    with open("test_scenes.txt", 'r') as f:
            for line in f.readlines():
                if line.strip() not in seqs:
                    seqs.append(line.strip()) # e.g. "2011_09_30_drive_0028"
                    
    for seq in seqs:
        msgs.append('=========================')
        msgs.append('processing seq {}'.format(seq))
        print('=========================')
        print('processing seq {}'.format(seq))

        scene = "_".join(seq.split("_")[:3]) # "2011_09_30"
        
        raw_dir = "kitti_extract/{}/{}_extract".format(scene, seq)
        sync_dir  = "kitti/kitti_raw/{}/{}_sync".format(scene, seq)

        raw_oxt   = []
        sync_oxt  = []
        raw_num   = len([x for x in os.listdir('{}/oxts/data/'.format(raw_dir)) if x[-4:] == '.txt'])
        oxt_num   = len([x for x in os.listdir('{}/oxts/data/'.format(sync_dir)) if x[-4:] == '.txt'])
        for _i in tqdm(range(raw_num)):
            with open('{}/oxts/data/{:010d}.txt'.format(raw_dir, _i), mode='r') as f:
                tmp_list = []
                for _line in f.readlines():
                    if _line[-1] == '\n':
                        _line = _line[:-1]
                    tmp_list.append(_line)
                assert len(tmp_list) == 1
                raw_oxt.append(tmp_list[0])
        for _j in tqdm(range(oxt_num)):
            with open('{}/oxts/data/{:010d}.txt'.format(sync_dir, _j), mode='r') as f:
                tmp_list = []
                for _line in f.readlines():
                    if _line[-1] == '\n':
                        _line = _line[:-1]
                    tmp_list.append(_line)
                assert len(tmp_list) == 1
                sync_oxt.append(tmp_list[0])
        
        raw_time  = []
        sync_time = []
        with open('{}/oxts/timestamps.txt'.format(raw_dir), mode='r') as f:
            for _line in tqdm(f.readlines()):
                if _line[-1] == '\n':
                    _line = _line[:-1]
                raw_time.append(_line)
        with open('{}/oxts/timestamps.txt'.format(sync_dir), mode='r') as f:
            for _line in tqdm(f.readlines()):
                if _line[-1] == '\n':
                    _line = _line[:-1]
                sync_time.append(_line)
        
        assert len(raw_oxt) == len(raw_time)
        assert len(sync_oxt) == len(sync_time)

        matched_oxt = []
        matched_timestamp = []
        imu_num = []
        for _i in tqdm(range(len(sync_time) - 1)):
            if sync_time[_i] not in raw_time or sync_time[_i + 1] not in raw_time:
                matched_oxt.append('nan')
                matched_timestamp.append('nan')
                imu_num.append('nan')
            else:
                raw_ind_0 = raw_time.index(sync_time[_i])
                raw_ind_1 = raw_time.index(sync_time[_i + 1])
                no_outlier = True
                for _k in range(raw_ind_0, raw_ind_1, 1):
                    if no_outlier:
                        # check whether there is a jump in timestamps in raw_imu
                        tmp_time_0 = raw_time[_k]
                        tmp_time_0 = [float(x) for x in tmp_time_0.split(' ')[1].split(':')]
                        tmp_time_0 = 3600 * tmp_time_0[0] + 60 * tmp_time_0[1] + tmp_time_0[2]
                        tmp_time_1 = raw_time[_k + 1]
                        tmp_time_1 = [float(x) for x in tmp_time_1.split(' ')[1].split(':')]
                        tmp_time_1 = 3600 * tmp_time_1[0] + 60 * tmp_time_1[1] + tmp_time_1[2]
                        tdiff = tmp_time_1 - tmp_time_0 # should be 0.01 (100Hz for raw_data)
                        if abs(tdiff - 0.01) > 0.005: # allow for a 5ms drift
                            no_outlier = False
                            matched_oxt.append('nan')
                            matched_timestamp.append('nan')
                            imu_num.append('nan')
                            msgs.append('warning: raw_time[{}] and [{}] failed for drift check (tdiff: {})'.format(_k, _k+1, tdiff))

                if no_outlier:
                    tmp_gap = raw_ind_1 - raw_ind_0 + 1
                    if tmp_gap < 11:
                        matched_oxt.append('nan')
                        matched_timestamp.append('nan')
                        imu_num.append('nan')
                        msgs.append('warning: imu_gap: {} for raw_time[{}] and [{}]'.format(tmp_gap, raw_ind_0, raw_ind_1))
                    else:
                        matched_oxt.append(' | '.join(raw_oxt[raw_ind_0:raw_ind_1 + 1]))
                        matched_timestamp.append(' | '.join(raw_time[raw_ind_0:raw_ind_1 + 1]))
                        imu_num.append(tmp_gap)

        msgs.append('-------------------------')
        msgs.append('total number of items: {}'.format(len(matched_oxt)))
        msgs.append('summarization of imu nums between two images:')
        for _s in set(imu_num):
            msgs.append('-> {}: {}'.format(_s, imu_num.count(_s)))
        
        path_matched_oxt = '{}/matched_oxts'.format(sync_dir)
        if os.path.isdir(path_matched_oxt):
            shutil.rmtree(path_matched_oxt)
        os.mkdir(path_matched_oxt)
        os.mkdir('{}/data'.format(path_matched_oxt))

        with open('{}/matched_timestamps.txt'.format(path_matched_oxt), mode='w') as f:
            for _time in matched_timestamp:
                f.write('{}\n'.format(_time))
        
        with open('{}/matched_oxts.txt'.format(path_matched_oxt), mode='w') as f:
            for _oxt in matched_oxt:
                f.write('{}\n'.format(_oxt))
        
        for _i, _oxt in enumerate(matched_oxt):
            with open('{}/data/{:010d}.txt'.format(path_matched_oxt, _i), mode='w') as f:
                f.write(_oxt)
        
        with open('match_kitti_imu.log', mode='w') as f:
            for _msg in msgs:
                print(_msg)
                f.write('{}\n'.format(_msg))

        
def prepare_kitti():
    """
    match 100 hz imu data from raw_data to the dataset released in kitti odometry leaderboard
    requires the download of 
    (1) sync dataset from the kitti raw data website   -> need image_02/ (10 hz) and oxts/ (10 hz) for matching
    (2) unsync dataset from the kitti raw data website -> need oxts/ (100 hz) for matching  
    """
    match_imu() # generate matched_oxts.txt and matched_timestamps for sync_data



if __name__ == "__main__":
   # check_imu()
   prepare_kitti()
