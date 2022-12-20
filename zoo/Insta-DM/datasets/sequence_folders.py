'''
Seokju Lee

(+) added customized outputs: flow_fwd, flow_bwd, segmentation mask (src/tgt), instance mask (src/tgt)
(+) added recursive_check_nonzero_inst

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random
import math
from matplotlib import pyplot as plt
from flow_io import flow_read
from rigid_warp import flow_warp
import pdb


def load_as_float(path):
    return imread(path).astype(np.float32)


def load_flo_as_float(path):
    out = np.array(flow_read(path)).astype(np.float32)    
    return out


def load_seg_as_float(path):
    out = np.load(path).astype(np.float32)
    return out


def L2_norm(x, dim=1, keepdim=True):
    curr_offset = 1e-10
    l2_norm = torch.norm(torch.abs(x) + curr_offset, dim=dim, keepdim=True)
    return l2_norm


def find_noc_masks(fwd_flow, bwd_flow):
    '''
    fwd_flow: torch.size([1, 2, 256, 832])
    bwd_flow: torch.size([1, 2, 256, 832])
    output: torch.size([1, 1, 256, 832]), torch.size([1, 1, 256, 832])

    input shape of flow_warp(): torch.size([bs, 2, 256, 832])
    '''
    bwd2fwd_flow, _ = flow_warp(bwd_flow, fwd_flow)
    fwd2bwd_flow, _ = flow_warp(fwd_flow, bwd_flow)

    fwd_flow_diff = torch.abs(bwd2fwd_flow + fwd_flow)
    bwd_flow_diff = torch.abs(fwd2bwd_flow + bwd_flow)

    fwd_consist_bound = torch.max(0.05 * L2_norm(fwd_flow), torch.Tensor([3.0]))
    bwd_consist_bound = torch.max(0.05 * L2_norm(bwd_flow), torch.Tensor([3.0]))

    noc_mask_0 = (L2_norm(fwd_flow_diff) < fwd_consist_bound).type(torch.FloatTensor)     # noc_mask_tgt, torch.Size([1, 1, 256, 832]), torch.float32
    noc_mask_1 = (L2_norm(bwd_flow_diff) < bwd_consist_bound).type(torch.FloatTensor)     # noc_mask_src, torch.Size([1, 1, 256, 832]), torch.float32
    # pdb.set_trace()

    return noc_mask_0, noc_mask_1


def inst_iou(seg_src, seg_tgt, valid_mask):
    '''
    => Which channel instance of seg_tgt does the instances of seg_src match?
    => seg_src의 인스턴스들이 seg_tgt의 몇 번째 채널 인스턴스에 매칭되는가?

    seg_src: torch.Size([1, n_inst, 256, 832])
    seg_tgt:  torch.Size([1, n_inst, 256, 832])
    valid_mask: torch.Size([1, 1, 256, 832])
    '''
    n_inst_src = seg_src.shape[1]
    n_inst_tgt = seg_tgt.shape[1]

    seg_src_m = seg_src * valid_mask.repeat(1,n_inst_src,1,1)
    seg_tgt_m = seg_tgt * valid_mask.repeat(1,n_inst_tgt,1,1)
    # pdb.set_trace()
    '''
    plt.figure(1), plt.imshow(seg_src.sum(dim=0).sum(dim=0)), plt.colorbar(), plt.ion(), plt.show()
    plt.figure(2), plt.imshow(seg_tgt.sum(dim=0).sum(dim=0)),  plt.colorbar(), plt.ion(), plt.show()
    plt.figure(3), plt.imshow(valid_mask[0,0]),  plt.colorbar(), plt.ion(), plt.show()
    plt.figure(4), plt.imshow(seg_src_m.sum(dim=0).sum(dim=0)),  plt.colorbar(), plt.ion(), plt.show()
    '''
    for i in range(n_inst_src):
        if i == 0: 
            match_table = torch.from_numpy(np.zeros([1,n_inst_tgt]).astype(np.float32))
            continue;

        overl = (seg_src_m[:,i].unsqueeze(1).repeat(1,n_inst_tgt,1,1) * seg_tgt_m).clamp(min=0,max=1).squeeze(0).sum(1).sum(1)
        union = (seg_src_m[:,i].unsqueeze(1).repeat(1,n_inst_tgt,1,1) + seg_tgt_m).clamp(min=0,max=1).squeeze(0).sum(1).sum(1)

        iou_inst = overl / union
        match_table = torch.cat((match_table, iou_inst.unsqueeze(0)), dim=0)

    iou, inst_idx = torch.max(match_table,dim=1)
    # pdb.set_trace()

    return iou, inst_idx


def recursive_check_nonzero_inst(tgt_inst, ref_inst):
    assert( tgt_inst[0].mean() == ref_inst[0].mean() )
    n_inst =  int(tgt_inst[0].mean())
    for nn in range(n_inst):
        if tgt_inst[nn+1].mean() == 0:
            tgt_inst[0] -= 1
            ref_inst[0] -= 1
            if nn+1 == n_inst:
                tgt_inst[nn+1:] = 0
                ref_inst[nn+1:] = 0
            else:
                tgt_inst[nn+1:] = torch.cat([tgt_inst[nn+2:], torch.zeros(1, tgt_inst.size(1), tgt_inst.size(2))], dim=0)    # re-ordering
                ref_inst[nn+1:] = torch.cat([ref_inst[nn+2:], torch.zeros(1, ref_inst.size(1), ref_inst.size(2))], dim=0)    # re-ordering
            return recursive_check_nonzero_inst(tgt_inst, ref_inst)
        if ref_inst[nn+1].mean() == 0:
            tgt_inst[0] -= 1
            ref_inst[0] -= 1
            if nn+1 == n_inst:
                tgt_inst[nn+1:] = 0
                ref_inst[nn+1:] = 0
            else:
                tgt_inst[nn+1:] = torch.cat([tgt_inst[nn+2:], torch.zeros(1, tgt_inst.size(1), tgt_inst.size(2))], dim=0)    # re-ordering
                ref_inst[nn+1:] = torch.cat([ref_inst[nn+2:], torch.zeros(1, ref_inst.size(1), ref_inst.size(2))], dim=0)    # re-ordering
            return recursive_check_nonzero_inst(tgt_inst, ref_inst)
    return tgt_inst, ref_inst


class SequenceFolder(data.Dataset):
    """
    A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)

    """

    def __init__(self, root, train, seed=None, shuffle=True, max_num_instances=20, sequence_length=3, transform=None, proportion=1, begin_idx=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/'image'/folder[:-1] for folder in open(scene_list_path)]
        self.is_shuffle = shuffle
        self.crawl_folders(sequence_length)
        self.mni = max_num_instances
        self.transform = transform
        split_index = int(math.floor(len(self.samples)*proportion))
        self.samples = self.samples[:split_index]
        if begin_idx: 
            self.samples = self.samples[begin_idx:]
        # pdb.set_trace()
        

    def crawl_folders(self, sequence_length):
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length, demi_length + 1))
        shifts.pop(demi_length)
        for scene in self.scenes:
            sceneff = Path(Path.dirname(scene).parent+'/flow_f/'+scene.split('/')[-1])
            scenefb = Path(Path.dirname(scene).parent+'/flow_b/'+scene.split('/')[-1])
            scenei = Path(Path.dirname(scene).parent+'/segmentation/'+scene.split('/')[-1])
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))

            imgs = sorted(scene.files('*.jpg'))
            flof = sorted(sceneff.files('*.flo'))   # 00: src, 01: tgt
            flob = sorted(scenefb.files('*.flo'))   # 00: tgt, 01: src
            segm = sorted(scenei.files('*.npy')) 

            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length, len(imgs)-demi_length):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': [],
                          'flow_fs':[], 'flow_bs':[], 'tgt_seg':segm[i], 'ref_segs':[]}   # ('tgt_insts':[], 'ref_insts':[]) will be processed when getitem() is called
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                    sample['ref_segs'].append(segm[i+j])
                for j in range(-demi_length, 1):
                    sample['flow_fs'].append(flof[i+j])
                    sample['flow_bs'].append(flob[i+j])
                sequence_set.append(sample)
            # pdb.set_trace()
        if self.is_shuffle:
            random.shuffle(sequence_set)
        self.samples = sequence_set


    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        
        flow_fs = [torch.from_numpy(load_flo_as_float(flow_f)) for flow_f in sample['flow_fs']]
        flow_bs = [torch.from_numpy(load_flo_as_float(flow_b)) for flow_b in sample['flow_bs']]

        tgt_seg = torch.from_numpy(load_seg_as_float(sample['tgt_seg']))
        ref_segs = [torch.from_numpy(load_seg_as_float(ref_seg)) for ref_seg in sample['ref_segs']]

        tgt_sort = torch.cat([torch.zeros(1).long(), tgt_seg.sum(dim=(1,2)).argsort(descending=True)[:-1]], dim=0)
        ref_sorts = [ torch.cat([torch.zeros(1).long(), ref_seg.sum(dim=(1,2)).argsort(descending=True)[:-1]], dim=0) for ref_seg in ref_segs ]
        tgt_seg = tgt_seg[tgt_sort]
        ref_segs = [ref_seg[ref_sort] for ref_seg, ref_sort in zip(ref_segs, ref_sorts)]

        tgt_insts = []
        ref_insts = []

        for i in range( len(ref_imgs) ):
            noc_f, noc_b = find_noc_masks(flow_fs[i].unsqueeze(0), flow_bs[i].unsqueeze(0))

            if i < len(ref_imgs)/2:     # first half            
                seg0 = ref_segs[i].unsqueeze(0)
                seg1 = tgt_seg.unsqueeze(0)
            else:                       # second half
                seg0 = tgt_seg.unsqueeze(0)
                seg1 = ref_segs[i].unsqueeze(0)

            seg0w, _ = flow_warp(seg1, flow_fs[i].unsqueeze(0))
            seg1w, _ = flow_warp(seg0, flow_bs[i].unsqueeze(0))

            n_inst0 = seg0.shape[1]
            n_inst1 = seg1.shape[1]

            ### Warp seg0 to seg1. Find IoU between seg1w and seg1. Find the maximum corresponded instance in seg1.
            iou_01, ch_01 = inst_iou(seg1w, seg1, valid_mask=noc_b)
            iou_10, ch_10 = inst_iou(seg0w, seg0, valid_mask=noc_f)

            seg0_re = torch.zeros(self.mni+1, seg0.shape[2], seg0.shape[3])
            seg1_re = torch.zeros(self.mni+1, seg1.shape[2], seg1.shape[3])
            non_overlap_0 = torch.ones([seg0.shape[2], seg0.shape[3]])
            non_overlap_1 = torch.ones([seg0.shape[2], seg0.shape[3]])

            num_match = 0
            for ch in range(n_inst0):
                condition1 = (ch == ch_10[ch_01[ch]]) and (iou_01[ch] > 0.5) and (iou_10[ch_01[ch]] > 0.5)
                condition2 = ((seg0[0,ch] * non_overlap_0).max() > 0) and ((seg1[0,ch_01[ch]] * non_overlap_1).max() > 0)
                if condition1 and condition2 and (num_match < self.mni): # matching success!
                    num_match += 1
                    seg0_re[num_match] = seg0[0,ch] * non_overlap_0
                    seg1_re[num_match] = seg1[0,ch_01[ch]] * non_overlap_1
                    non_overlap_0 = non_overlap_0 * (1 - seg0_re[num_match])
                    non_overlap_1 = non_overlap_1 * (1 - seg1_re[num_match])
            seg0_re[0] = num_match
            seg1_re[0] = num_match
            # pdb.set_trace()

            if seg0_re[0].mean() != 0 and seg0_re[int(seg0_re[0].mean())].mean() == 0: pdb.set_trace()
            if seg1_re[0].mean() != 0 and seg1_re[int(seg1_re[0].mean())].mean() == 0: pdb.set_trace()
            
            if i < len(ref_imgs)/2:     # first half
                tgt_insts.append(seg1_re.detach().cpu().numpy().transpose(1,2,0))
                ref_insts.append(seg0_re.detach().cpu().numpy().transpose(1,2,0))
            else:                       # second half
                tgt_insts.append(seg0_re.detach().cpu().numpy().transpose(1,2,0))
                ref_insts.append(seg1_re.detach().cpu().numpy().transpose(1,2,0))
        # pdb.set_trace()
        '''
        plt.close('all')
        plt.figure(1), plt.imshow(tgt_insts[0].sum(dim=0)), plt.grid(linestyle=':', linewidth=0.4), plt.colorbar(), plt.ion(), plt.show()
        plt.figure(2), plt.imshow(tgt_insts[1].sum(dim=0)), plt.grid(linestyle=':', linewidth=0.4), plt.colorbar(), plt.ion(), plt.show()
        plt.figure(3), plt.imshow(ref_insts[0].sum(dim=0)), plt.grid(linestyle=':', linewidth=0.4), plt.colorbar(), plt.ion(), plt.show()
        plt.figure(4), plt.imshow(ref_insts[1].sum(dim=0)), plt.grid(linestyle=':', linewidth=0.4), plt.colorbar(), plt.ion(), plt.show()
        
        '''

        if self.transform is not None:
            imgs, segms, intrinsics = self.transform([tgt_img] + ref_imgs, tgt_insts + ref_insts, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
            tgt_insts = segms[:int(len(ref_imgs)/2+1)]
            ref_insts = segms[int(len(ref_imgs)/2+1):]

        else:
            intrinsics = np.copy(sample['intrinsics'])
        
        ### While passing through RandomScaleCrop(), instances could be flied-out and become zero-mask. -> Need filtering!
        for sq in range( len(ref_imgs) ):
            tgt_insts[sq], ref_insts[sq] = recursive_check_nonzero_inst(tgt_insts[sq], ref_insts[sq])


        if tgt_insts[0][0].mean() != 0 and tgt_insts[0][int(tgt_insts[0][0].mean())].mean() == 0: pdb.set_trace()
        if tgt_insts[1][0].mean() != 0 and tgt_insts[1][int(tgt_insts[1][0].mean())].mean() == 0: pdb.set_trace()
        if ref_insts[0][0].mean() != 0 and ref_insts[0][int(ref_insts[0][0].mean())].mean() == 0: pdb.set_trace()
        if ref_insts[1][0].mean() != 0 and ref_insts[1][int(ref_insts[1][0].mean())].mean() == 0: pdb.set_trace()

        if tgt_insts[0][0].mean() != tgt_insts[0][1:].mean(-1).mean(-1).nonzero().size(0): pdb.set_trace()
        if tgt_insts[1][0].mean() != tgt_insts[1][1:].mean(-1).mean(-1).nonzero().size(0): pdb.set_trace()
        if ref_insts[0][0].mean() != ref_insts[0][1:].mean(-1).mean(-1).nonzero().size(0): pdb.set_trace()
        if ref_insts[1][0].mean() != ref_insts[1][1:].mean(-1).mean(-1).nonzero().size(0): pdb.set_trace()

        # pdb.set_trace()
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics), tgt_insts, ref_insts

    def __len__(self):
        return len(self.samples)
