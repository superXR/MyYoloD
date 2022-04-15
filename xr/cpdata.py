'''
Descripttion: 
version: 
Author: 徐瑞
Date: 2021-12-14 10:43:40
LastEditors: 徐瑞
LastEditTime: 2021-12-14 11:17:32
'''
import os
import sys
from tqdm import tqdm
from random import random


img_fdir = 'D:\StudyFiles\project\Graduation_design\data\TestOnTraining\images'
det_anno = 'D:\StudyFiles\project\Graduation_design\data\\bdd100k\det_annotations\data2\zwt\\bdd\\bdd100k\labels\\100k'
da_seg = 'D:\StudyFiles\project\Graduation_design\data\\bdd100k\da_seg_annotations\\bdd_seg_gt'
ll_seg = 'D:\StudyFiles\project\Graduation_design\data\\bdd100k\ll_seg_annotations\\bdd_lane_gt'

det_anno_fdir = 'D:\StudyFiles\project\Graduation_design\data\TestOnTraining\det_annotations'
da_seg_anno_fdir = 'D:\StudyFiles\project\Graduation_design\data\TestOnTraining\da_seg_annotations'
ll_seg_anno_fdir = 'D:\StudyFiles\project\Graduation_design\data\TestOnTraining\ll_seg_annotations'


def cpdata(mode):
    img_dir = os.path.join(img_fdir, mode)
    det_dir = os.path.join(det_anno, mode)
    da_seg_dir = os.path.join(da_seg, mode)
    ll_seg_dir = os.path.join(ll_seg, mode)

    
    det_anno_dir = os.path.join(det_anno_fdir, mode)
    da_seg_anno_dir = os.path.join(da_seg_anno_fdir, mode)
    ll_seg_anno_dir = os.path.join(ll_seg_anno_fdir, mode)
    
    img_path_list = os.listdir(img_dir)
    for img_path in tqdm(img_path_list):
        img_name = img_path.split('.')[0]
        det_path = os.path.join(det_dir, img_name + '.json')
        da_seg_path = os.path.join(da_seg_dir, img_name + '.png')
        ll_seg_path = os.path.join(ll_seg_dir, img_name + '.png')

        cp = 'copy {} {}'.format(det_path, det_anno_dir)
        os.system(cp)
        cp = 'copy {} {}'.format(da_seg_path, da_seg_anno_dir)
        os.system(cp)
        cp = 'copy {} {}'.format(ll_seg_path, ll_seg_anno_dir)
        os.system(cp)

def select_img(path):
    img_list = os.listdir(path)


if __name__ == '__main__':
    cpdata('train')
    cpdata('val')