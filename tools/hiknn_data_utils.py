# Copyright (c) OpenMMLab. All rights reserved.
import os
from concurrent import futures as futures
from os import path as osp

import mmengine
import numpy as np
import pdb
import math 


class HikNNMVData(object):
    """ScanNet data.

    Generate scannet infos for scannet_converter.

    Args:
        root_path (str): Root path of the raw data.
        split (str, optional): Set split type of the data. Default: 'train'.
    """

    def __init__(self, root_path, split='train'):
        self.root_dir = root_path
        self.save_path = root_path
        self.split = split
        self.split_dir = osp.join(root_path)
        self.sample_id_list =  ['00']
        self.test_mode = (split == 'test')

    def __len__(self):
        return len(self.sample_id_list)
    # Use a unified intrinsics, same for all scenes
    # def get_intrinsics(self):
    #     return np.array([[288.9353025,0,159.5,0],[0,288.9353025,119.5,0],[0,0,1,0],[0,0,0,1]])

    def get_infos(self, num_workers=4, has_label=False, sample_id_list=None):
        """Get data infos.

        This method gets information from the raw data.

        Args:
            num_workers (int, optional): Number of threads to be used.
                Default: 4.
            has_label (bool, optional): Whether the data has label.
                Default: True.
            sample_id_list (list[int], optional): Index list of the sample.
                Default: None.

        Returns:
            infos (list[dict]): Information of the raw data.
        """

        def process_single_scene(sample_idx):
            print(f'sample_idx: {sample_idx}')
            info = dict()
            pc_info = {'scene_idx': sample_idx}
            info['point_cloud'] = pc_info
            files = os.listdir(osp.join(self.root_dir, 'points', sample_idx))
            files.sort(key=lambda x: int(x.split('/')[-1][:-4]))
                
            info['pts_paths'] = [osp.join('points', sample_idx, file) for file in files]
            info['super_pts_paths'] = [osp.join('super_points', sample_idx, file) for file in files]
            info['img_paths'] = [osp.join('../hik', sample_idx, 'image', file.split('.')[0]+'.jpg') for file in files]
            # info['poses'] = [np.load(osp.join(self.root_dir, 'SceneNN', sample_idx, 'pose', file.split('.')[0].zfill(5)+'.npy')) for file in files]

            if not self.test_mode:
                info['pts_instance_mask_paths'] = [osp.join('instance_mask', sample_idx, file) for file in files]
                info['pts_semantic_mask_paths'] = [osp.join('semantic_mask', sample_idx, file) for file in files]  
            return info

        sample_id_list = sample_id_list if sample_id_list is not None \
            else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        # list organization
        return list(infos)

class HikNNData(object):
    """ScanNet data.
    Generate scannet infos for scannet_converter.

    Args:
        root_path (str): Root path of the raw data.
        split (str, optional): Set split type of the data. Default: 'train'.
        scannet200 (bool): True for ScanNet200, else for ScanNet.
        save_path (str, optional): Output directory.
    """

    def __init__(self, root_path, split='train', save_path=None):
        self.root_dir = root_path
        self.save_path = root_path if save_path is None else save_path
        self.split = split
        self.split_dir = osp.join(root_path)
        self.classes = [
            'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
            'bookshelf', 'picture', 'counter', 'desk', 'curtain',
            'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
            'garbagebin'
        ]
        self.cat_ids = np.array([
            3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39
        ])

        self.cat2label = {cat: self.classes.index(cat) for cat in self.classes}
        self.label2cat = {self.cat2label[t]: t for t in self.cat2label}
        self.cat_ids2class = {
            nyu40id: i
            for i, nyu40id in enumerate(list(self.cat_ids))
        }
        assert split in ['train', 'val', 'test']
        self.sample_id_list = ['00']
        self.test_mode = (split == 'test')

    def __len__(self):
        return len(self.sample_id_list)

    def get_images(self, idx):
        paths = []
        path = osp.join(self.root_dir, 'posed_images', idx)
        for file in sorted(os.listdir(path)):
            if file.endswith('.jpg'):
                paths.append(osp.join('posed_images', idx, file))
        return paths

    def get_extrinsics(self, idx):
        extrinsics = []
        path = osp.join(self.root_dir, 'posed_images', idx)
        for file in sorted(os.listdir(path)):
            if file.endswith('.txt') and not file == 'intrinsic.txt':
                extrinsics.append(np.loadtxt(osp.join(path, file)))
        return extrinsics

    def get_intrinsics(self, idx):
        matrix_file = osp.join(self.root_dir, 'posed_images', idx,
                               'intrinsic.txt')
        mmengine.check_file_exist(matrix_file)
        return np.loadtxt(matrix_file)

    def get_infos(self, num_workers=4, has_label=False, sample_id_list=None):
        """Get data infos.

        This method gets information from the raw data.

        Args:
            num_workers (int, optional): Number of threads to be used.
                Default: 4.
            has_label (bool, optional): Whether the data has label.
                Default: True.
            sample_id_list (list[int], optional): Index list of the sample.
                Default: None.

        Returns:
            infos (list[dict]): Information of the raw data.
        """

        def process_single_scene(sample_idx):
            print(f'{self.split} sample_idx: {sample_idx}')
            info = dict()
            pc_info = {'num_features': 6, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info
            pts_filename = osp.join(self.root_dir, 'scenenn_instance_data',
                                    f'{sample_idx}_vert.npy')
            points = np.load(pts_filename).astype(np.float32)
            mmengine.mkdir_or_exist(osp.join(self.save_path, 'points'))
            points.tofile(
                osp.join(self.save_path, 'points', f'{sample_idx}.bin'))
            info['pts_path'] = osp.join('points', f'{sample_idx}.bin')

            sp_filename = osp.join(self.root_dir, 'scenenn_instance_data',
                                    f'{sample_idx}_sp_label.npy')
            super_points = np.load(sp_filename)
            mmengine.mkdir_or_exist(osp.join(self.save_path, 'super_points'))
            super_points.tofile(
                osp.join(self.save_path, 'super_points', f'{sample_idx}.bin'))
            info['super_pts_path'] = osp.join('super_points', f'{sample_idx}.bin')

            # update with RGB image paths if exist
            if os.path.exists(osp.join(self.root_dir, 'posed_images')):
                info['intrinsics'] = self.get_intrinsics(sample_idx)
                all_extrinsics = self.get_extrinsics(sample_idx)
                all_img_paths = self.get_images(sample_idx)
                # some poses in ScanNet are invalid
                extrinsics, img_paths = [], []
                for extrinsic, img_path in zip(all_extrinsics, all_img_paths):
                    if np.all(np.isfinite(extrinsic)):
                        img_paths.append(img_path)
                        extrinsics.append(extrinsic)
                info['extrinsics'] = extrinsics
                info['img_paths'] = img_paths

        sample_id_list = sample_id_list if sample_id_list is not None \
            else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)




    
