import torch
import os
import glob
from torch.utils.data import Dataset
import numpy as np
import pasco.data.semantic_kitti.io_data as SemanticKittiIO
import yaml
from tqdm import tqdm
import random
import pickle
import torch.nn.functional as F
from pasco.models.augmenter import Augmenter
from pasco.models.misc import compute_scene_size
from pasco.models.transform_utils import (
    transform_scene,
    transform,
    generate_random_transformation,
)
from pasco.data.semantic_kitti.collate import collate_fn
from pasco.data.semantic_kitti.params import thing_ids
import random
from typing import List, Dict

from torchvision.transforms import transforms
from torchvision.transforms.functional import hflip, rotate, _get_inverse_affine_matrix, to_tensor, to_pil_image
from PIL import Image
from torch import nn

# transformation between Cartesian coordinates and polar coordinates
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


def polar2cat(input_xyz_polar):
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)

class KittiDataset(Dataset):
    def __init__(
        self,
        split,
        root,
        preprocess_root,
        config_path,
        n_subnets=2,
        data_aug=False,
        max_angle=90.0,
        translate_distance=0.2,
        scale_range=0,
        visualize=False,
        max_items=None,
        n_fuse_scans=1,
        complete_scale=8,
        frame_interval=5,
        frame_ids=[],
        using_img=False,
        rgb_img="P2",
    ):
        super().__init__()
        self.root = root

        self.complete_scale = complete_scale
        self.data_aug = data_aug
        self.max_angle = max_angle
        self.translate_distance = translate_distance
        self.scale_range = scale_range
        self.max_translation = np.array([3.0, 3.0, 2.0]) * translate_distance
        print(self.data_aug, self.max_angle, self.scale_range, self.translate_distance)

        self.preprocess_root = preprocess_root
        self.instance_label_root = os.path.join(preprocess_root, "instance_labels_v2")
        self.dataset_config = yaml.safe_load(open(config_path, "r"))
        self.remap_lut = self.get_remap_lut()
        self.n_classes = 20
        self.max_extent = (51.2, 25.6, 4.4)
        self.min_extent = np.array([0, -25.6, -2.0])
        self.n_subnets = n_subnets
        self.augmenter = Augmenter()
        self.n_fuse_scans = n_fuse_scans

        splits = {
            "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
            "val": ["08"],
            "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
            "trainval": [
                "00",
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
                "08",
                "09",
                "10",
            ],
        }

        self.split = split
        self.sequences = splits[split]
        self.scene_size = (51.2, 51.2, 6.4)
        self.vox_origin = np.array([0, -25.6, -2])
        self.voxel_size = 0.2  # 0.2m
        self.grid_size = [int(i / self.voxel_size) for i in self.scene_size]
        self.img_W = 640
        self.img_H = 360
        self.IMAGE_SIZE = [self.img_H , self.img_W]
        self.thing_ids = thing_ids

        self.scans = []
        for sequence in self.sequences:

            glob_path = os.path.join(
                self.root, "dataset", "sequences", sequence, "voxels", "*.bin"
            )
            for voxel_path in glob.glob(glob_path):
                filename = os.path.basename(voxel_path)
                frame_id = os.path.splitext(filename)[0]
                if float(frame_id) % frame_interval != 0:
                    continue

                if len(frame_ids) > 0 and visualize and frame_id not in frame_ids:
                    continue

                self.scans.append(
                    {
                        "sequence": sequence,
                        "frame_id": frame_id,
                    }
                )
        self.scans.sort(key=lambda x: x["frame_id"])

        if max_items is not None:
            self.scans = self.scans[:max_items]

        self.calibrations = []
        self.times = []
        self.poses = []

        self.load_calib_poses()
        
        self.resize = transforms.Compose([
            transforms.Resize(size=self.IMAGE_SIZE)
        ])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ),
        ])
        self.augment = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4)  # not strengthened
            ], p=0.5),
            # transforms.RandomGrayscale(p=0.1),
        ])
        self.pix_fusion = True
        self.img_aug = True
        self.flip_aug = False
        self.flip_aug_rate = 0.5
        self.rotate_aug = False
        self.rotate_max_angle = [-15, 15]
        self.mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
    def get_SemKITTI_label_name(self, label_mapping):
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        SemKITTI_label_name = dict()
        for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
            SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]

        return SemKITTI_label_name, semkittiyaml['learning_map']

    def __getitem__(self, idx):
        if self.split == "val":
            idx_list = [idx] * self.n_subnets
        else:
            idx_list = np.random.choice(
                len(self.scans), self.n_subnets - 1, replace=False
            )
            idx_list = idx_list.tolist() + [idx]

        items = []
        random.shuffle(idx_list)
        for id in idx_list:
            items.append(self.get_individual(id))

        return collate_fn(items, self.complete_scale)
    
    def _mappcd2img(self, seq, pts, im_size, color_lorr='P2'):
        
        # P, Tr = self.P_dict[seq + "_" +color_lorr], self.Tr_dict[seq]
        # print("color_lorr", color_lorr)
        P = self.calibrations[int(seq)][color_lorr]
        Tr = self.calibrations[int(seq)]["Tr"]
        pts_homo = np.column_stack((pts, np.array([1] * pts.shape[0], dtype=pts.dtype)))
        # Tr_homo = np.row_stack((Tr, np.array([0, 0, 0, 1], dtype=Tr.dtype)))
        Tr_homo = Tr
        # print("pts_homo", pts_homo, pts_homo.shape)
        # print("Tr_homo", Tr_homo, Tr_homo.shape)
        pixel_coord = np.matmul(Tr_homo, pts_homo.T)
        pixel_coord = np.matmul(P, pixel_coord).T
        pixel_coord = pixel_coord / (pixel_coord[:, 2].reshape(-1, 1))
        pixel_coord = pixel_coord[:, :2]

        x_on_image = (pixel_coord[:, 0] >= 0) & (pixel_coord[:, 0] <= (im_size[0] - 1))
        y_on_image = (pixel_coord[:, 1] >= 0) & (pixel_coord[:, 1] <= (im_size[1] - 1))
        mask = x_on_image & y_on_image & (pts[:, 0] > 0) # only front points

        return pixel_coord, mask
    
    def get_individual(self, idx):

        scales = [1, 2, 4]

        scan = self.scans[idx]
        sequence = scan["sequence"]
        frame_id = scan["frame_id"]
        # print("===============================Load data v3===============================")
        data = self.load_data_v3(sequence, frame_id, n_fuse_scans=self.n_fuse_scans)
        # print("==========================================================================")
        semantic_label_sparse_F, semantic_label_sparse_C = data["semantic_label_sparse"]
        instance_label_sparse_F, instance_label_sparse_C = data["instance_label_sparse"]
        semantic_label_origin, instance_label_origin = (
            data["semantic_label_origin"],
            data["instance_label_origin"],
        )
        input_pcd_instance_label = data["input_pcd_instance_label"]
        #--------------------Mapping----------------------
        _, input_pcd_semantic_label_map = self.get_SemKITTI_label_name("/media/anda/hdd3/Phat/PaSCo/semantic-kitti.yaml")

        flat_instance_label = input_pcd_instance_label.view(-1).tolist()

        # Map using list comprehension
        mapped_semantic = [input_pcd_semantic_label_map[int(i)] for i in flat_instance_label]
        input_pcd_semantic_label = torch.tensor(mapped_semantic, dtype=input_pcd_instance_label.dtype).view_as(input_pcd_instance_label)
        #====================================================
        in_feat, in_coord = data["in_feat"]
        T = data["T"]

        min_C_semantic = semantic_label_sparse_C.min(dim=0)[0]
        max_C_semantic = semantic_label_sparse_C.max(dim=0)[0]

        if instance_label_sparse_C.shape[0] > 0:
            min_C_instance = instance_label_sparse_C.min(dim=0)[0]
            max_C_instance = instance_label_sparse_C.max(dim=0)[0]
            min_C = torch.min(min_C_semantic, min_C_instance)
            max_C = torch.max(max_C_semantic, max_C_instance)
        else:
            min_C = min_C_semantic
            max_C = max_C_semantic
        min_C = torch.floor(min_C.float() / self.complete_scale) * self.complete_scale
        min_C = min_C.int()
        max_C = torch.ceil(max_C)

        scene_size = compute_scene_size(min_C, max_C, scale=self.complete_scale).int()

        semantic_label = torch.full(
            (scene_size[0], scene_size[1], scene_size[2]), 255, dtype=torch.uint8
        )
        nonnegative_coords = semantic_label_sparse_C - min_C
        nonnegative_coords = nonnegative_coords.long()
        semantic_label[
            nonnegative_coords[:, 0], nonnegative_coords[:, 1], nonnegative_coords[:, 2]
        ] = semantic_label_sparse_F.squeeze()

        instance_label = torch.full(
            (scene_size[0], scene_size[1], scene_size[2]), 0, dtype=torch.uint8
        )
        if instance_label_sparse_C.shape[0] > 0:
            nonnegative_coords = instance_label_sparse_C - min_C
            nonnegative_coords = nonnegative_coords.long()
            instance_label[
                nonnegative_coords[:, 0],
                nonnegative_coords[:, 1],
                nonnegative_coords[:, 2],
            ] = instance_label_sparse_F.squeeze()

        mask_label = self.prepare_mask_label(semantic_label, instance_label)
        mask_label_origin = self.prepare_mask_label(
            semantic_label_origin, instance_label_origin
        )

        complete_voxel = semantic_label.clone().float()
        complete_voxel[(semantic_label > 0) & (semantic_label != 255)] = 1
        complete_voxel_remove_255 = complete_voxel.clone()
        complete_voxel_remove_255[semantic_label == 255] = 0  # ignore 255

        scales = [1, 2, 4]
        geo_labels = {}
        sem_labels = {}

        temp = semantic_label.clone().long()
        temp[temp == 255] = 20
        sem_label_oh = F.one_hot(temp, num_classes=21).permute(3, 0, 1, 2).float()
        for scale in scales:
            if scale == 1:
                downscaled_label = complete_voxel
                downscaled_sem_label = semantic_label
            else:
                downscaled_label = (
                    F.max_pool3d(
                        complete_voxel_remove_255.unsqueeze(0).unsqueeze(0),
                        kernel_size=scale,
                        stride=scale,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
                downscaled_mask_255 = (
                    F.avg_pool3d(
                        complete_voxel.unsqueeze(0).unsqueeze(0),
                        kernel_size=scale,
                        stride=scale,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
                downscaled_label[downscaled_mask_255 == 255] = 255

                sem_label_oh_occ = sem_label_oh.clone()
                sem_label_oh_occ[0, :, :, :] = 0
                sem_label_oh_occ[20, :, :, :] = 0
                downscaled_sem_label = F.avg_pool3d(
                    sem_label_oh_occ.unsqueeze(0), kernel_size=scale, stride=scale
                ).squeeze(0)
                downscaled_sem_label = torch.argmax(downscaled_sem_label, dim=0)

                sem_label_oh_0_255 = sem_label_oh.clone()
                sem_label_oh_0_255[1:20, :, :, :] = 0
                downscaled_sem_label_0_255_oh = F.avg_pool3d(
                    sem_label_oh_0_255.unsqueeze(0), kernel_size=scale, stride=scale
                ).squeeze(0)
                downscaled_sem_label_0_255 = torch.full_like(downscaled_sem_label, 255)
                downscaled_sem_label_0_255[
                    downscaled_sem_label_0_255_oh[20, :, :, :] == 1
                ] = 0

                empty_mask = downscaled_sem_label == 0
                downscaled_sem_label[empty_mask] = downscaled_sem_label_0_255[
                    empty_mask
                ]

            sem_labels["1_{}".format(scale)] = downscaled_sem_label.type(torch.uint8)
            geo_labels["1_{}".format(scale)] = downscaled_label.type(torch.uint8)
        camera = data["camera"]
        ori_camera = data["ori_camera"]
        masks = data["masks"]
        pixel_coordinates = data["pixel_coordinates"]
        valid_mask = data["valid_mask"]
        ret_data = {
            "xyz": data["xyz"],
            "frame_id": frame_id,
            "sequence": sequence,
            "in_feat": in_feat.float(),
            "in_coord": in_coord,
            "T": T,
            "min_C": min_C,
            "max_C": max_C,
            "semantic_label": semantic_label,
            "instance_label": instance_label,
            "mask_label": mask_label,
            "geo_labels": geo_labels,
            "sem_labels": sem_labels,
            "semantic_label_origin": semantic_label_origin,
            "instance_label_origin": instance_label_origin,
            "mask_label_origin": mask_label_origin,
            "input_pcd_instance_label": input_pcd_instance_label,
            "input_pcd_semantic_label": input_pcd_semantic_label,
            "camera": camera,
            "ori_camera": ori_camera,
            "masks": masks,
            "pixel_coordinates": pixel_coordinates,
            "valid_mask": valid_mask,
        }

        return ret_data

    def load_file(self, path):
        with open(path, "rb") as handle:
            data = pickle.load(handle)

            embedding = data["embedding"]
            random_embedding_idx = np.random.randint(0, embedding.shape[0])
            embedding = embedding[random_embedding_idx].T

            xyz_and_density = data["coords"]
            xyz = xyz_and_density[:, :3]
            vote = data["vote"]
            intensity = xyz_and_density[:, 3:]

        return xyz, vote, intensity, embedding

    def voxelize(self, xyz, vote):
        vox_origin = self.vox_origin.reshape(1, 3)
        coords = (xyz - vox_origin.reshape(1, 3)) // self.voxel_size

        voxel_centers = (
            coords.astype(np.float32) + 0.5
        ) * self.voxel_size + vox_origin.reshape(1, 3)
        return_xyz = xyz - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz), axis=1)
        return return_xyz, vote, coords

    def load_input_pcd_instance_label(self, sequence, frame_id):
        
        path = os.path.join(
            self.root,
            "dataset",
            "sequences",
            sequence,
            "labels",
            "{}.label".format(frame_id),
        )
        path_img2 = os.path.join(
            self.root,
            "images",
            "dataset",
            "sequences",
            sequence,
            "image_2",
            "{}.png".format(frame_id),
        )
        path_img3 = os.path.join(
            self.root,
            "images",
            "dataset",
            "sequences",
            sequence,
            "image_3",
            "{}.png".format(frame_id),
        )

        instance_label = np.fromfile(path, dtype=np.int32).reshape((-1, 1))
        instance_label = instance_label & 0xFFFF  # delete high 16 digits binary
        # exit()
        img1 = Image.open(path_img2).convert('RGB')
        img2 = Image.open(path_img3).convert('RGB')
        return instance_label, img1, img2

    def load_data_v3(self, sequence, frame_id, downsample=1, n_fuse_scans=1):
        # print("n_fuse_scans", n_fuse_scans)
        data_path = os.path.join(
            self.instance_label_root,
            sequence,
            "{}_1_{}.pkl".format(frame_id, downsample),
        )

        with open(data_path, "rb") as handle:
            data = pickle.load(handle)
            semantic_label = data["semantic_labels"].astype(np.uint8)
            instance_label = data["instance_labels"].astype(np.uint8)

        for fuse_idx in range(n_fuse_scans):
            plus_idx = fuse_idx * 5
            number_idx = int(frame_id) + plus_idx
            add_frame_id = "{:06}".format(number_idx)
            # print("plus_idx", plus_idx)
            # print("number_idx", number_idx)
            # print("add_frame_id", add_frame_id)
            path = os.path.join(
                self.preprocess_root,
                "waffleiron_v2/sequences",
                sequence,
                "seg_feats_tta",
                "{}.pkl".format(add_frame_id),
            )
            # print("path", path)
            if os.path.exists(path):
                # print(f"path {path} exist and {fuse_idx}")
                if fuse_idx == 0:
                    pose0 = self.poses[int(sequence)][int(frame_id)]
                    xyz, vote, intensity, embedding = self.load_file(path)
                    vote_intensity = np.concatenate((vote, intensity), axis=1)
                    input_pcd_instance_label, img1, img2 = self.load_input_pcd_instance_label(
                        sequence, frame_id
                    )
                    # print("xyz", xyz.shape)
                    # print("vote", vote.shape)
                    # print("intensity", intensity.shape)
                    # print("embedding", embedding.shape)
                    # print("vote_intensity", vote_intensity.shape)
                    # print("input_pcd_instance_label", input_pcd_instance_label.shape)
                    # print("-----------------------------------")
                else:
                    pose = self.poses[int(sequence)][int(add_frame_id)]
                    add_xyz, add_vote, add_intensity, add_embedding = self.load_file(path) #adding
                    add_xyz = self.fuse_multi_scan(add_xyz, pose0, pose)
                    xyz = np.concatenate((xyz, add_xyz), axis=0)
                    vote = np.concatenate((vote, add_vote), axis=0)
                    #---------------------------adding-------------------------------------------
                    intensity = np.concatenate((intensity, add_intensity), axis=0) 
                    embedding = np.concatenate((embedding, add_embedding), axis=0)
                    vote_intensity = np.concatenate((vote, intensity), axis=1)  
                    input_pcd_instance_label_adding, add_img1, add_img2 = self.load_input_pcd_instance_label(
                        sequence, add_frame_id
                    )
                    input_pcd_instance_label = np.concatenate((input_pcd_instance_label, input_pcd_instance_label_adding), axis=0)
                    img1 = np.concatenate((img1, add_img1), axis=0)
                    img2 = np.concatenate((img2, add_img2), axis=0)
                    #----------------------------------------------------------------------------
                    # print("----------fuse_idx # 0---------------")
                    # print("xyz add", xyz.shape)
                    # print("vote add", vote.shape)
                    # print("intenaddy add", intensity.shape)
                    # print("embedding add ", embedding.shape)
                    # print("vote_intensity add ", vote_intensity.shape)
                    # print("input_pcd_instance_label add", input_pcd_instance_label.shape)
                    # print("-----------------------------------")
                 
        # xyz_pol = cart2polar(xyz)
        keep = (
            (xyz[:, 0] < self.max_extent[0])
            & (xyz[:, 0] >= self.min_extent[0])
            & (xyz[:, 1] < self.max_extent[1])
            & (xyz[:, 1] >= self.min_extent[1])
            & (xyz[:, 2] < self.max_extent[2])
            & (xyz[:, 2] >= self.min_extent[2])
        )
        # xyz_raw = xyz
        xyz = xyz[keep]
        # print("xyz shape", xyz.shape)
        intensity = intensity[keep]
        vote_intensity = vote_intensity[keep]
        embedding = embedding[keep]
        input_pcd_instance_label = input_pcd_instance_label[keep]
        #----------------------------------Image---------------------------
        pixel_coordinates1, mask1 = self._mappcd2img(sequence, xyz, img1.size, "P2")    #P2 = left
        pixel_coordinates2, mask2 = self._mappcd2img(sequence, xyz, img2.size, "P3")    #P3 = right
        # print("img1.size", img1.size)
        # print("img2.size", img2.size)
        # print("before--------------------")
        # print("max x pixel_coordinates1", pixel_coordinates1[:,0].max())
        # print("min x pixel_coordinates1", pixel_coordinates1[:,0].min())
        # print("max y pixel_coordinates1", pixel_coordinates1[:,1].max())
        # print("min y pixel_coordinates1", pixel_coordinates1[:,1].min())
        pixel_coordinates1[:, 0] = pixel_coordinates1[:, 0] / (img1.size[0] - 1) * 2 - 1.0
        pixel_coordinates1[:, 1] = pixel_coordinates1[:, 1] / (img1.size[1] - 1) * 2 - 1.0
        pixel_coordinates2[:, 0] = pixel_coordinates2[:, 0] / (img2.size[0] - 1) * 2 - 1.0
        pixel_coordinates2[:, 1] = pixel_coordinates2[:, 1] / (img2.size[1] - 1) * 2 - 1.0
        # print("after--------------------")
        # print("max x pixel_coordinates1", pixel_coordinates1[:,0].max())
        # print("min x pixel_coordinates1", pixel_coordinates1[:,0].min())
        # print("max y pixel_coordinates1", pixel_coordinates1[:,1].max())
        # print("min y pixel_coordinates1", pixel_coordinates1[:,1].min())
        pixel_coordinates = np.array([pixel_coordinates1, pixel_coordinates2])
        masks = np.array([mask1, mask2])
        
        # print("mask1", mask1.shape, type(mask1))
        # print("mask2", mask2.shape, type(mask2))
        # print("masks", masks, type(masks))
        # print("masks[0]", masks[0].shape)
        # masks = np.logical_or(mask1, mask2)

        img1 = self.resize(img1)
        img2 = self.resize(img2)
        ori_camera = np.stack((np.array(img1).astype('float32'), np.array(img2).astype('float32')), axis=0)
        if self.img_aug:
            img1 = self.augment(img1)
            img2 = self.augment(img2)
        img1 = self.transform(img1).permute((1,2,0))
        img2 = self.transform(img2).permute((1,2,0))
        camera = np.stack((img1, img2), axis=0)
        valid_mask = np.array([-1] * xyz.shape[0])
        monocular = True
        if monocular:
            valid_mask[mask1] = 1
        else:
            valid_mask[mask1] = 1
            valid_mask[mask2] = 2
            
        if self.pix_fusion:
            if monocular: #如果只是单目
                camera = camera[0]
                ori_camera = ori_camera[0]
                masks = masks[0]
                pixel_coordinates = pixel_coordinates[0]
                
        # print("pixel_coordinates before", pixel_coordinates.shape)    
        # print("masks before", masks.shape)   
        # print("valid_mask before", valid_mask.shape)   
            
        #----------------------------------Image---------------------------
        if self.data_aug:
            T = generate_random_transformation(
                max_angle=self.max_angle,
                flip=True,
                scale_range=self.scale_range,
                max_translation=self.max_translation,
            )

        else:
            T = torch.eye(4)

        semantic_label = torch.from_numpy(semantic_label)
        semantic_label_origin = semantic_label.clone()
        semantic_coords = torch.nonzero(semantic_label != 255)
        semantic_label_sparse, semantic_coords, to_coords_bnd = transform_scene(
            semantic_coords, T, semantic_label.unsqueeze(0) + 1
        )
        non_zero = semantic_label_sparse.sum(dim=1) != 0
        semantic_label_sparse = semantic_label_sparse[non_zero]
        semantic_label_sparse -= 1
        semantic_coords = semantic_coords[non_zero]

        instance_label = torch.from_numpy(instance_label)
        instance_label_origin = instance_label.clone()
        instance_coords = torch.nonzero(instance_label)
        if instance_coords.shape[0] > 0:
            instance_label_sparse, instance_coords, _ = transform_scene(
                instance_coords,
                T,
                instance_label.unsqueeze(0) + 1,
                to_coords_bnd=to_coords_bnd,
            )
        else:
            instance_label_sparse = torch.zeros((0, 1), dtype=torch.uint8)
            instance_coords = torch.zeros((0, 3)).long()
        non_zero = instance_label_sparse.sum(dim=1) != 0
        instance_label_sparse = instance_label_sparse[non_zero]
        instance_coords = instance_coords[non_zero]
        instance_label_sparse -= 1

        # xyz = transform_xyz(torch.from_numpy(xyz), T).numpy()
        radius = np.linalg.norm(xyz, axis=1)[..., np.newaxis]
        feat = np.concatenate((vote_intensity, radius, embedding), axis=1)        #origin
        # feat = np.concatenate((intensity, radius), axis=1)
        return_xyz, feat, coords = self.voxelize(xyz, feat)
        in_feat = np.concatenate([feat, return_xyz], axis=1)
        in_coords = torch.from_numpy(coords)
        in_coords = transform(in_coords, T)
        in_coords = in_coords.long()
        in_feat = torch.from_numpy(in_feat)
        # print("in_feat before", in_feat.shape)
        # print("in_coords before", in_coords, in_coords.shape)
        # print("input_pcd_instance_label before", input_pcd_instance_label, input_pcd_instance_label.shape)
        if self.split == "train":
            # print("========================Training Stage=========================")
            in_keep, semantic_keep, instance_keep = self.crop(
                semantic_coords, in_coords, instance_coords
            )
            in_feat = in_feat[in_keep]
            in_coords = in_coords[in_keep]
            # print("in_feat after", in_feat.shape)
            # print("in_coords after", in_coords.shape)
            semantic_label_sparse = semantic_label_sparse[semantic_keep]
            semantic_coords = semantic_coords[semantic_keep]
            instance_label_sparse = instance_label_sparse[instance_keep]
            instance_coords = instance_coords[instance_keep]
            
            masks = masks[in_keep]
            pixel_coordinates = pixel_coordinates[in_keep]
            valid_mask = valid_mask[in_keep]
            # input_pcd_semantic_label = input_pcd_semantic_label[in_keep]
            input_pcd_instance_label = input_pcd_instance_label[in_keep]
            # print("pixel_coordinates after", pixel_coordinates.shape)    
            # print("masks after", masks.shape)   
            # print("valid_mask after", valid_mask.shape)  
        # print("in_coords after", in_coords, in_coords.shape)
        # print("input_pcd_instance_label after", input_pcd_instance_label, input_pcd_instance_label.shape)
        xyz = xyz - self.vox_origin.reshape(1, 3)
        # print("semantic_label_sparse shape", semantic_label_sparse, semantic_label_sparse.shape, semantic_label_sparse.max())
        # print("instance_label_sparse shape", instance_label_sparse, instance_label_sparse.shape, instance_label_sparse.min())
        # exit()
        return {
            "xyz": xyz,
            "in_feat": (in_feat, in_coords),
            "semantic_label_sparse": (
                semantic_label_sparse.type(torch.uint8),
                semantic_coords,
            ),
            "instance_label_sparse": (
                instance_label_sparse.type(torch.uint8),
                instance_coords,
            ),
            "semantic_label_origin": semantic_label_origin.type(torch.uint8),
            "instance_label_origin": instance_label_origin.type(torch.uint8),
            "input_pcd_instance_label": torch.from_numpy(input_pcd_instance_label),
            "T": T,
            "camera": camera,
            "ori_camera": ori_camera,
            "masks": masks,
            "pixel_coordinates": pixel_coordinates,
            "valid_mask": valid_mask,
        }

    @staticmethod
    def crop(semantic_coords, in_coords, instance_coords):
        min_coords = semantic_coords.min(dim=0)[0]
        max_coords = semantic_coords.max(dim=0)[0]
        crop_size = (max_coords - min_coords) * 0.8
        new_min_coords = min_coords + (
            max_coords - min_coords - crop_size
        ) * np.random.rand(3)
        new_max_coords = new_min_coords + crop_size
        in_keep = (
            (in_coords[:, 0] >= new_min_coords[0])
            & (in_coords[:, 0] < new_max_coords[0])
            & (in_coords[:, 1] >= new_min_coords[1])
            & (in_coords[:, 1] < new_max_coords[1])
        )
        semantic_keep = (
            (semantic_coords[:, 0] >= new_min_coords[0])
            & (semantic_coords[:, 0] < new_max_coords[0])
            & (semantic_coords[:, 1] >= new_min_coords[1])
            & (semantic_coords[:, 1] < new_max_coords[1])
        )
        instance_keep = (
            (instance_coords[:, 0] >= new_min_coords[0])
            & (instance_coords[:, 0] < new_max_coords[0])
            & (instance_coords[:, 1] >= new_min_coords[1])
            & (instance_coords[:, 1] < new_max_coords[1])
        )
        return in_keep, semantic_keep, instance_keep

    def load_calib_poses(self):
        """
        load calib poses and times.
        """

        ###########
        # Load data
        ###########

        self.calibrations = []
        self.times = []
        self.poses = []
        self.pcd_files = []
        for seq in range(0, 22):
        # for seq in range(0, 11):  #orginal
            seq_folder = os.path.join(
                self.root, "dataset", "sequences", str(seq).zfill(2)
            )

            # Read Calib
            self.calibrations.append(
                self.parse_calibration(os.path.join(seq_folder, "calib.txt"))
            )

            # Read times
            self.times.append(
                np.loadtxt(os.path.join(seq_folder, "times.txt"), dtype=np.float32)
            )

            # Read poses
            poses_f64 = self.parse_poses(
                os.path.join(seq_folder, "poses.txt"), self.calibrations[-1]
            )
            self.poses.append([pose.astype(np.float32) for pose in poses_f64])

            # #Read image
            for pcd_name in sorted(os.listdir(os.path.join(seq_folder, 'velodyne'))):   
                self.pcd_files.append(os.path.join(self.root, "dataset", "sequences", str(seq).zfill(2), 'velodyne', str(pcd_name)))
            #     self.img_files[0].append(os.path.join(self.root, "images", "dataset", "sequences", str(seq).zfill(2), 'image_2', str(pcd_name[:-4]) + '.png'))
            #     self.img_files[1].append(os.path.join(self.root, "images", "dataset", "sequences", str(seq).zfill(2), 'image_3', str(pcd_name[:-4]) + '.png'))
            
    def parse_calibration(self, filename):
        """read calibration file with given filename

        Returns
        -------
        dict
            Calibration matrices as 4x4 numpy arrays.
        """
        calib = {}
        # calib_file_test = open("/media/anda/hdd31/Phat/PaSCo/gpfsdswork/semanticKITTI/dataset/sequences/00/calib.txt")
        # print("calib_file_test", calib_file_test)
        # print("filename :", filename)
        calib_file = open(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

        calib_file.close()

        return calib

    def parse_poses(self, filename, calibration):
        """read poses file with per-scan poses from given filename

        Returns
        -------
        list
            list of poses as 4x4 numpy arrays.
        """
        file = open(filename)

        poses = []

        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses

    def fuse_multi_scan(self, points, pose0, pose):

        hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
        new_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)
        new_points = new_points[:, :3]
        new_coords = new_points - pose0[:3, 3]
        new_coords = np.sum(np.expand_dims(new_coords, 2) * pose0[:3, :3], axis=1)
        new_coords = np.hstack((new_coords, points[:, 3:]))

        return new_coords

    @staticmethod
    def prepare_target(target: torch.Tensor, ignore_labels: List[int]) -> Dict:
        # z, y, x = target.shape
        unique_ids = torch.unique(target)
        unique_ids = torch.tensor(
            [unique_id for unique_id in unique_ids if unique_id not in ignore_labels]
        )
        masks = []

        for id in unique_ids:
            masks.append(target == id)

        masks = torch.stack(masks)

        return {"labels": unique_ids, "masks": masks}

    def prepare_mask_label(self, semantic_label, instance_label):
        mask_semantic_label = self.prepare_target(
            semantic_label, ignore_labels=[0, 255]
        )  # NOTE: remove empty class

        stuff_filtered_mask = [
            t not in self.thing_ids for t in mask_semantic_label["labels"]
        ]
        stuff_semantic_labels = mask_semantic_label["labels"][stuff_filtered_mask]
        stuff_semantic_masks = mask_semantic_label["masks"][stuff_filtered_mask]
        labels = [stuff_semantic_labels]
        masks = [stuff_semantic_masks]

        mask_instance_label = self.prepare_instance_target(
            semantic_target=semantic_label,
            instance_target=instance_label,
            ignore_label=0,
        )  # The empty class is already included in the stuff_semantic_labels

        if mask_instance_label is not None:  # there are thing objects
            labels.append(mask_instance_label["labels"])
            masks.append(mask_instance_label["masks"])

        mask_label = {
            "labels": torch.cat(labels, dim=0),
            "masks": torch.cat(masks, dim=0),
        }

        return mask_label

    @staticmethod
    def prepare_instance_target(
        semantic_target: torch.Tensor, instance_target: torch.Tensor, ignore_label: int
    ) -> dict:
        # z, y, x = target.shape
        unique_instance_ids = torch.unique(instance_target)

        unique_instance_ids = unique_instance_ids[unique_instance_ids != ignore_label]
        masks = []
        semantic_labels = []

        for id in unique_instance_ids:
            masks.append(instance_target == id)
            semantic_labels.append(semantic_target[instance_target == id][0])

        if len(masks) == 0:
            return None

        masks = torch.stack(masks)
        semantic_labels = torch.tensor(semantic_labels)

        return {
            # "instance_ids": unique_instance_ids,
            "labels": semantic_labels,
            "masks": masks,
        }

    def get_label(self, invalid_path, label_path):

        INVALID = SemanticKittiIO._read_invalid_SemKITTI(invalid_path)
        LABEL = SemanticKittiIO._read_label_SemKITTI(label_path)

        LABEL = self.remap_lut[LABEL.astype(np.uint16)].astype(
            np.float32
        )  # Remap 20 classes semanticKITTI SSC
        LABEL[np.isclose(INVALID, 1)] = (
            255  # Setting to unknown all voxels marked on invalid mask...
        )
        LABEL = LABEL.reshape(self.grid_size)

        return LABEL

    def __len__(self):
        return len(self.scans)

    @staticmethod
    def read_calib(calib_path):
        """
        Modify from https://github.com/utiasSTARS/pykitti/blob/d3e1bb81676e831886726cc5ed79ce1f049aef2c/pykitti/utils.py#L68
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        # 3x4 projection matrix for left camera
        calib_out["P2"] = calib_all["P2"].reshape(3, 4)
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)
        return calib_out

    def get_remap_lut(self):
        """
        remap_lut to remap classes of semantic kitti for training...
        :return:
        """

        # make lookup table for mapping
        maxkey = max(self.dataset_config["learning_map"].keys())

        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(self.dataset_config["learning_map"].keys())] = list(
            self.dataset_config["learning_map"].values()
        )

        # in completion we have to distinguish empty and invalid voxels.
        # Important: For voxels 0 corresponds to "empty" and not "unlabeled".
        remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
        remap_lut[0] = 0  # only 'empty' stays 'empty'.

        return remap_lut


if __name__ == "__main__":
    kitti_config = "/gpfswork/rech/kvd/uyl37fq/code/uncertainty/uncertainty/data/semantic_kitti/semantic-kitti.yaml"
    kitti_root = "/gpfsdswork/dataset/SemanticKITTI"
    kitti_preprocess_root = "/lustre/fsn1/projects/rech/kvd/uyl37fq/monoscene_preprocess/kitti"

    dataset = KittiDataset(
        "val",
        kitti_root,
        kitti_preprocess_root,
        kitti_config,
        n_subnets=1,
        instance_cut_mix=False,
        max_angle=30,
        translate_distance=0.2,
        data_aug=False,
    )

    frame_id_with_trucks = []
    frame_id_with_bicycles = []
    frame_id_with_persons = []
    counts = []
    for i in tqdm(range(len(dataset))):
        counts.append(len(dataset[i]["mask_label_origin"][0]["labels"]))
        if i % 20 == 0 or i == len(dataset) - 1:
            print(np.min(counts), np.max(counts), np.mean(counts))
