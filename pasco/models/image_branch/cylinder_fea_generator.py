# -*- coding:utf-8 -*-
# author: Xinge

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import torch_scatter

from pasco.models.image_branch.swiftnet import SwiftNetRes18
from pasco.models.image_branch.util.LIfusion_block import Feature_Gather, Atten_Fusion_Conv
from pasco.models.image_branch.util.cam import returnCAM
from pasco.models.image_branch.pvp_generation import shift_voxel_grids, return_tensor_index, return_tensor_index_v2
from pasco.models.fuser import VisFuser



import time
from pasco.utils.measure_time import measure_time, measure_time_for_class

dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2_vits14.eval()

INDEX_SHIFT = [[0,-1,-1,-1], [0, -1,-1,0], [0, -1,-1,1], [0, -1,0,-1], [0, -1,0,0], [0,-1,0,1],
               [0,-1,1,-1], [0,-1,1,0], [0,-1,1,1], [0,0,-1,-1], [0,0,-1,0], [0,0,-1,1],
               [0,0,0,-1], [0,0,0,0], [0,0,0,1], [0,0,1,-1], [0,0,1,0], [0,0,1,1],
               [0,1,-1,-1],[0,1,-1,0], [0,1,-1,1], [0,1,0,-1], [0,1,0,0], [0,1,0,1], 
               [0,1,1,-1], [0,1,1,0], [0,1,1,1]]
# @measure_time_for_class
class cylinder_fea(nn.Module):

    def __init__(self, cfgs, grid_size, nclasses, fea_dim=3,
                 out_pt_fea_dim=64, max_pt_per_encode=64, fea_compre=None, use_sara=False, tau=0.7, 
                 use_att=False, head_num=2):
        super(cylinder_fea, self).__init__()

        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(fea_dim),

            nn.Linear(fea_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, out_pt_fea_dim)
        )

        self.nclasses = nclasses
        self.max_pt = max_pt_per_encode
        self.fea_compre = fea_compre
        self.grid_size = grid_size
        kernel_size = 3
        self.local_pool_op = torch.nn.MaxPool2d(kernel_size, stride=1,
                                                padding=(kernel_size - 1) // 2,
                                                dilation=1)
        self.pool_dim = out_pt_fea_dim

        self.use_pix_fusion = cfgs['model']['pix_fusion']
        self.to_128 = nn.Sequential(
                nn.Linear(self.pool_dim, 128)
            )
        if self.use_pix_fusion:
            # self.use_dinov2 = False
            # if self.use_dinov2:
            #     self.pix_branch = dinov2_vits14
            # else:
                self.pix_branch = SwiftNetRes18(num_feature=(128, 128, 128), pretrained_path=cfgs['model']['pix_fusion_path'])
            
        if self.use_pix_fusion:
            self.pixcls = nn.Linear(128, nclasses, bias=False)
            # self.pixcls = nn.Sequential(
            #     nn.Linear(128, 128),
            #     nn.ReLU(),
            #     nn.Linear(128, nclasses, bias=False)
            # )
            self.temp = 0.1
        
        if self.fea_compre is not None:
            if self.use_pix_fusion:
                self.fea_compression = nn.Sequential(
                    nn.Linear(128, self.fea_compre),
                    nn.ReLU())
            else:
                self.fea_compression = nn.Sequential(
                    nn.Linear(128, self.fea_compre),
                    nn.ReLU())
            # #adding img_encoder
            # self.img_fea_compression = nn.Sequential(
            #         nn.Linear(128, self.fea_compre),
            #         nn.ReLU())
            self.pt_fea_dim = self.fea_compre
        else:
            self.pt_fea_dim = self.pool_dim
        
        self.use_sara = use_sara
        self.tau = tau
        self.cam_relu = nn.ReLU()
        self.cam_softmax = nn.Softmax(dim=1)
        self.sara_avgpool = nn.AdaptiveAvgPool1d(1)
        if self.use_sara:
            self.sara_to_128 = nn.Sequential(
                nn.Linear(self.pool_dim + 128 *2, 128)
            )
        
        self.use_att = use_att
        self.head_num = head_num
        if self.use_att:
            self.atten_fusion = torch.nn.MultiheadAttention(embed_dim=128, num_heads=self.head_num, dropout=0.1, batch_first=True)
        
        # self.use_att_img_branch = True
        # if self.use_att_img_branch:
        #     self.query_proj = nn.Linear(out_pt_fea_dim, 128)
        #     self.key_proj = nn.Linear(128*2, 128)
        #     self.atten_img_branch = torch.nn.MultiheadAttention(embed_dim=128, num_heads=self.head_num, dropout=0.1, batch_first=True)
        # self.VisFuser = VisFuser(128, 128, norm_cfg=None)

    def forward(self, pt_fea, xy_ind, fusion_dict):
        # start_mlp = time.perf_counter()
        device = torch.device('cuda')
        cur_dev = pt_fea[0].get_device()
        pt_ind = []
        # print("len(xy_ind)", len(xy_ind), xy_ind[0].shape)
        for i_batch in range(len(xy_ind)):
            pt_ind.append(F.pad(xy_ind[i_batch], (1, 0), 'constant', value=i_batch))
        # print("pt_ind", pt_ind[0].shape, len(pt_ind))
        # MLP sub-branch
        mlp_fea = []
        for _, f in enumerate(pt_fea):
            mlp_fea.append(self.PPmodel(f)) # MLP features
        # print("mlp feature: ", time.perf_counter() - start_mlp)
        # print("mlp_fea", mlp_fea[0].shape, len(mlp_fea))
        ####################################################
        # DINOv2 to extract feature of image
        # self.use_dinov2 = True
        #     if self.use_dinov2:
        #         with torch.no_grad():
        #             features = dinov2_vits14(imf_no_zero_tensor)
        #             segmentation_head = nn.Conv2d(in_channels=features.shape[1], out_channels=self.nclasses, kernel_size=1)
        #             segmentation_logits = segmentation_head(features)
        #             segmentation_probs = torch.softmax(segmentation_logits, dim=1)
        #             fusion_tensor = torch.cat((pt_fea_no_zero_tensor, segmentation_probs, imf_no_zero_tensor), dim=1)
        #             fusion_tensor = self.sara_to_128(fusion_tensor)
        #################################################### 
        # Asynchronous Compensated Pixel Alignment (ACPA Module)
        # The concret alignment logit is defined in dataset.py
        ####################################################
        # start_mapping = time.perf_counter()
        if self.use_pix_fusion:
            im = fusion_dict['camera'][0]
            masks = fusion_dict['masks']
            pixel_coords = fusion_dict['pixel_coordinates']
            valid_mask = fusion_dict['valid_mask']
            label_point = fusion_dict['input_pcd_semantic_label']
            batch_size = len(xy_ind)
            # print("batch_size", batch_size)
            # print("im shape", len(im), im[0].shape, im)
            # print("masks shape", len(masks), masks[0].shape, masks)
            # print("pixel_coords shape", len(pixel_coords), pixel_coords[0].shape, pixel_coords)
            # print("valid_mask", valid_mask, valid_mask[0].shape)
            if im.shape[-1] == 3:   
                # im = im.permute(0, 1, 4, 2, 3)
                im = im.transpose(2,0,1)
                # exit()
            # im = im.view(-1, im.shape[2], im.shape[3], im.shape[4]).contiguous()
            # print("im shape", im.shape, im, type(im))
            # im = np.array(im).unsqueeze(0)
            im = np.expand_dims(im, axis=0)
            im = torch.from_numpy(im).to(device)
            pix_fea, multi_scale_feature = self.pix_branch(im)
            pix_fea = pix_fea.view(batch_size, -1, pix_fea.size(1), pix_fea.size(2), pix_fea.size(3))
            # print("pix_fea shape", pix_fea.shape)
            imf_no_zero_tensor = []
            pt_ind_no_zero_tensor = []
            pt_fea_no_zero_tensor = []
            label_no_zero_tensor = []
            # print("preparing time: ", time.perf_counter()-start_mapping)
            # for_time = time.perf_counter()
            # tranverse the batch
            for mask, coord, img, pt_ind_sg, pt_fea_sg, vm, label in zip(masks, pixel_coords, pix_fea, pt_ind, mlp_fea, valid_mask, label_point):
                # print("label before", label, label.shape)
                # print("pt_ind_sg before", pt_ind_sg.shape)
                # print("pt_fea_sg before", pt_fea_sg.shape)
                coord = torch.from_numpy(coord).float().to(device)
                # print("mask", mask, mask.shape)
                mask = torch.from_numpy(mask).to(device)
                # print("mask", mask, mask.shape)
                label = label.to(device)
                # print("coord", coord, coord.shape)
                # print("img", img.shape)
                imf = torch.zeros(size=(mask.size(0), img.size(1)), device=device) # (N, C)
                # imf_copy = torch.zeros(size=(mask.size(0), img.size(1)), device=device)
                # print("imf shape", imf.shape)
                # feat_gather = time.perf_counter()
                imf_list = Feature_Gather(img.to(device), coord.unsqueeze(0)).permute(0, 2, 1)  # (B, N, C)
                # print("fea_gather: ", time.perf_counter() - feat_gather)
                # print("imf_list", imf_list, imf_list.shape)
                # print("coord", coord, coord.size(0))
                # print("mask", mask, mask.size(0))
                assert mask.size(0) == coord.size(0)
                # mask_time = time.perf_counter()
                # for idx in range(mask.size(0)):
                #     imf[mask[idx]] = imf_list[0, mask[idx], :]

                imf[mask] = imf_list[0, mask, :]
                # print("result: ", torch.all(imf_copy == imf))

                # print("mask_time: ", time.perf_counter() - mask_time)
                # 过滤掉不在相机内的点
                # filter_time = time.perf_counter()
                point_with_pix_mask = vm > -1
                imf_no_zero = imf[point_with_pix_mask]
                # print("imf_no_zero", imf_no_zero, imf_no_zero.shape)
                
                label_no_zero = label[point_with_pix_mask]              #thêm vào
                pt_ind_no_zero = pt_ind_sg[point_with_pix_mask]
                pt_fea_no_zero = pt_fea_sg[point_with_pix_mask]
                # print("filter time", time.perf_counter() - filter_time)

                # time_append = time.perf_counter() 
                label_no_zero_tensor.append(label_no_zero)              #thêm vào
                imf_no_zero_tensor.append(imf_no_zero)
                pt_ind_no_zero_tensor.append(pt_ind_no_zero)
                pt_fea_no_zero_tensor.append(pt_fea_no_zero)
            #     print("time_append", time.perf_counter() - time_append)
            # print("for time: ", time.perf_counter()-for_time)
            # cat_time = time.perf_counter()
            imf_no_zero_tensor = torch.cat(imf_no_zero_tensor, dim=0)
            pt_ind_no_zero_tensor = torch.cat(pt_ind_no_zero_tensor, dim=0)
            pt_fea_no_zero_tensor = torch.cat(pt_fea_no_zero_tensor, dim=0)
            label_no_zero_tensor = torch.cat(label_no_zero_tensor, dim=0)           #thêm vào
            # print("cat time: ", time.perf_counter() - cat_time)
            # print("projecting: ", time.perf_counter() - start_mapping)
            # print("label_no_zero_tensor after", label_no_zero_tensor, label_no_zero_tensor.shape)
            # print("pt_ind_no_zero_tensor after", pt_ind_no_zero_tensor.shape)
            # print("pt_fea_no_zero_tensor after", pt_fea_no_zero_tensor.shape)
            # print("imf_no_zero_tensor after", imf_no_zero_tensor, imf_no_zero_tensor.shape)
            ####################################################
            # Semantic-Aware Region Alignment(SARA Module)
            ####################################################
            # start_time_sara = time.perf_counter()
            
            if self.use_sara:
                # image pixel classifaication
                pix_logits = self.pixcls(imf_no_zero_tensor) # pix_number x 128
                # temperature to adjust pix_logits
                # pix_logits = pix_logits / self.temp
                softmax_pix_logits = self.cam_softmax(pix_logits)
                probs, idx = softmax_pix_logits.sort(1, True)
                best_idx = idx[:, 0]
                
                weight_matrix = self.pixcls.weight.data # pix_number x n_classes
                # weight_matrix = self.pixcls[2].weight.data
                stacked_pixfea = pix_fea.view(-1, pix_fea.size(2), pix_fea.size(3), pix_fea.size(4)) # (bs*cam_num, C, H, W)
                
                cam = returnCAM(stacked_pixfea, weight_matrix) # (nclasses, bs*cam_num, width, height)
                # apply filtering by predifined self.tau
                filtered_cam = self.cam_relu(cam - self.tau)
                
                # get pixel sets (bs*cam_num, H*W, C)
                pixel_set = stacked_pixfea.view(stacked_pixfea.shape[0], stacked_pixfea.shape[1],-1).permute(0,2,1)
                # get filtered gate (nclasses, bs*cam_num, H*W)
                filtered_gate = filtered_cam.view(filtered_cam.shape[0], filtered_cam.shape[1],-1)
                sara_feature_per_cls = [] # (nclasses, C)
                for cls in range(self.nclasses):
                    per_cls_nonzero_mask = filtered_gate[cls] > 0 # (bs*cam_num, H*W)
                    # exclude zeros positions at first, to save memory
                    per_cls_filtered_gate = (filtered_gate[cls][filtered_gate[cls]>0] + self.tau)
                    per_cls_filtered_pixfeat = pixel_set[per_cls_nonzero_mask]
                    if per_cls_nonzero_mask.sum()>0:
                        selected_feat = per_cls_filtered_gate.unsqueeze(1) * per_cls_filtered_pixfeat
                        sara_feature_per_cls.append(self.sara_avgpool(selected_feat.permute(1,0)).squeeze(1))
                    else:
                        sara_feature_per_cls.append(torch.zeros(pixel_set.shape[-1],device=cur_dev))

                sara_feature_per_cls = torch.stack(sara_feature_per_cls)
                sara_feature = sara_feature_per_cls[best_idx] # (N, C)
                # choose from concatenation or addiction
                # fusion_tensor_cam = torch.cat((sara_feature, imf_no_zero_tensor), dim=1)
                # key_img = self.key_proj(fusion_tensor_cam.unsqueeze(1)) # (bs*cam_num, 1, 128)
                # query_pcd = self.query_proj(pt_fea_no_zero_tensor.unsqueeze(1)) # (bs*cam_num, 1, 128)
                # fusion_tensor = self.atten_img_branch(query_pcd, key_img, key_img)[0] # (bs*cam_num, 1, C) -> (bs*cam_num, C)
                # fusion_tensor = fusion_tensor.squeeze(1) # (bs*cam_num, C)
                # fusion_tensor = self.VisFuser(fusion_tensor_cam, pt_fea_no_zero_tensor)
                fusion_tensor = torch.cat((pt_fea_no_zero_tensor, sara_feature, imf_no_zero_tensor), dim=1)
                fusion_tensor = self.sara_to_128(fusion_tensor)
                
            else:
                cam = None
                softmax_pix_logits = None
                fusion_tensor = imf_no_zero_tensor            #ori : có
                    
            # end_time_sara = time.perf_counter()
            # print("time sara: ", end_time_sara-start_time_sara)
        ####################################################
        # Point-to-Voxel Aggregation (PVP)
        ####################################################
        # start_time_PVP = time.perf_counter()
        # original cylinder sub-branch
        # concate everything
        cat_pt_ind = torch.cat(pt_ind, dim=0)
        pt_num = cat_pt_ind.shape[0]

        # MLP sub-branch
        cat_mlp_fea = torch.cat(mlp_fea, dim=0)

        ### shuffle the data
        # shuffled_ind = torch.randperm(pt_num, device=cur_dev)
        # cat_mlp_fea_test = cat_mlp_fea[shuffled_ind, :]
        # cat_pt_ind_test = cat_pt_ind[shuffled_ind, :]
        # fusion_tensor_test = fusion_tensor[shuffled_ind, :]

        # unique xy grid index
        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind, return_inverse=True, return_counts=True, dim=0) # ori_cylinder_data ↔ unq，unq_inv
        unq = unq.type(torch.int64)
        # get cylinder voxel features
        ori_cylinder_data = torch_scatter.scatter_max(cat_mlp_fea, unq_inv, dim=0)[0]
        ori_cylinder_data = self.to_128(ori_cylinder_data)
                
        
        # fused cylinder
        if self.use_pix_fusion:
            unq_no_zero, unq_inv_no_zero, unq_cnt_no_zero = torch.unique(pt_ind_no_zero_tensor, return_inverse=True,
                                                                        return_counts=True, dim=0)  
            unq_no_zero = unq_no_zero.type(torch.int64)
            fusion_pooled = torch_scatter.scatter_max(fusion_tensor, unq_inv_no_zero, dim=0)[0] # sparse voxelization; fusion_pooled ↔ unq_no_zero，unq_inv_no_zero
            #(N_in_POVcam x C)
            cat_unq = torch.cat((unq, unq_no_zero), dim=0)
            unq_1, unq_inv_1, unq_cnt_1 = torch.unique(cat_unq, return_inverse=True, return_counts=True, dim=0) # fused_cylinder_data ↔ unq_1，unq_inv_1
            # directly addiction, like residual link.
            cat_fea = torch.cat((ori_cylinder_data, fusion_pooled), dim=0)
            fused_cylinder_data = torch_scatter.scatter_add(cat_fea, unq_inv_1, dim=0)
            # Voxel attention via sparse points operation.
            
            key_value_list = []
            if self.use_att:
                CHANNEL = fusion_pooled.shape[1]
                shifted_index = shift_voxel_grids(unq_no_zero, INDEX_SHIFT, self.grid_size, cur_dev) # (27, sp_v_n, 3)
                for i, each_shift_index in enumerate(shifted_index):
                    # select_ind = [return_tensor_index(value=shift_ind, t=unq_no_zero) for shift_ind in each_shift_index] # too slow
                    select_ind = return_tensor_index_v2(value=each_shift_index, t=unq_no_zero) # (27, 2)
                    select_ind = torch.tensor(select_ind, device=cur_dev) # (sp_v_n,)
                    condition = (select_ind >= 0).unsqueeze(1).expand_as(fusion_pooled) # (sp_v_n, CHANNEL)
                    tmp = fusion_pooled[select_ind] # (sp_v_n, CHANNEL)
                    tmp= tmp.masked_fill(~condition, 0) # fill zero if select_ind  == -1
                    key_value_list.append(tmp)     
                key = torch.stack(key_value_list)
                key = key.permute(1, 0, 2) # (sp_v_n, 27, CHANNEL)
                feat_query = fusion_pooled.unsqueeze(1) # (sp_v_n, 1, CHANNEL)
                out, _att_weights = self.atten_fusion(feat_query, key, key) # (sp_v_n, 1, CHANNEL)
                out = out.squeeze(1)
                cat_att_fea = torch.cat((fused_cylinder_data, out), dim=0)
                fused_cylinder_data = torch_scatter.scatter_add(cat_att_fea, unq_inv_1, dim=0)
        # end_time_PVP = time.perf_counter()  
        # print("time PVP: ", end_time_PVP - start_time_PVP)      

        if self.fea_compre:
            processed_pooled_data = self.fea_compression(fused_cylinder_data)
            # processed_image_data = self.img_fea_compression(fusion_pooled)      #adding
        else:
            processed_pooled_data = fused_cylinder_data
        # start = time.perf_counter()   
        # #fill processed_image_data for matching processed_pooled_data, the feature at outside unq_no_zero is zero
        # processed_image_data_full = torch.zeros((unq.shape[0], processed_image_data.shape[1]), device=processed_image_data.device, dtype=processed_image_data.dtype)
        # mask_data = (unq[None, :, :] == unq_no_zero[:, None, :]).all(-1)
        # idx_in_unq = mask_data.float().argmax(dim=1)
        # processed_image_data_full[idx_in_unq] = processed_image_data
        # print(time.perf_counter()  - start)
        if self.use_pix_fusion:
            # return unq, processed_pooled_data, softmax_pix_logits, cam, label_no_zero_tensor
            return unq, processed_pooled_data, multi_scale_feature, softmax_pix_logits, cam, label_no_zero_tensor
        else:
            return unq, processed_pooled_data, None, None