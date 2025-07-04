import torch
import torch.nn as nn

from pasco.models.transformer.position_encoding import PositionEmbeddingSineSparse
from pasco.models.layers import SPCDense3Dv2
import MinkowskiEngine as ME
from pasco.models.dropout import MinkowskiSpatialDropout
from pasco.models.decoder_img import DecoderGenerativeSepConvV2
# from pasco.models.encoder_img import Encoder3DSepV2
# from pasco.models.decoder_v3 import DecoderGenerativeSepConvV2
from pasco.models.encoder_v2 import Encoder3DSepV2
from pasco.models.misc import compute_scene_size
import torch_scatter
import torch.nn.functional as F

from pasco.utils.measure_time import measure_time, measure_time_for_class
# from pasco.models.deformable_att import MSDeformAttn
from pasco.models.transformer.blocks import CrossAttentionLayer, SelfAttentionLayer
# @measure_time_for_class
class CylinderFeat(nn.Module):

    def __init__(
        self, fea_dim=3, out_pt_fea_dim=64, max_pt_per_encode=64, fea_compre=None
    ):
        super(CylinderFeat, self).__init__()

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
            nn.Linear(256, out_pt_fea_dim),
        )

        self.max_pt = max_pt_per_encode
        self.fea_compre = fea_compre
        kernel_size = 3
        self.local_pool_op = torch.nn.MaxPool2d(
            kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1
        )
        self.pool_dim = out_pt_fea_dim

        # point feature compression
        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                nn.Linear(self.pool_dim, self.fea_compre), nn.ReLU()
            )
            self.pt_fea_dim = self.fea_compre
        else:
            self.pt_fea_dim = self.pool_dim

    def forward(self, pt_fea, xy_ind):

        cur_dev = pt_fea[0].get_device()

        ### concate everything
        cat_pt_ind = []
        for i_batch in range(len(xy_ind)):
            cat_pt_ind.append(F.pad(xy_ind[i_batch], (1, 0), "constant", value=i_batch))

        cat_pt_fea = torch.cat(pt_fea, dim=0)
        cat_pt_ind = torch.cat(cat_pt_ind, dim=0)
        pt_num = cat_pt_ind.shape[0]

        ### shuffle the data
        shuffled_ind = torch.randperm(pt_num, device=cur_dev)
        cat_pt_fea = cat_pt_fea[shuffled_ind, :]
        cat_pt_ind = cat_pt_ind[shuffled_ind, :]

        ### unique xy grid index
        unq, unq_inv, unq_cnt = torch.unique(
            cat_pt_ind, return_inverse=True, return_counts=True, dim=0
        )
        unq = unq.type(torch.int64)

        ### process feature
        processed_cat_pt_fea = self.PPmodel(cat_pt_fea)
        pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0]

        if self.fea_compre:
            # print("fea_compre", self.fea_compre)
            processed_pooled_data = self.fea_compression(pooled_data)
        else:
            # print("None")s
            processed_pooled_data = pooled_data

        return unq, processed_pooled_data
    
#offset predict branch
class PredBranch(nn.Module):
    def __init__(self, cin, cout):
        super(PredBranch, self).__init__()
        self.pred_layer = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Conv2d(cin, cout, kernel_size=1, stride=1, padding=0, dilation=1)
        )
    
    def forward(self, x):
        pred = self.pred_layer(x)
        return pred
# @measure_time_for_class
class UNet3DV2(nn.Module):

    def __init__(
        self,
        in_channels,
        n_classes,
        dense3d_dropout,
        decoder_dropouts,
        encoder_dropouts,
        transformer_predictor,
        n_infers,
        heavy_decoder=True,
        dropout_type="spatial",
        use_se_layer=False,
        num_queries=100,
        query_sample_ratio=0.5,
        f_maps=32,
        drop_path_rate=0.0,
    ):
        super(UNet3DV2, self).__init__()
        print(
            "dropout_type: ",
            dropout_type,
            "heavy_decoder: ",
            heavy_decoder,
            "dense 3d dropout: ",
            dense3d_dropout,
        )
        self.n_infers = n_infers
        self.num_queries_sampled = int(num_queries * query_sample_ratio)
        if dropout_type == "spatial":
            sparse_dropout_layer = MinkowskiSpatialDropout
            dense_dropout_layer = nn.Dropout3d

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        self.transformer_predictor = transformer_predictor
        sparse_norm_layer = ME.MinkowskiBatchNorm

        sparse_act_layer = ME.MinkowskiReLU

        enc_depth = 0
        dec_depth = 21
        total_depth = enc_depth + dec_depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        self.encoder = Encoder3DSepV2(
            in_channels,
            f_maps,
            heavy_decoder=heavy_decoder,
            n_heads=1,
            drop_path_rates=dpr[:enc_depth],
            use_se_layer=use_se_layer,
            dropout_layer=sparse_dropout_layer,
            dropouts=encoder_dropouts,
            norm_layer=sparse_norm_layer,
            act_layer=sparse_act_layer,
        )

        dense_f = f_maps[-1]

        self.dense3d = nn.Sequential(
            SPCDense3Dv2(init_size=dense_f),
            dense_dropout_layer(dense3d_dropout),
        )

        self.decoder_generative = DecoderGenerativeSepConvV2(
            f_maps,
            heavy_decoder=heavy_decoder,
            n_classes=n_classes,
            use_se_layer=use_se_layer,
            n_infers=n_infers,
            drop_path_rates=dpr[enc_depth:],
            dropout_layer=sparse_dropout_layer,
            query_dim=self.transformer_predictor.query_dim,
            transformer_predictor=self.transformer_predictor,
            dropouts=decoder_dropouts,
            act_layer=sparse_act_layer,
            num_queries=num_queries,
            norm_layer=sparse_norm_layer,
        )

        self.encoder = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.encoder)
        self.decoder_generative = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(
            self.decoder_generative
        )

        conv_dim = 192      #ori: 192
        N_steps = conv_dim // 3
        self.pe_layer = PositionEmbeddingSineSparse(N_steps, normalize=True)
        # self.pe_layer_img = PositionEmbeddingSineSparse(N_steps, normalize=True)

        # self.mask_dim = 128
        # self.hidden_dim = 384
        # self.mask_feat_proj = nn.Linear(self.mask_dim , self.hidden_dim )
        # self.mask_feat_proj_img = nn.Linear(self.mask_dim, self.hidden_dim )

        self.sigmoid_sparse = ME.MinkowskiSigmoid()
        
        #instance branch from CFNet 
        # self.score_thresh = 0.22
        # self.point_feat_out_channels = 32
        # self.pred_layer_offset = PredBranch(self.point_feat_out_channels, 3)
        # self.pred_layer_hmap = nn.Sequential(
        #     PredBranch(self.point_feat_out_channels, 1),
        #     nn.Sigmoid()
        # )
        # self.MSDeformAttn = MSDeformAttn()
        # self.dense3d_img = nn.Sequential(
        #     SPCDense3Dv2(init_size=dense_f),
        #     dense_dropout_layer(dense3d_dropout),
        # )
        # self.CrossAttentionLayer = CrossAttentionLayer(d_model = 384, nhead = 4)
        # self.SelfAttentionLayer = SelfAttentionLayer(d_model = 384, nhead = 4)
        # self.to128 = nn.Linear(self.hidden_dim, self.mask_dim)
    def dense_bottleneck(
        self, deepest_features, bs, global_min_coords, global_max_coords,
    ):
        # l, bs, num_queries, dim

        scale = deepest_features.tensor_stride[0]
        max_coordinate = deepest_features.C[:, 1:].max(dim=0)[0].int()
        global_max_coords = torch.max(global_max_coords, max_coordinate)
        scene_size = (
            compute_scene_size(global_min_coords, global_max_coords, scale) // scale
        )

        dense_shape = torch.Size(
            (bs, deepest_features.shape[1], scene_size[0], scene_size[1], scene_size[2])
        )
        deepest_features_dense = deepest_features.dense(
            dense_shape, min_coordinate=torch.IntTensor([*global_min_coords])
        )[0]
        # if mode == "voxel":
        deepest_features_dense = self.dense3d(deepest_features_dense)
        # else:       #mode = image
        #     deepest_features_dense = self.dense3d_img(deepest_features_dense)

        deepest_features_t = ME.to_sparse(deepest_features_dense)

        coords = deepest_features_t.C.clone()

        coords[:, 1:] = coords[:, 1:] * scale + global_min_coords.reshape(1, -1)
        deepest_features = ME.SparseTensor(
            features=deepest_features_t.F,
            coordinates=coords,
            tensor_stride=scale,
            coordinate_manager=deepest_features.coordinate_manager,
        )

        return deepest_features

    def forward(
        self,
        in_feat,
        multi_scale_img_feat,
        pixel_coordinates,
        bs,
        Ts,
        global_min_coords,
        global_max_coords,
        min_Cs,
        max_Cs,
        class_frequencies,
        is_predict_panop=True,
        sem_labels=None,
        test=False,
    ):

        # encoder part
        encoders_features = self.encoder(in_feat)

        deepest_features = self.dense_bottleneck(
            encoders_features[-1],
            bs,
            global_min_coords=global_min_coords,
            global_max_coords=global_max_coords,
            # mode="voxel",
        )
        # deepest_img_features = self.dense_bottleneck(
        #     img_features,
        #     bs,
        #     global_min_coords=global_min_coords,
        #     global_max_coords=global_max_coords,
        #     mode="image",
        # )
        # # position encoding voxel
        # pe_voxel = deepest_features.C.reshape(-1, 4)
        # pe_voxel = self.pe_layer(pe_voxel[:, 1:])
        # pe_voxel = pe_voxel.reshape(-1, self.hidden_dim)
        # deepest_features_coord = deepest_features.C
        # deepest_features_feat = self.mask_feat_proj(deepest_features.F)

        # # position encoding img
        # pe_img = deepest_img_features.C.reshape(-1, 4)
        # pe_img = self.pe_layer_img(pe_img[:, 1:])
        # pe_img = pe_img.reshape(-1, self.hidden_dim)
        # deepest_img_features_coord = deepest_img_features.C
        # deepest_img_features_feat = self.mask_feat_proj_img(deepest_img_features.F)
        # # output = self.MSDeformAttn(query=deepest_features, value=deepest_img_features)
        # crossAtt_dense = self.CrossAttentionLayer(
        #     q_embed = deepest_features_feat, 
        #     bb_feat = deepest_img_features_feat, 
        #     pos = pe_img, 
        #     query_pos = pe_voxel
        # )
        # selfAtt_dense = self.SelfAttentionLayer(crossAtt_dense)
        # fused_feat = self.to128(selfAtt_dense)
        # fused_feat = ME.SparseTensor(fused_feat, 
        #     coordinate_map_key=deepest_features.coordinate_map_key,
        #     coordinate_manager=deepest_features.coordinate_manager
        #     ) 

        encoders_features = encoders_features[:-1]          #[sc1, sc2, sc4]
        
        """
        #pred center offset, choose scale 1_4 with channel 32:
        # pred_offset = self.pred_layer_offset(encoders_features[-1]).float().squeeze(-1).transpose(1, 2).contiguous()
        # pred_hmap = self.pred_layer_hmap(encoders_features[-1]).float().squeeze(1)
        # # pred_offset_high_conf = pred_offset.detach() * (pred_hmap > self.score_thresh).float()  
        # query_intermediate = [(pred_offset, pred_hmap)]
        """
        query_intermediate = None
        decoder_out = self.decoder_generative(
            deepest_features,           #original
            # fused_feat,
            multi_scale_img_feat,       #adding
            pixel_coordinates,          #adding
            encoders_features,
            query_intermediate,
            class_frequencies=class_frequencies,
            global_min_coords=global_min_coords,
            global_max_coords=global_max_coords,
            min_Cs=min_Cs,
            max_Cs=max_Cs,
            Ts=Ts,
            is_predict_panop=is_predict_panop,
            sem_labels=sem_labels,
            test=test,
        )

        return decoder_out
