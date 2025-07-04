import os
from pasco.data.semantic_kitti.kitti_dm import KittiDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import click
import numpy as np
import torch
from pasco.data.semantic_kitti.params import class_names, class_frequencies
from pasco.models.net_panoptic_sparse import Net
from pytorch_lightning.strategies import DDPStrategy
from pasco.utils.torch_util import set_random_seed
from pytorch_lightning.plugins.environments import SLURMEnvironment
import signal

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import time
# from torch import Tensor
# from visualize_pcl import visualize_accumulate_pcd





@click.command()
@click.option('--log_dir', default="logs", help='logging directory')
@click.option('--dataset_root', default="/media/anda/hdd3/Phat/PaSCo/gpfsdswork/semanticKITTI")
@click.option('--config_path', default="semantic-kitti.yaml")
@click.option('--config_img_path', default="/media/anda/hdd3/Phat/PaSCo/cfg/pa_po_kitti_val.yaml")
@click.option('--dataset_preprocess_root', default="/media/anda/hdd3/Phat/PaSCo/pasco_preprocess/kitti")
@click.option('--n_infers', default=1, help='number of subnets')

@click.option('--lr', default=3e-4, help='learning rate')
@click.option('--wd', default=0.0, help='weight decay')
@click.option('--bs', default=1, help="batch size")
@click.option('--scale', default=1, help="Scale")
@click.option('--n_gpus', default=2, help="number of GPUs")
@click.option('--n_workers_per_gpu', default=3, help="Number of workers per GPU")
@click.option('--exp_prefix', default="exp", help='prefix of logging directory')
@click.option('--enable_log', default=True, help='Enable logging')

@click.option('--transformer_dropout', default=0.2)
@click.option('--net_3d_dropout', default=0.0)
@click.option('--n_dropout_levels', default=3)

@click.option('--max_angle', default=5.0, help='random augmentation angle from -max_angle to max_angle')
@click.option('--translate_distance', default=0.2, help='randomly translate 3D scene in 3 dimensions: np.array([3.0, 3.0, 2.0]) * translate_distance')
@click.option('--point_dropout_ratio', default=0.05, help='randomly drop from 0 to 5% points in 3D input')
@click.option('--data_aug', default=False, help='Use data augmentation if True')
@click.option('--scale_range', default=0.0, help='random scaling the scene')

@click.option('--alpha', default=0.0, help='uncertainty weight')


@click.option('--transformer_enc_layers', default=0, help='Transformer encoder layer')
@click.option('--transformer_dec_layers', default=1, help='Transformer decoder layer')

@click.option('--num_queries', default=100, help='Number of queries')
@click.option('--mask_weight', default=40.0, help='mask weight')
@click.option('--occ_weight', default=1.0, help='mask weight')


@click.option('--use_se_layer', default=False, help='mask weight')
@click.option('--heavy_decoder', default=False, help='mask weight')

@click.option('--use_voxel_query_loss', default=True, help='uncertainty weight')

@click.option('--accum_batch', default=1, help='') # Quite slow to train, test later
@click.option('--n_fuse_scans', default=1, help='#scans to fuse')

@click.option('--pretrained_model', default="")
@click.option('--f', default=64)
@click.option('--seed', default=42)
@click.option('--using_img', default=False)
@click.option('--version_model', default="0")
@click.option('--max_epochs', default=35)
@click.option('--rgb_img', default="P2")
@click.option('--aux_loss', default=False)
def main(
    lr, wd, 
    bs, scale, alpha,
    n_workers_per_gpu, n_gpus,
    exp_prefix, log_dir, enable_log,
    data_aug, mask_weight, heavy_decoder,
    transformer_dropout, net_3d_dropout, n_dropout_levels,
    point_dropout_ratio, use_voxel_query_loss,
    max_angle, translate_distance, scale_range, seed,
    transformer_dec_layers, transformer_enc_layers, n_infers, occ_weight,
    num_queries, use_se_layer, accum_batch, pretrained_model,
    dataset_root, dataset_preprocess_root, config_path, n_fuse_scans, f, config_img_path, using_img, version_model, max_epochs, rgb_img,aux_loss):
##-----------------------------------------------------------   
    def create_colormap(num_classes):
        # 20 màu đã chọn kỹ, tránh trùng lặp/na ná nhau
        colors = [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Yellow
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Cyan
            (255, 165, 0),    # Orange
            (128, 0, 128),    # Purple
            (0, 128, 128),    # Teal
            (128, 128, 0),    # Olive
            (0, 0, 128),      # Navy
            (128, 0, 0),      # Maroon
            (0, 128, 0),      # Dark Green
            (192, 192, 192),  # Silver
            (128, 128, 128),  # Gray
            (255, 192, 203),  # Pink
            (210, 105, 30),   # Chocolate
            (75, 0, 130),     # Indigo
            (60, 179, 113),   # Medium Sea Green
            (244, 164, 96)    # Sandy Brown
        ]
        # colormap = {}
        # for label in range(num_classes):
        #     colormap[label] = colors[label]
        # return colormap
        return colors

    def overlay_points_on_image(image, uv_points, labels, colors, alpha=0.3, point_radius=20):
        overlay = image.copy()
        for (u,v), label in zip(uv_points, labels):
            # if label == 9:
            # u = int(round(u))
            # v = int(round(v))
            u_int = int(u)
            v_int = int(v)
            if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                # color = colormap.get(label,(255,255,255))
                color = colors[label]
                # cv2.circle(overlay, (u,v), point_radius, color, -1)
                overlay[v_int, u_int] = color
        blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        return blended
##----------------------------------------------------------- 
    set_random_seed(seed)
    
    encoder_dropouts = [point_dropout_ratio, 0.0, 0.0, 0.0, 0.0, 0.0]
    decoder_dropouts = [0.0, 0.0, 0.0, 0.0, 0.0]
    for l in range(n_dropout_levels):
        encoder_dropouts[len(encoder_dropouts) - l - 1] = net_3d_dropout
        decoder_dropouts[l] = net_3d_dropout

    print("max_epochs", max_epochs)
    print("log_dir", log_dir)
    exp_name = exp_prefix
    
    exp_name += "bs{}_Fuse{}".format(bs, n_fuse_scans)
    exp_name += "_alpha{}_wd{}_lr{}_Aug{}R{}T{}S{}_DropoutPoints{}Trans{}net3d{}nLevels{}".format(
        alpha, wd, lr, 
        data_aug, max_angle, translate_distance, scale_range,
        point_dropout_ratio, transformer_dropout, net_3d_dropout, n_dropout_levels)
    exp_name += "_TransLay{}Enc{}Dec_queries{}".format(transformer_enc_layers, transformer_dec_layers, num_queries)
    exp_name += "_maskWeight{}".format(mask_weight)
    
    if occ_weight != 1.0:
        exp_name += "_occWeight{}".format(occ_weight) 
    
    exp_name += "_nInfers{}".format(n_infers)
    
    if not use_voxel_query_loss:
        exp_name += "_noVoxelQueryLoss"
    if not heavy_decoder:
        exp_name += "_noHeavyDecoder"

    if not using_img:
        exp_name += "_without_Img"
    else:
        exp_name += "_withImg"
    
    if version_model:
        exp_name += f"_{version_model}"


    query_sample_ratio = 1.0
    print("n_workers_per_gpu", n_workers_per_gpu)
    print(exp_name)
    print("data_aug", data_aug)

    n_classes = 20
    
    class_weights = []

    for _ in range(n_infers):
        class_weight = torch.ones(n_classes + 1)
        class_weight[0] = 0.1
        class_weight[-1] = 0.1 # dustbin class
        class_weights.append(class_weight)

    complt_num_per_class = class_frequencies["1_1"]
    compl_labelweights = complt_num_per_class / np.sum(complt_num_per_class)
    compl_labelweights = np.power(np.amax(compl_labelweights) / compl_labelweights, 1 / 3.0)
    compl_labelweights = torch.from_numpy(compl_labelweights).float()
    
    # start_data_module = time.time()
    data_module = KittiDataModule(
        root=dataset_root,
        config_path=config_path,
        preprocess_root=dataset_preprocess_root,
        batch_size=int(bs / n_gpus),
        num_workers=int(n_workers_per_gpu),
        data_aug=data_aug,
        max_angle=max_angle,
        translate_distance=translate_distance,
        scale_range=scale_range,
        max_val_items=None,
        n_fuse_scans=n_fuse_scans,
        n_subnets=n_infers,
        using_img=using_img,
        rgb_img=rgb_img,
    )
    # print("-----------------------Time-----------------------")
    # print("Time data module: ", time.time() - start_data_module)
    # data_module.setup()
    # print("data_module", len(data_module.train_ds), len(data_module.val_ds))
    # exit()
    # print("bs", int(bs / n_gpus))
    # # Checking key and value, shape of each key in dataset
    ##---------------------------checking map---------------------------
    # import itertools
    # data_module.setup()
    # train_loader = data_module.train_dataloader()
    # # # print(train_loader)
    # batch = next(itertools.islice(train_loader, 10, None))
    # # for item, batch in enumerate(train_loader):
    # #     print("item", item)
    # #     print("Batch keys:", batch.keys())
    # image = batch["ori_camera"][0]
    # h, w , _ = image.shape
    # print("h,w", h, w)
    # uv_points = batch["pixel_coordinates"][0]
    # uv_points[:, 0] = (uv_points[:, 0] + 1)/2*(w-1)
    # uv_points[:, 1] = (uv_points[:, 1] + 1)/2*(h-1)
    # print("x", uv_points[:, 0])
    # print("y", uv_points[:, 1])
    # num_points = len(uv_points)
    # num_classes = 20
    # labels = batch["input_pcd_semantic_label"][0]
    # uv_label_data = np.column_stack((uv_points, labels))
    # colormap = create_colormap(num_classes)
    # blended_img = overlay_points_on_image(
    #     image,
    #     uv_points,
    #     labels,
    #     colormap
    # )
    # save_path = "/media/anda/hdd3/Phat/PaSCo/check_mapping/mapping.png"
    # save_txt_path = '/media/anda/hdd3/Phat/PaSCo/check_mapping/uv_points_labels.txt'
    # # np.savetxt(save_txt_path, uv_label_data, fmt="%.4f %.4f %d")
    # plt.figure(figsize=(12, 10))
    # plt.imshow(blended_img.astype(np.uint8))
    # plt.axis('off')
    # # plt.title('Overlay 2D Points with Semantic Labels')
    # plt.show()
    # plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    # exit()
    ##---------------------------checking map---------------------------
        # for key, value in batch.items():
            

    # exit()
    # batch_0 = next(iter(train_loader))
    # print("Number of samples in the batch:", len(batch_0), type(batch_0))
    # # print("batch", batch_0)
    # for key, value in batch_0.items():
    #     if key == "input_pcd_semantic_label" or key == "input_pcd_instance_label":
    #         print("key",key)
    #         print("value", value)
    # exit()
    # print("Shape of the first sample:", batch_0[0].shape)
    # print(batch_0['xyz'][0].shape)
    # xyz = batch_0['xyz'][0]

   
    # with open("3_frame.txt", "w") as file:
    #     for key, value in batch_0.items():
    #         file.write(f"key: {key}\n")
    #         file.write(f"value: {value}\n\n")
    #         if isinstance(value, dict):
    #             file.write(f"Dictionary has {len(value)} keys\n")
    #             # Nếu muốn in thông tin kích thước của các giá trị bên trong dictionary:
    #             for sub_key, sub_value in value.items():
    #                 file.write(f"  - key: {sub_key}, ")
    #                 if hasattr(sub_value[0], "shape"):
    #                     file.write(f"shape: {sub_value[0].shape}\n")
    #                 else:
    #                     file.write(f"value size: {len(sub_value) if hasattr(sub_value, '__len__') else 'N/A'}\n")
    #         elif isinstance(value[0], np.ndarray):
    #             file.write(f"Length of list: {len(value)}\n\n")
    #             file.write(f"Array shape: {value[0].shape}\n")
    #         elif isinstance(value[0], Tensor):
    #             file.write(f"Length of list: {len(value)}\n\n")
    #             file.write(f"Array shape: {value[0].shape}\n")
    #         elif isinstance(value, Tensor):
    #             file.write(f"Length of list: {len(value)}\n\n")
    #             file.write(f"Array shape: {value[0].shape}\n")
    #         elif isinstance(value[0], dict):
    #             file.write(f"Dictionary has {len(value[0])} keys\n")
    #             # Nếu muốn in thông tin kích thước của các giá trị bên trong dictionary:
    #             for sub_key, sub_value in value[0].items():
    #                 file.write(f"  - key: {sub_key}, ")
    #                 if hasattr(sub_value[0], "shape"):
    #                     file.write(f"shape: {sub_value[0].shape}\n")
    #                 else:
    #                     file.write(f"value size: {len(sub_value) if hasattr(sub_value, '__len__') else 'N/A'}\n")

    # exit()
    model = Net(
        cfg = config_img_path,
        heavy_decoder=heavy_decoder,
        class_frequencies=class_frequencies,
        n_classes=n_classes,
        occ_weight=occ_weight,
        class_names=class_names,
        lr=lr,
        weight_decay=wd,
        class_weights=class_weights,
        transformer_dropout=transformer_dropout,
        encoder_dropouts=encoder_dropouts,
        decoder_dropouts=decoder_dropouts,
        dense3d_dropout=net_3d_dropout,
        scale=scale, # not use
        enc_layers=transformer_enc_layers,
        dec_layers=transformer_dec_layers,
        aux_loss=aux_loss,
        num_queries=num_queries,
        mask_weight=mask_weight,
        use_se_layer=use_se_layer,
        alpha=alpha,
        query_sample_ratio=query_sample_ratio,
        n_infers=n_infers,
        f=f,
        compl_labelweights=compl_labelweights,
        use_voxel_query_loss=use_voxel_query_loss,   
        using_img=using_img,
    )
    
    

    if enable_log:
        logger = TensorBoardLogger(save_dir=log_dir, name=exp_name, version="")
        lr_monitor = LearningRateMonitor(logging_interval="step")
        
        checkpoint_callbacks = [
            ModelCheckpoint(
                save_last=True,
                monitor="val_subnet" + str(n_infers) + "/pq_dagger_all",
                save_top_k=50,
                mode="max",
                filename="{epoch:03d}-{val_subnet" + str(n_infers) + "/pq_dagger_all:.5f}",
            ),
            lr_monitor,
        ]
    else:
        logger = False
        checkpoint_callbacks = False

    model_path = os.path.join(log_dir, exp_name, "checkpoints/last.ckpt")
    if not os.path.isfile(model_path) and pretrained_model != "":
        assert os.path.isfile(pretrained_model), "Pretrained model not found"
        model = Net.load_from_checkpoint(checkpoint_path=pretrained_model)
        print("Load pretrained model from {}".format(model_path))

    if os.path.isfile(model_path):
        
        trainer = Trainer(
            accumulate_grad_batches=accum_batch,
            limit_val_batches=0.25 * accum_batch * n_gpus,
            limit_train_batches=0.25 * accum_batch * n_gpus,
            callbacks=checkpoint_callbacks,
            resume_from_checkpoint=model_path,            
            max_epochs=max_epochs,
            gradient_clip_val=0.5,
            logger=logger,
            check_val_every_n_epoch=1,
            accelerator="gpu",
            strategy=DDPStrategy(find_unused_parameters=True),
            num_nodes=1,
            devices=n_gpus,
            sync_batchnorm=True,
            # plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],
        )
    else:
        # Train from scratch
        # print("0.25 * accum_batch * n_gpus,",0.25 * accum_batch * n_gpus)
        trainer = Trainer(
            accumulate_grad_batches=accum_batch,
            limit_val_batches=0.25 * accum_batch * n_gpus,
            limit_train_batches=0.25 * accum_batch * n_gpus,
            callbacks=checkpoint_callbacks,            
            max_epochs=max_epochs,
            gradient_clip_val=0.5,
            logger=logger,
            strategy=DDPStrategy(find_unused_parameters=True),
            check_val_every_n_epoch=1,
            accelerator="gpu",
            devices=n_gpus,
            num_nodes=1,
            sync_batchnorm=True,
            # plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)]
        )
    start_train_fit = time.time()
    trainer.fit(model, data_module)
    print("time train fit: ", time.time() - start_train_fit)

if __name__ == "__main__":
    main()
