from mayavi import mlab
try:
    engine = mayavi.engine
except NameError:
    from mayavi.api import Engine
    engine = Engine()
    engine.start()
from pasco.data.semantic_kitti.kitti_dm import KittiDataModule
from mayavi.mlab import *
import click
import numpy as np
import torch
from pasco.utils.torch_util import set_random_seed
import yaml

@click.command()
@click.option('--log_dir', default="logs", help='logging directory')
@click.option('--dataset_root', default="/gpfsdswork/dataset/SemanticKITTI")
@click.option('--config_path', default="semantic-kitti.yaml")
@click.option('--dataset_preprocess_root', default="/lustre/fsn1/projects/rech/kvd/uyl37fq/monoscene_preprocess/kitti")
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
@click.option('--data_aug', default=True, help='Use data augmentation if True')
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
@click.option('--f', default=32)
@click.option('--seed', default=42)
@click.option('--visualize_pcd', default=True)
@click.option('--visualize_grids', default=False)

    
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
    dataset_root, dataset_preprocess_root, config_path, n_fuse_scans, f,
    visualize_pcd, visualize_grids):
    
    
    def visualize_accumulate_pcd(xyz, labels):
        yaml_path = '/media/anda/hdd31/Phat/PaSCo/semantic-kitti.yaml'
        with open(yaml_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
        color_map = {k: np.array(v[::-1]) for k,v in yaml_data["color_map"].items()}
        
        X,Y,Z =  xyz[:, 0], xyz[: , 1], xyz[:, 2]
        unique_labels = np.unique(labels.astype(np.float32))
        # Gán màu ngẫu nhiên cho mỗi label
        # color_map = {label: (np.random.rand(3) * 255).astype(np.uint8) for label in unique_labels}
        # Chuyển label thành màu RGB
        # colors = np.array([color_map[label] for label in labels.flatten()])
        # colors = np.array(color_map.get(label) for label in labels)
        mlab.figure(size=(800, 800))
        nodes = mlab.points3d(X,Y,Z,labels.flatten(), mode="point")
        color_array = np.zeros((len(unique_labels), 4), dtype=np.uint8)
        for i, label in enumerate(unique_labels):
            color_array[i, :3] = color_map[label]  # RGB
            color_array[i, 3] = 255  # Alpha (độ trong suốt)

        # Ánh xạ màu sắc vào bảng màu
        nodes.module_manager.scalar_lut_manager.lut.table = color_array

        mlab.show()
        # show()
    def visualize_grid(grid):
        # Tạo figure
        mlab.figure(size=(800, 800))
        print("grid shape", grid.shape)
        # Hiển thị toàn bộ volume dưới dạng 3D
        # src = mlab.pipeline.scalar_field(grid)

        # Thêm các lát cắt theo 3 trục X, Y, Z
        # Vẽ tất cả lát cắt theo trục Z
        # for i in range(0, grid.shape[2], 2):  # Tăng step để tránh quá tải
        #     mlab.pipeline.image_plane_widget(src, plane_orientation='z_axes', slice_index=i)
        mlab.contour3d(grid.cpu().numpy(), contours=8, opacity=0.5)
        # Hiển thị tất cả lát cắt
        mlab.outline()
        mlab.show()
        
    set_random_seed(seed)
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
        )

    # print("bs", int(bs / n_gpus))
    # Checking key and value, shape of each key in dataset
    data_module.setup()
    train_loader = data_module.train_dataloader()
    batch_0 = next(iter(train_loader))
    # print("Number of samples in the batch:", len(batch_0), type(batch_0))
    # exit()
    # print("Shape of the first sample:", batch_0[0].shape)
    # print(batch_0['xyz'][0].shape)
    xyz = batch_0['xyz'][0]
    # grid = batch_0['sem_labels']['1_1'][0]
    grid = batch_0['semantic_label'][0]
    instance_label = batch_0['input_pcd_instance_label'][0].cpu().numpy()
    print("instance_label shape", instance_label.shape)
    if visualize_pcd:
        visualize_accumulate_pcd(xyz, instance_label)
    elif visualize_grids:
        visualize_grid(grid)
        


if __name__ == "__main__":
    main()