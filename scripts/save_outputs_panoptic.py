import os
import click
import torch
import pickle

from pasco.data.semantic_kitti.kitti_dm import KittiDataModule
from pasco.data.semantic_kitti.params import class_frequencies
from pasco.utils.torch_util import enable_dropout
from pasco.models.net_panoptic_sparse import Net
from pasco.models.transform_utils import transform
from pasco.utils.torch_util import set_random_seed
from tqdm import tqdm


set_random_seed(42)

@click.command()
@click.option('--is_enable_dropout', default=False, help="run")
@click.option('--n_infers', default=1, help="run")
@click.option('--max_angle', default=30.0, help="")
@click.option('--translate_distance', default=0.2, help="")
@click.option('--iou_threshold', default=0.1, help="") # visualize better with 0.1 value
@click.option('--n_workers_per_gpu', default=3, help="Number of workers per GPU")
@click.option('--dataset_root', default="/media/anda/hdd3/Phat/PaSCo/gpfsdswork/semanticKITTI_full")
@click.option('--dataset_preprocess_root', default="/media/anda/hdd3/Phat/PaSCo/pasco_preprocess/kitti")
@click.option('--config_path', default="/media/anda/hdd3/Phat/PaSCo/semantic-kitti.yaml")
@click.option('--model_path', default="/media/anda/hdd3/Phat/PaSCo/logs/pasco_singlebs1_Fuse1_alpha0.0_wd0.0_lr0.0001_AugTrueR30.0T0.2S0.0_DropoutPoints0.05Trans0.2net3d0.0nLevels3_TransLay0Enc1Dec_queries100_maskWeight40.0_nInfers1_noHeavyDecoder/checkpoints/epoch=000-val_subnet1/pq_dagger_all=12.95338.ckpt")
@click.option('--model_name', default="file_output_without_image_branch_1")
@click.option('--frame_interval', default=5)
@click.option('--using_img', default=False)
@click.option('--rgb_img', default="P2")
def main(
        dataset_root, config_path, dataset_preprocess_root,
        n_workers_per_gpu, 
        max_angle, translate_distance, iou_threshold,
        is_enable_dropout, n_infers,
        frame_interval, model_path, model_name, using_img, rgb_img
):
    torch.set_grad_enabled(False)
    print(f"Model path: {model_path}")
    print(f"Model name: {model_name}") 
    frame_ids = list(range(0, 100, 5))              # ori 4080
    frame_ids = [int(frame_id) for frame_id in frame_ids]

    
    frame_ids = [ '{:06d}'.format(int(number)) for number in frame_ids]
    data_module = KittiDataModule(
        root=dataset_root,
        config_path=config_path,
        preprocess_root=dataset_preprocess_root,
        batch_size=1,
        num_workers=int(n_workers_per_gpu),
        n_subnets=n_infers,
        translate_distance=translate_distance,
        max_angle=max_angle,
        frame_interval=frame_interval,
        using_img=using_img,
        rgb_img=rgb_img,
    )
    
    data_module.setup_val_loader_visualization(frame_ids=frame_ids, 
                                               data_aug=True,
                                               max_items=None)
    


    model = Net.load_from_checkpoint(model_path, cfg=config_path,
                                     class_frequencies=class_frequencies, iou_threshold=iou_threshold)
    model.cuda()
    model.eval()
    if is_enable_dropout:
        enable_dropout(model)

    
    dataset = data_module.val_ds

    for i in tqdm(range(len(dataset))):
        batch = dataset[i]
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].cuda()
            elif isinstance(batch[k], list):
                for j in range(len(batch[k])):
                    if isinstance(batch[k][j], torch.Tensor):
                        batch[k][j] = batch[k][j].cuda()
                
        panop_outs, gt_panoptic_segs, gt_segments_infos, ssc_preds = model.step_inference(batch, "test", draw=True)
        frame_id = batch['frame_id'][0]
        semantic_label_origins = batch['semantic_label_origin'][0]
        instance_label_origins = batch['instance_label_origin'][0]
        
        input_coords = batch['in_coords'][0]
        T = batch['Ts'][0]
        input_coords = transform(input_coords, torch.inverse(T))
        xyz = batch['xyz'][0]



        for i_infer in [n_infers]:
            gt_panoptic_seg = gt_panoptic_segs[i_infer]
            gt_segments_info = gt_segments_infos[i_infer]
            ssc_pred = ssc_preds[i_infer]
            
            panop_out = panop_outs[i_infer]
            panoptic_seg_denses = panop_out['panoptic_seg_denses']
            semantic_seg_denses = panop_out['semantic_seg_denses']
            segments_infos = panop_out['segments_infos']
            vox_confidence_denses = panop_out['ssc_confidence'].unsqueeze(0)
            instance_confidence_denses = panop_out['ins_uncertainty_denses']
            
            
            for segments_info in segments_infos[0]:
                for k in segments_info:
                    if isinstance(segments_info[k], torch.Tensor):
                        segments_info[k] = segments_info[k].cpu().numpy()
    
            subnet_out = {
                "ssc_pred": ssc_pred,
                "pred_panoptic_seg": panoptic_seg_denses.detach().cpu().numpy(),
                "pred_segments_info": segments_infos,
                "vox_confidence_denses": vox_confidence_denses.detach().cpu().numpy(),
                "instance_confidence_denses": instance_confidence_denses.detach().cpu().numpy(),
                "xyz": xyz,
                "gt_panoptic_seg": gt_panoptic_seg,
                "gt_segments_info": gt_segments_info,
                "semantic_label_origin": semantic_label_origins.cpu().numpy(),
                "instance_label_origin": instance_label_origins.cpu().numpy(),
            }
            
            # model_name = "file_output_without_image_branch_1"
            save_dir = os.path.join("output", model_name)
            os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(save_dir, "{}_{}.pkl".format(frame_id, i_infer))
            with open(filepath, "wb") as handle:
                pickle.dump(subnet_out, handle)
                print("wrote to", filepath)
                



if __name__ == "__main__":
    main()
