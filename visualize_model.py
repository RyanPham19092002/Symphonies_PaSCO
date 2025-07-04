import torch

# Load checkpoint
ckpt_path = "/media/anda/hdd3/Phat/PaSCo/logs/pasco_singlebs1_Fuse1_alpha0.0_wd0.0_lr0.0001_AugTrueR30.0T0.2S0.0_DropoutPoints0.05Trans0.2net3d0.0nLevels3_TransLay0Enc1Dec_queries100_maskWeight40.0_nInfers1_noHeavyDecoder_withImg_10_aug_img_Att_img_branch/checkpoints/epoch=030-val_subnet1/pq_dagger_all=23.26874.ckpt"
checkpoint = torch.load(ckpt_path, map_location='cpu')

# Kiểm tra các key chính trong checkpoint
print("Các khóa trong checkpoint:")
print(checkpoint.keys())

# Nếu là Lightning, state_dict thường nằm trong 'state_dict'
state_dict = checkpoint.get('state_dict', checkpoint)

print("\nCác layer và kích thước weight:")
for k, v in state_dict.items():
    print(f"{k:60s} => {tuple(v.shape)}")
