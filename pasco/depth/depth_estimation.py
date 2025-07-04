import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image

image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")
def depth_estimation(img):
        inputs = image_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        post_processed_output = image_processor.post_process_depth_estimation(
                            outputs,
                            target_sizes=[(img.size[1], img.size[0])],
                        )
        predicted_depth = post_processed_output[0]["predicted_depth"]
        depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
        depth = depth.detach().cpu().numpy() * 255

        #create image to visualize
        depth_rgb = Image.fromarray(depth.astype("uint8"))
        depth_rgb = depth_rgb.convert("RGB")
        # colormap = plt.get_cmap('jet')
        # depth_colored = colormap(depth / 255.0)  # ra shape (H, W, 4) RGBA

        # # Bỏ kênh alpha và chuyển sang uint8
        # depth_colored = (depth_colored[..., :3] * 255).astype(np.uint8)

        # # Lưu với PIL
        # depth_img = Image.fromarray(depth_colored)
        depth_rgb.save("depth.png")
        return depth