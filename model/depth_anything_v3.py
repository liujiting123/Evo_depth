import torch
import torch.nn as nn
import torch.nn.functional as F
from model.depth_anything_3.api import DepthAnything3
import glob
import os

    
class DAV3Module(nn.Module):
    def __init__(self, config=None):
        super(DAV3Module, self).__init__()
        checkpoint_dir = os.environ.get(
            "EVO1_DA3_CHECKPOINT_DIR",
            "/home/bozhao/code/ljt/Evo_da3_film_true_else/checkpoints/",
        )
        self.model = DepthAnything3.from_pretrained(checkpoint_dir, local_files_only=True)
        
        

    def inference_images(self, images):
        # images: List of image paths
        return self.model.inference(images)
    
    def extract_features(self, image_tensor,image_mask = None):
        feats = self.model._get_dpt_embeddings(image_tensor)
        assert image_mask is not None, "Image mask must be provided for feature extraction."
        final_feats = []
        for i in range(len(image_mask)):
            if image_mask[i] == True:
                final_feats.append(feats[:, i, :, :])
        
        return torch.stack(final_feats, dim=1) if final_feats else feats


    def forward(self, image_tensor, image_mask):
        
        feats = self.extract_features(image_tensor, image_mask)
        return feats

if __name__ == "__main__":
    model = DAV3Module().to("cuda")
    example_path = "/home/bozhao/code/ljt/Depth-Anything-3/assets/examples/SOH/"
    images = sorted(glob.glob(os.path.join(example_path, "*.png")))
    prediction = model.inference_images(
        images,
    )
    # prediction.processed_images : [N, H, W, 3] uint8   array
    print(prediction.processed_images.shape)
    # prediction.depth            : [N, H, W]    float32 array
    print(prediction.depth.shape)  
    # prediction.conf             : [N, H, W]    float32 array
    print(prediction.conf.shape)  
    # prediction.extrinsics       : [N, 3, 4]    float32 array # opencv w2c or colmap format
    print(prediction.extrinsics.shape)
    # prediction.intrinsics       : [N, 3, 3]    float32 array
    print(prediction.intrinsics.shape)


    import matplotlib.pyplot as plt
    import numpy as np

    depth = prediction.depth  # [N, H, W]

    i = 0  # 看第 i 张
    d = depth[i]

    # 去掉无效值（可选，但强烈建议）
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    # 归一化到 0~1
    d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)

    plt.figure(figsize=(6, 6))
    plt.imshow(d_norm, cmap="magma")  # magma / viridis / inferno 都很好
    plt.colorbar(label="normalized depth")
    plt.title("Depth map")
    plt.axis("off")
    plt.show()
