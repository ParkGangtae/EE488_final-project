# main.py

from vision_transformer import VisionTransformer

def MainModel(nOut=256, **kwargs):
    # 커스텀 Vision Transformer 모델
    return VisionTransformer(img_size=224, patch_size=32, in_channels=3, num_classes=nOut)
