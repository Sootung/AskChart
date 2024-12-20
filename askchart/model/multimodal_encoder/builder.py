import os
from .clip_encoder import CLIPVisionTower
# from moellava.train.train import rank0_print
import transformers
a, b, c = transformers.__version__.split('.')[:3]
if a == '4' and int(b) >= 37:
    from .siglip_encoder import SiglipVisionTower
# from .languagebind import LanguageBindImageTower, LanguageBindVideoTower
import ipdb

# ============================================================================================================

def build_image_tower(image_tower_cfg, **kwargs):
    image_tower = getattr(image_tower_cfg, 'mm_image_tower', getattr(image_tower_cfg, 'image_tower', None))
    is_absolute_path_exists = os.path.exists(image_tower)
    # rank0_print(f"original image_tower: {image_tower}, is_absolute_path_exists: {is_absolute_path_exists}")
    abs_path = '/hpc2hdd/home/chartmoe/sootung/models'
    abs_image_tower = os.path.join(abs_path, image_tower) if abs_path else image_tower
    # print(f"abs_image_tower: {abs_image_tower}, abs_path: {abs_path}")
    # Determine which image tower to load
    if 'checkpoints' in image_tower:
        selected_image_tower = image_tower
    elif image_tower.startswith("openai") or image_tower.startswith("google"):
        selected_image_tower = abs_image_tower
    
    print(f"selected_image_tower: {selected_image_tower}")
    # ipdb.set_trace()
    # add absolute path and image_tower

    if image_tower.startswith("openai") or image_tower.startswith("laion"): # or "moe-v2" in selected_image_tower.lower() 
        return CLIPVisionTower(selected_image_tower, args=image_tower_cfg, cache_dir='./cache_dir', **kwargs)
    if is_absolute_path_exists or image_tower.startswith("google"):
        return SiglipVisionTower(selected_image_tower, args=image_tower_cfg, cache_dir='./cache_dir', **kwargs)
    if image_tower.endswith('LanguageBind_Image'):
        return LanguageBindImageTower(image_tower, args=image_tower_cfg, cache_dir='./cache_dir', **kwargs)

    raise ValueError(f'Unknown image tower: {image_tower}')
# ============================================================================================================


def build_video_tower(video_tower_cfg, **kwargs):
    video_tower = getattr(video_tower_cfg, 'mm_video_tower', getattr(video_tower_cfg, 'video_tower', None))
    if video_tower.endswith('LanguageBind_Video_merge'):
        return LanguageBindVideoTower(video_tower, args=video_tower_cfg, cache_dir='./cache_dir', **kwargs)
    raise ValueError(f'Unknown video tower: {video_tower}')
# ============================================================================================================
