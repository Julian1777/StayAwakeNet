import os
import json
import glob
import shutil

img_dir = os.path.join('dataset', 'original', 'images')
anno_dir = os.path.join('dataset', 'original', 'annotations')

final_img_dir = os.path.join('dataset', 'images')
final_anno_dir = os.path.join('dataset', 'annotations')

os.makedirs(final_img_dir, exist_ok=True)
os.makedirs(final_anno_dir, exist_ok=True)

class_map = {
    "hand-hold": 0,
    "hand-no-hold": 1,
    "steering-wheel": 2
}

anno_files = sorted(glob.glob(os.path.join(anno_dir, '*.json')))

for idx, anno_path in enumerate(anno_files, 1):
    with open(anno_path, 'r') as f:
        data = json.load(f)

    img_name = data['key']
    img_path = os.path.join(img_dir, img_name)
    new_img_name = f'image_{idx:03d}.jpg'
    new_anno_name = f'anno_{idx:03d}.txt'

    shutil.copy(img_path, os.path.join(final_img_dir, new_img_name))

    