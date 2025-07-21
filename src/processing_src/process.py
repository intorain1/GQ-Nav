# import os
# import zipfile

# base_dir = "/media/mspx/Elements1/mp3d/v1/scans/"

# for root, dirs, files in os.walk(base_dir):
#     zip_name = "matterport_color_images.zip"
#     if zip_name in files:
#         extracted_dir = os.path.join(root, zip_name.replace(".zip", ""))  
#         if not os.path.exists(extracted_dir):
#             zip_path = os.path.join(root, zip_name)
#             print(f"unzip: {zip_path}")
#             with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#                 zip_ref.extractall(root)  
#         else:
#             print(f"pass: {root}")

import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

def combine_horizontally(images):
    if len(images) != 6:
        print("error! need six photos per time")
        return None

    min_height = min(img.shape[0] for img in images)
    resized = [cv2.resize(img, (int(img.shape[1] * min_height / img.shape[0]), min_height)) 
               for img in images]
    return cv2.hconcat(resized)

def process_scenes(folder_path, output_folder):

    os.makedirs(output_folder, exist_ok=True)
    
    all_files = glob(os.path.join(folder_path, "*_i*_*.jpg")) + \
                glob(os.path.join(folder_path, "*_i*_*.png"))
    scene_ids = set([os.path.basename(f).split('_')[0] for f in all_files])
    
    print(f"find {len(scene_ids)} scenes")
    
    for scene_id in tqdm(scene_ids, desc="processing"):
        groups = {'up': [], 'mid': [], 'down': []}
        for angle, prefix in [('up', 'i0'), ('mid', 'i1'), ('down', 'i2')]:
            for seq in range(6):
                img_path = os.path.join(folder_path, f"{scene_id}_{prefix}_{seq}.jpg")
                if not os.path.exists(img_path):
                    img_path = os.path.join(folder_path, f"{scene_id}_{prefix}_{seq}.png")
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        groups[angle].append(img)
                    else:
                        print(f"warning: {scene_id}_{prefix}_{seq} read error")
                else:
                    print(f"warning: {scene_id}_{prefix}_{seq} not exists")
        
        if all(len(imgs) == 6 for imgs in groups.values()):
            top_row = combine_horizontally(groups['up'])    
            mid_row = combine_horizontally(groups['mid'])    
            bottom_row = combine_horizontally(groups['down']) 
            
            if top_row is not None and mid_row is not None and bottom_row is not None:
                max_width = max(top_row.shape[1], mid_row.shape[1], bottom_row.shape[1])
                top_row = cv2.resize(top_row, (max_width, top_row.shape[0]))
                mid_row = cv2.resize(mid_row, (max_width, mid_row.shape[0]))
                bottom_row = cv2.resize(bottom_row, (max_width, bottom_row.shape[0]))
                
                final_image = cv2.vconcat([top_row, mid_row, bottom_row])
                output_path = os.path.join(output_folder, f"{scene_id}_COMBINED.jpg")
                cv2.imwrite(output_path, final_image)
                print(f"saved: {scene_id}_COMBINED.jpg")
            else:
                print(f"error: {scene_id} failed")
        else:
            print(f"warning: {scene_id} incompleted jupmed")


if __name__ == "__main__":
    # Get all directories in /media/mspx/Elements1/mp3d/v1/scans
    scan_root = "/media/mspx/Elements1/mp3d/v1/scans"
    scan_dirs = [d for d in os.listdir(scan_root) if os.path.isdir(os.path.join(scan_root, d))]
    for scan in scan_dirs:
        input_folder = os.path.join(scan_root, scan, scan, "matterport_color_images")
        output_folder = input_folder.replace("matterport_color_images", "combined_images")
        os.makedirs(output_folder, exist_ok=True)
        process_scenes(input_folder, output_folder)