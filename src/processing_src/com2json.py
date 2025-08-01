import sys
sys.path.append('/home/mspx/icra/GQ-Nav/src')
from semantic import SemanticSegmenter
import numpy as np
import cv2
from PIL import Image
import os
import json
import matplotlib.pyplot as plt
import tqdm

def read_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # Join lines with '. ' and add a trailing '.' if not present
    joined = '. '.join(line.strip() for line in lines if line.strip())
    if not joined.endswith('.'):
        joined += '.'
    return joined

if __name__ == "__main__":
    nodespace = '"' + read_txt('/home/mspx/icra/GQ-Nav/src/processing_src/object.txt') + '"'

    segmenter = SemanticSegmenter(
        grouding_model_config='/home/mspx/icra/GQnav/third_party/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
        grounding_model_checkpoint='/home/mspx/icra/GQnav/model/groundingdino_swint_ogc.pth',
        sam_version='vit_b',
        sam_checkpoint='/home/mspx/icra/GQnav/model/sam_vit_b_01ec64.pth',
        node_space=nodespace,
        # node_space='table. tv. chair. cabinet. sofa. bed. windows. kitchen. bedroom. living room. mirror. plant. curtain. painting. picture',
        # node_space="table. tv. window. chair. refrigerator. sports. bench. piano. table. fireplace. couch. rug. desk. doors. staircase. sink. toilet.",
        device='cuda'
    )
    scan_root = "/media/mspx/Elements1/mp3d/v1/scans"
    scan_dirs = [d for d in os.listdir(scan_root) if os.path.isdir(os.path.join(scan_root, d))]
    folders = [os.path.join(scan_root, scan, scan, "combined_images") for scan in scan_dirs]
    output_json_dir = '/home/mspx/icra/GQ-Nav/datasets/R2R/objects_list'
    for folder in tqdm.tqdm(folders, desc="Processing folders"):
        scan = os.path.basename(folder.split(os.sep)[-2])

        test_image_dir = folder

        image_files = [os.path.join(test_image_dir, f) for f in os.listdir(test_image_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        results = {}

        for image_file in image_files:
            basename = os.path.basename(image_file)
            scene_id = basename.lower().split('_combined')[0]

            depth_image_file = image_file.replace('combined_images', "combined_depth__mages")
            image = np.array(Image.open(image_file))
            depth = np.array(Image.open(depth_image_file))
            mask, xyxy, conf, caption = segmenter.get_segmentation(image)
            print(depth.shape)
            scene_list = []
            for i in range(len(caption)):
                entry = {}
                if conf[i] > 0.9:
                    obj_mask = mask[i]
                    ys, xs = np.where(obj_mask)
                    if len(xs) == 0 or len(ys) == 0:
                        continue 
                    cx = int(np.mean(xs))
                    cy = int(np.mean(ys))
                    # Calculate heading for panorama: map x position to [0, 360) degrees
                    img_width = image.shape[1]
                    heading = (cx / img_width) * 360.0 - 180
                    # Get distance from depth image at centroid
                    # If depth image has 3 channels, take the first channel
                    if depth.ndim == 3 and depth.shape[2] == 3:
                        distance = float(depth[int(cy), int(cx), 0]) * 25 / 1000
                    else:
                        distance = float(depth[int(cy), int(cx)])
                    entry[caption[i]] = {
                        "heading": float(heading),
                        "distance": distance
                    }
                scene_list.append(entry)
            results[scene_id] = scene_list

        json_path = os.path.join(output_json_dir, f"{scan}.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
