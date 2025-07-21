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

            image = np.array(Image.open(image_file))
            mask, xyxy, conf, caption = segmenter.get_segmentation(image)

            scene_list = []
            for i in range(len(caption)):
                entry = {}
                if conf[i] > 0.9:
                    entry[caption[i]] = {
                        "heading": float(np.random.uniform(-180, 180)),
                        "distance": float(np.random.uniform(0.5, 2.0))
                    }
                scene_list.append(entry)
            results[scene_id] = scene_list

        json_path = os.path.join(output_json_dir, f"{scan}.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
