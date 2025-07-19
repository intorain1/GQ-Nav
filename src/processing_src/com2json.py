from semantic import SemanticSegmenter
import numpy as np
import cv2
from PIL import Image
import os
import json
import matplotlib.pyplot as plt
import tqdm

if __name__ == "__main__":
    segmenter = SemanticSegmenter(
        grouding_model_config='/home/mspx/icra/GQnav/third_party/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
        grounding_model_checkpoint='/home/mspx/icra/GQnav/model/groundingdino_swint_ogc.pth',
        sam_version='vit_b',
        sam_checkpoint='/home/mspx/icra/GQnav/model/sam_vit_b_01ec64.pth',
        # node_space='table. tv. chair. cabinet. sofa. bed. windows. kitchen. bedroom. living room. mirror. plant. curtain. painting. picture',
        node_space="table. tv. window. chair. refrigerator. sports. bench. piano. table. fireplace. couch. rug. desk. doors. staircase. sink. toilet.",
        device='cuda'
    )
    test_image_dir = '/home/mspx/icra/GQnav/test_image'
    image_files = [os.path.join(test_image_dir, f) for f in os.listdir(test_image_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    results = {}

    for image_file in tqdm.tqdm(image_files, desc="Processing images"):
        basename = os.path.basename(image_file)
        scene_id = basename.lower().split('_combined')[0]

        image = np.array(Image.open(image_file))
        mask, xyxy, conf, caption = segmenter.get_segmentation(image)

        # Example: simulate heading/distance for each detection
        scene_list = []
        for i in range(len(caption)):
            entry = {}
            if conf[i] > 0.85:
                # Dummy heading/distance, replace with your actual calculation
                entry[caption[i]] = {
                    "heading": float(np.random.uniform(-180, 180)),
                    "distance": float(np.random.uniform(0.5, 2.0))
                }
            scene_list.append(entry)
        results[scene_id] = scene_list

    # Save all results to a single JSON file
    json_path = os.path.join(test_image_dir, "scenes.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
