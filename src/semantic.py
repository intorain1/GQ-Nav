import sys
sys.path.append('/home/mspx/icra/GQnav/third_party/Grounded-Segment-Anything/')
from grounded_sam_demo import load_model, get_grounding_output
import GroundingDINO.groundingdino.datasets.transforms as T
from segment_anything import sam_model_registry, SamPredictor

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt

class SemanticSegmenter:
    def __init__(self, grouding_model_config, grounding_model_checkpoint,
                 sam_version, sam_checkpoint, node_space, device='cuda'):
        self.device = device
        self.groundingdino_config_file = grouding_model_config
        self.groundingdino_checkpoint = grounding_model_checkpoint
        self.sam_version = sam_version
        self.sam_checkpoint = sam_checkpoint
        self.node_space = node_space
        self.model, self.predictor = self.get_grounded_sam(device)

    def get_grounded_sam(self, device):
        model = load_model(self.groundingdino_config_file, self.groundingdino_checkpoint, device=device)
        predictor = SamPredictor(sam_model_registry[self.sam_version](checkpoint=self.sam_checkpoint).to(device))
        return model, predictor
    
    def get_segmentation(
        self, image: np.ndarray
    ) -> tuple:
        groundingdino = self.model
        sam_predictor = self.predictor
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_resized, _ = transform(Image.fromarray(image), None)  # 3, h, w
        boxes_filt, caption = get_grounding_output(groundingdino, image_resized, caption=self.node_space, box_threshold=0.3, text_threshold=0.25, with_logits=False, device=self.device)
        if len(caption) == 0:
            return None, None, None, None
        sam_predictor.set_image(image)

        # size = image_pil.size
        H, W = image.shape[0], image.shape[1]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)

        mask, conf, _ = sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(self.device),
            multimask_output = False,
        )
        mask, xyxy, conf = mask.squeeze(1).cpu().numpy(), boxes_filt.squeeze(1).numpy(), conf.squeeze(1).cpu().numpy()
        return mask, xyxy, conf, caption
    
    def visualize_segmentation(
        self, 
        image: np.ndarray, 
        masks: np.ndarray, 
        boxes: np.ndarray, 
        confidences: np.ndarray, 
        captions: list
    ) -> Image.Image:
        """
        Visualize segmentation results with large text
        
        Args:
            image: Original RGB image as numpy array
            masks: Array of segmentation masks (N, H, W)
            boxes: Array of bounding boxes (N, 4) in [x1, y1, x2, y2] format
            confidences: Array of confidence scores (N,)
            captions: List of caption strings (N,)
            
        Returns:
            PIL Image with visualizations
        """
        # Convert numpy array to PIL Image
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        
        # Calculate font size based on image dimensions
        base_font_size = max(24, int(min(image.shape[:2]) * 0.03))
        
        try:
            # Try to load a large font
            font = ImageFont.truetype("arial.ttf", base_font_size)
            font_bold = ImageFont.truetype("arialbd.ttf", base_font_size)
        except IOError:
            # Fallback to default font (will be small)
            font = ImageFont.load_default()
            font_bold = font
        
        # Generate distinct colors for each object
        colors = self._generate_colors(len(masks))
        
        for i, (mask, box, conf, caption) in enumerate(zip(masks, boxes, confidences, captions)):
            color = colors[i]
            
            # Create overlay for mask
            mask_overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
            mask_draw = ImageDraw.Draw(mask_overlay)
            
            # Convert mask to PIL format and apply color
            mask_pil = Image.fromarray((mask * 150).astype(np.uint8), "L")
            color_overlay = Image.new("RGBA", img_pil.size, color + (100,))
            mask_overlay.paste(color_overlay, (0, 0), mask_pil)
            
            # Composite mask overlay
            img_pil = Image.alpha_composite(img_pil.convert("RGBA"), mask_overlay)
            draw = ImageDraw.Draw(img_pil)
            
            # Draw bounding box with thicker lines
            draw.rectangle([tuple(box[:2]), tuple(box[2:])], 
                        outline=color, 
                        width=4)
            
            # Prepare text
            text = f"{caption} {conf:.2f}"
            
            # Estimate text size
            try:
                text_width = font_bold.getlength(text)
                text_height = base_font_size
            except AttributeError:
                # Fallback if font doesn't support getsize
                text_width = len(text) * base_font_size * 0.6
                text_height = base_font_size
            
            # Text position - above bounding box
            text_x = box[0]
            text_y = max(0, box[1] - text_height - 5)
            
            # Draw text background with padding
            padding = 5
            draw.rectangle(
                [
                    (text_x - padding, text_y - padding),
                    (text_x + text_width + padding*2, text_y + text_height + padding)
                ],
                fill=(0, 0, 0, 180)
            )
            
            # Draw text with bold font if available
            try:
                draw.text((text_x, text_y), text, fill="white", font=font_bold)
            except:
                draw.text((text_x, text_y), text, fill="white", font=font)
        
        return img_pil.convert("RGB")
    
    def _generate_colors(self, n: int) -> list:
        colors = []
        for i in range(n):
            hue = i / n
            saturation = 0.7 + 0.3 * (i % 3)
            value = 0.8 + 0.2 * (i % 2)
            r, g, b = hsv_to_rgb([hue, saturation, value])
            colors.append((int(r*255), int(g*255), int(b*255)))
        return colors


if __name__ == "__main__":
    segmenter = SemanticSegmenter(
        grouding_model_config='/home/mspx/icra/GQnav/third_party/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
        grounding_model_checkpoint='/home/mspx/icra/GQnav/model/groundingdino_swint_ogc.pth',
        sam_version='vit_b',
        sam_checkpoint='/home/mspx/icra/GQnav/model/sam_vit_b_01ec64.pth',
        # node_space='table. tv. chair. cabinet. sofa. bed. windows. kitchen. bedroom. living room. mirror. plant. curtain. painting. picture',
        node_space="table. tv. window. chair. refrigerator. sports. bench. piano. couch. rug. desk. doors. staircase. sink. toilet. decoration. pillow.",
        device='cuda'
    )
    
    image = np.array(Image.open('/home/mspx/icra/GQnav/test_image/0b22fa63d0f54a529c525afbf2e8bb25_COMBINED.jpg'))
    mask, xyxy, conf, caption = segmenter.get_segmentation(image)
    # print(mask, xyxy, conf, caption)

    if mask is not None:
        vis_image = segmenter.visualize_segmentation(
            image=image,
            masks=mask,
            boxes=xyxy,
            confidences=conf,
            captions=caption
        )
        vis_image.save("segmentation_results.jpg")
        print("Saved visualization to segmentation_results.jpg")
        
        plt.figure(figsize=(12, 8))
        plt.imshow(vis_image)
        plt.axis('off')
        plt.title("Segmentation Results")
        plt.show()
    else:
        print("No objects detected")