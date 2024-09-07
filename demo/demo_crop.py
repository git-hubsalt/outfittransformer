import argparse
import os
from gdino import GroundingDINOAPIWrapper, visualize
from PIL import Image
import numpy as np

def crop_and_save(image_pil, box, class_name, output_path):
    # Convert box coordinates to integers
    box = [int(b) for b in box]
    # Crop the image
    cropped_image = image_pil.crop(box)
    # Save the cropped image
    cropped_image.save(output_path)

def get_args():
    parser = argparse.ArgumentParser(description="Interactive Inference")
    parser.add_argument(
        "--token",
        type=str,
        help="The token for T-Rex2 API. We are now opening free API access to T-Rex2",
    )
    parser.add_argument(
        "--box_threshold", type=float, default=0.3, help="The threshold for box score"
    )
    return parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_dir, '..', 'asset', 'k-fashion')
path = os.path.abspath(path)
os.chdir(path)
image_files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

if __name__ == "__main__":
    args = get_args()
    gdino = GroundingDINOAPIWrapper(args.token)
    for image in image_files:
        image_path = os.path.join(path, image)
        image_name_without_ext = image.split('.')[0]
        
        prompts = dict(image=image_path, prompt='t-shirt.pants')
        results = gdino.inference(prompts)
        
        # Open the original image
        image_pil = Image.open(prompts['image'])
        
        # Crop and save each detected object
        for i, (box, class_name, _) in enumerate(zip(results['boxes'], results['categorys'], results['scores'])):
            output_dir = os.path.join(script_dir, '..', 'asset', 'crop')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'{image_name_without_ext}_{i}.jpg')
            crop_and_save(image_pil, box, class_name, output_path)
        
        # Visualize the results on the original image (optional)
        # image_pil_with_boxes = visualize(image_pil, results)
        # image_pil_with_boxes.save('asset/demo_output_with_boxes.jpg')