import torch
from torch.utils.data import DataLoader
from src.datasets.kfashion import DatasetArguments, KFashionDataset
from src.datasets.processor import FashionInputProcessor
from src.models.embedder import KORCLIPEmbeddingModel
from src.models.recommender import RecommendationModel
from src.models.load import load_model
from src.utils.utils import save_model
from PIL import Image
import os
import numpy as np
from dataclasses import dataclass
from src.utils.utils import *
from model_args import Args
import itertools
import time

args = Args()

args.data_dir = './data/sample'
args.model_path = './checkpoint/cp/240930/epoch_3_AUC_0.000.pth'

# Inference Setting
args.num_workers = 0 
args.inference_batch_size = 1
args.with_cuda = False

def load_image(image_path):
    path = os.path.join(args.data_dir, image_path)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def prepare_input(args, input_processor, categories, images, texts, styles):
    inputs = input_processor(
        category=categories,
        images=images,
        texts=texts,
        styles=styles,
        do_pad=True
    )
    return {k: v.unsqueeze(0) for k, v in inputs.items()}  # Add batch dimension

def inference(outfit, model, args, input_processor):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.with_cuda else "cpu")
    model = model.to(device)

    start_time = time.time()
    
    images = [load_image(img_path) for img_path in outfit['image']]
    
    inputs = prepare_input(args, input_processor, outfit['category'], images, outfit['text'], outfit['style'])
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        item_embeddings = model.batch_encode(inputs)
        cp_score = model.get_score(item_embeddings)

    inference_time = time.time() - start_time
    return cp_score.item(), inference_time

def load_items(file_path):
    tops = []
    bottoms = []
    with open(file_path, 'r') as file:
        for line in file:
            name, category, text = line.strip().split(', ')
            if category == '상의':
                tops.append((name, text))
            elif category == '하의':
                bottoms.append((name, text))
    return tops, bottoms

def main():
    start_time = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model, input_processor = load_model(args)
    model.to(device)

    file_path = os.path.join(args.data_dir, 'sample.txt')
    tops, bottoms = load_items(file_path)

    combinations = list(itertools.product(tops, bottoms))
    results = []

    total_inference_time = 0
    for top, bottom in combinations:
        outfit = {
            'category': ['상의', '하의'],
            'image': [top[0], bottom[0]],
            'text': [top[1], bottom[1]],
            'style': ['로맨틱', '로맨틱'] 
        }
        cp_score, inference_time = inference(outfit, model, args, input_processor)
        results.append((top[0], bottom[0], cp_score, inference_time))
        total_inference_time += inference_time

    # Sort results by cp_score in descending order
    results.sort(key=lambda x: x[2], reverse=True)

    # Print top 10 combinations
    print("Top 10 outfit combinations:")
    for i, (top, bottom, score, inf_time) in enumerate(results[:10], 1):
        print(f"{i}. Top: {top}, Bottom: {bottom}, Score: {score:.4f}, Inference Time: {inf_time:.4f} seconds")

    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print(f"Total inference time: {total_inference_time:.2f} seconds")
    print(f"Average inference time per outfit: {total_inference_time / len(combinations):.4f} seconds")

if __name__ == '__main__':
    main()