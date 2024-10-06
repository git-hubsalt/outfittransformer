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

args = Args()

args.data_dir = './data/sample'
args.model_path = './checkpoint/cp/240922/epoch_1_AUC_0.000.pth'

# Inference Setting
args.num_workers = 0 
args.inference_batch_size = 1
args.with_cuda = True

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

    images = [load_image(img_path) for img_path in outfit['image']]
    
    inputs = prepare_input(args, input_processor, outfit['category'], images, outfit['text'], outfit['style'])
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        item_embeddings = model.batch_encode(inputs)
        cp_score = model.get_score(item_embeddings)

    print(f"Compatibility Score: {cp_score.item()}")

    return cp_score

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model, input_processor = load_model(args)
    model.to(device)

    outfit = {'category': [], 'image': [], 'text': [], 'style': []}
    style = '모던'
    
    path = os.path.join(args.data_dir, 'sample.txt')
    with open(path, 'r') as file:
        # image, category, text 순
        items = file.readlines()
        for item in items:
            name, category, text = item.rstrip().split(', ')
            outfit['image'].append(name)
            outfit['category'].append(category)
            outfit['text'].append(text)
            outfit['style'].append(style)

    cp_score = inference(outfit, model, args, input_processor)
