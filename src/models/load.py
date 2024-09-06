from src.datasets.processor import FashionInputProcessor, FashionImageProcessor
from transformers import CLIPImageProcessor, CLIPTokenizer, AutoTokenizer, AutoImageProcessor
from src.models.embedder import CLIPEmbeddingModel, OutfitTransformerEmbeddingModel, KORCLIPEmbeddingModel
from src.models.recommender import RecommendationModel
import torch


def load_model(args):
    if args.use_clip_embedding:
        image_processor = CLIPImageProcessor.from_pretrained(args.clip_huggingface)
        text_tokenizer = CLIPTokenizer.from_pretrained(args.clip_huggingface)
    # else:
    #     image_processor = FashionImageProcessor()
    #     text_tokenizer = AutoTokenizer.from_pretrained(args.huggingface)
    else:
        image_processor = AutoImageProcessor.from_pretrained(args.kor_huggingface)
        text_tokenizer = AutoTokenizer.from_pretrained(args.kor_huggingface)

    input_processor = FashionInputProcessor(
        categories=args.categories,
        use_image=args.use_image,
        image_processor=image_processor, 
        use_text=args.use_text,
        text_tokenizer=text_tokenizer, 
        text_max_length=args.text_max_length, 
        text_padding='max_length', 
        text_truncation=True, 
        outfit_max_length=args.outfit_max_length
        )
    
    if args.use_clip_embedding:
        print(' >>>> loading CLIP Embedding model')
        embedding_model = CLIPEmbeddingModel(
            input_processor=input_processor,
            hidden=args.hidden,
            huggingface=args.clip_huggingface,
            normalize=args.normalize,
            linear_probing=True,
            )
    # else:
    #     embedding_model = OutfitTransformerEmbeddingModel(
    #         input_processor=input_processor,
    #         hidden=args.hidden,
    #         huggingface=args.huggingface,
    #         normalize=args.normalize
    #         )
    else:
        print(' >>>> loading Korean CLIP Embedding model for K-Fashion')
        embedding_model = KORCLIPEmbeddingModel(
            input_processor=input_processor,
            hidden=args.hidden,
            huggingface=args.kor_huggingface,
            normalize=args.normalize,
            linear_probing=True,
        )

    recommendation_model = RecommendationModel(
        embedding_model=embedding_model,
        ffn_hidden=args.ffn_hidden,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        )
    
    if args.load_model:
        checkpoint = torch.load(args.model_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        recommendation_model.load_state_dict(state_dict)
        print(f'[COMPLETE] Load from {args.model_path}')
    
    return recommendation_model, input_processor