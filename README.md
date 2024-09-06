# outfittransformer

### environment setting
- `pip install -r requirement.txt`

### dataset
- [Google Drive](https://drive.google.com/drive/folders/1MXxsS_fJY9dm01MRbfpLJgM7VbP1eKZq?usp=sharing)에서 download (original: [AIHub 'KFashion 이미지'](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=51) 데이터)
- `outfittransformer/data` 폴더 생성
- `compatibility_train.txt`, `compatibility_valid.txt`, `item_metadata.json` 파일 해당 경로로 이동

### train
- `model_arg.py`에서 주요 hyperparameter 수정 가능
```
python cp_train.py --task cp --train_batch 64 --valid_batch 96 --n_epochs 5 --learning_rate 1e-5 --scheduler_step_size 1000
```
### test

### inference
