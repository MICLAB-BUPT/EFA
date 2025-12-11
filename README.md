## Requirements

- Python 3.8
- PyTorch 2.4.1+cu124
- Transformers 4.46.3
- OpenCV 4.12.0.88
- Pandas 2.0.3
- NumPy 1.24.1
- Pillow 10.2.0
- scikit-learn 1.3.2
- DeepSpeed 0.17.6
- And other dependencies listed in `requirements.txt`

## Dataset Preparation
### Dataset Download

The model is trained and evaluated on the GYM88 dataset, which contains fitness exercise videos with quality annotations.
1. **Download the CoT-AFA dataset**: The dataset can be obtained from https://www.kaggle.com/datasets/dd34dc6f49a960a31e03af896f85be526a72f8c9a684defd715c75d62bedbdc2. 

2. **Dataset Structure**: After downloading, organize the dataset as follows:
   ```
   AQA_data/
   ├── workout_ori/          # Original video files (.mp4)
   │   ├── 00/               # Exercise class 0 videos
   │   ├── 01/               # Exercise class 1 videos
   │   └── ...              # Other exercise classes
   └── frames/               # Extracted video frames (.jpg)
       ├── 00_00/     # Frames for video 1
       ├── 00_01/     # Frames for video 2
       └── ...
   ```
## Configuration

### Modifying Dataset Paths

Edit the configuration file `_args/AFA.yaml` to update the dataset paths:

```yaml
dataset: {
    data_root: /path/to/AQA_data/workout_ori,  # Path to original videos
    video_dir: /path/to/AQA_data/frames,       # Path to extracted frames
    yaml_file: ./swinbert_val.yaml,
    train_datafile: train.pkl,                 # Path to train split pickle
    test_datafile: test.pkl,                   # Path to test split pickle
    max_seq_len: 256,
    input_frame_size: [398, 224],
    crop_frame_size: 224,
  }
```
## Training

To train the model:

```bash
CUDA_VISIBLE_DEVICES=0 python main_caption.py --config _args/args_AFA.json --path_output output
```

For multi-GPU training:
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=5567 main_caption.py --config _args/args_AFA.json --path_output output
```

## Evaluation

The model evaluates on COCO captioning metrics (BLEU, METEOR, ROUGE-L, CIDEr) and regression/classification metrics for quality assessment.

Evaluation results are saved in the `output/` directory with detailed metrics.
