import os
import glob
import time
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn import functional as F
import pandas as pd
from transformers import AutoTokenizer
import pickle
import glob
from utils.load_files import (
    load_from_yaml_file,
    find_file_path_in_yaml,
    load_box_linelist_file,
)
from toolz.sandbox import unzip
import os.path as op

def move_to_cuda(batch):
    if isinstance(batch, torch.Tensor):
        return batch.cuda(non_blocking=True)
    elif isinstance(batch, list):
        new_batch = [move_to_cuda(t) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(move_to_cuda(t) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: move_to_cuda(t) for n, t in batch.items()}
    else:
        return batch
    return new_batch
def get_dl(ds, args, worker_init_fn=None, collate_fn=None):
    if args.distributed:
        sp = torch.utils.data.distributed.DistributedSampler(
            ds, shuffle=(ds.split == 'train'))
    else:
        if ds.split == 'train':
            sp = torch.utils.data.RandomSampler(ds)
        else:
            sp = torch.utils.data.SequentialSampler(ds)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.size_batch, num_workers=args.n_workers,
        pin_memory=True, sampler=sp, worker_init_fn=worker_init_fn,
        collate_fn=collate_fn)
    return dl

def masking(txt, tokenizer, p_mask=0.15):
    (_B, _X) = txt.shape

    spc_txt = torch.logical_or(
        txt == tokenizer.pad_token_id, txt == tokenizer.mask_token_id
    )

    ans_mtm = torch.ones(txt.shape).long() * -1

    if p_mask <= 0:
        return {"txt": txt, "ans_mtm": ans_mtm}

    for i in range(_B):
        mask_mtm = torch.where(
            condition=torch.logical_and(
                torch.logical_not(spc_txt[i]),
                torch.rand(_X, device=txt.device) < p_mask,
            )
        )[0]

        for p in mask_mtm:
            ans_mtm[i][p], txt[i][p] = txt[i][p], tokenizer.mask_token_id
    return {"txt": txt, "ans_mtm": ans_mtm}


class Fitness_caption(Dataset):
    def __init__(
        self,
        is_training,
        split,
        data_root,
        video_dir,
        yaml_file,
        train_datafile,
        test_datafile,
        max_seq_len,
        input_frame_size=None,
        crop_frame_size=None,
    ):
        self.subset = split
        self.split = split
        self.is_training = is_training
        self.data_root = data_root
        self.crop_frame_size = crop_frame_size
        self.tokzr = AutoTokenizer.from_pretrained("bert-base-uncased")
        (
            self.cls_token_id,
            self.sep_token_id,
            self.pad_token_id,
            self.mask_token_id,
            self.unk_token_id,
        ) = self.tokzr.convert_tokens_to_ids(
            [
                self.tokzr.cls_token,
                self.tokzr.sep_token,
                self.tokzr.pad_token,
                self.tokzr.mask_token,
                self.tokzr.unk_token,
            ]
        )
        self.true_token_id = self.tokzr.convert_tokens_to_ids(["true"])[0]
        self.false_token_id = self.tokzr.convert_tokens_to_ids(["false"])[0]
        if not os.path.isfile(yaml_file):
            yaml_file = os.path.join(yaml_file)
            assert os.path.isfile(yaml_file), f"{yaml_file} does not exists"
        self.yaml_file = yaml_file
        if self.subset == "test":
            self.transforms = transforms.Compose(
                [
                    transforms.Resize(input_frame_size),
                    transforms.CenterCrop(crop_frame_size),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        elif self.subset == "train":
            self.transforms = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.Resize(input_frame_size),
                    transforms.RandomCrop(crop_frame_size),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            raise ValueError("subset should be train or test")

        print(f"subset: {self.subset}, is_training: {self.is_training}")

        self.pil_2_tensor = transforms.ToTensor()

        self.max_seq_len = max_seq_len

        # file paths
        self.cot_dir = "./dataset/comment"

        self.video_dir = video_dir
        excel_file = "./dataset/GYM88.xlsx"
        self.df = pd.read_excel(excel_file)

        videonames = []
        full_paths = []
        for dirpath, dirnames, filenames in os.walk(self.data_root):
            for filename in filenames:
                if os.path.isfile(os.path.join(dirpath, filename)):
                    filename_without_ext = os.path.splitext(filename)[0]
                    videonames.append(filename_without_ext)
                full_path = os.path.join(dirpath, filename)
                full_paths.append(full_path)
        full_paths.sort()
        videonames.sort()

        train_datafile_path = os.path.join(train_datafile)
        test_datafile_path = os.path.join(test_datafile)

        self.datalist = videonames
        if self.subset == "test":
            self.datalist = self._load_annotation(test_datafile_path)
        elif self.subset == "train":
            self.datalist = self._load_annotation(train_datafile_path)

        # 过滤无效样本
        clear_datalist = []
        for video_id in self.datalist:
            row = self.df[self.df["视频名字"] == video_id]
            if row["好坏"].empty:
                print(f"跳过 {video_id}: '好坏' 字段为空")
                continue
            try:
                quality_value = row["好坏"].iloc[0]
                if pd.isna(quality_value):
                    print(f"跳过 {video_id}: '好坏' 值为 NaN")
                    continue
                quality = float(quality_value)
                if quality != quality:  # 检查 NaN (float("nan") != float("nan"))
                    print(f"跳过 {video_id}: '好坏' 值为 NaN")
                    continue
            except (ValueError, TypeError):
                print(f"跳过 {video_id}: '好坏' 值无法转换为浮点数")
                continue
            clear_datalist.append(video_id)

        print(f"原始数据量: {len(self.datalist)}, 有效数据量: {len(clear_datalist)}")
        self.datalist = clear_datalist
        self.image_keys = clear_datalist

    def read_pickle(self, pickle_path):
        with open(pickle_path, "rb") as f:
            pickle_data = pickle.load(f)
        return pickle_data

    def _load_annotation(self, pkl_file):
        data_list = self.read_pickle(pkl_file)
        processed_data_list = []
        for video_id in data_list:
            filename = video_id[1]  # 获取第二个元素 (文件名)
            name_without_extension = filename.split(".")[0]
            processed_data_list.append(name_without_extension)
        return processed_data_list

    def read_image(self, image_all_path):
        image_list = []
        try:
            for image_path in image_all_path:
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
                img = transforms.ToTensor()(img)
                img = self.transforms(img)
                image_list.append(img.unsqueeze(0))
            image_list = torch.cat(image_list, dim=0)
            return image_list
        except Exception as e:
            print(f"Error reading image: {image_path}, {e}")
            return None  # 或者抛出异常

    def get_prompt(self, prompt_text=None):
        if prompt_text is None:
            prompt_text = "write a description about the video."
        toks = self.tokzr.tokenize(prompt_text)
        txt = (
            [self.cls_token_id]
            + self.tokzr.convert_tokens_to_ids(toks)
            + [self.sep_token_id]
        )
        mask = [1 if w != self.pad_token_id else w for w in txt]
        mask = torch.LongTensor(mask)
        txt = torch.LongTensor(txt)
        return txt, mask

    def __getitem__(self, index):
        # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        video_data = self.datalist[index]
        video_id = video_data
        row = self.df[self.df["视频名字"] == video_id]

        num_frame = 6
        data = {"video_id": video_id}
        video_id_full = video_id + ".mp4"
        underscore_index = video_id_full.find("_")
        video_subdir = video_id_full[:underscore_index]
        video_path = os.path.join(self.data_root, video_subdir, video_id_full)

        data["video_name"] = video_path
        aqa_class = int(video_subdir)
        aqa_class = torch.tensor(aqa_class).float()
        quality = float(row["好坏"].iloc[0])  # 已确保有效
        target = torch.tensor(quality).float()

        data["target"] = target
        data["aqa_class"] = aqa_class
        target_filename = f"{video_data}_with_label.txt"
        filepath = os.path.join(self.cot_dir, target_filename)
        videopath = os.path.join(self.video_dir, video_data)
        jpg_files = []
        if os.path.exists(videopath):
            jpg_files.extend(glob.glob(os.path.join(videopath, "*.jpg")))
            jpg_files.sort()

            if len(jpg_files) < 108:
                last_file = (
                    jpg_files[-1] if jpg_files else None
                )  # Get the last file path, handle empty list
                if last_file:
                    padding_size = 108 - len(jpg_files)
                    padding_files = [
                        last_file
                    ] * padding_size  # Create a list of repeated last file paths
                    jpg_files.extend(padding_files)  # Extend the original list
                else:
                    # Handle the case where videopath exists but contains no JPG files.
                    # You could, for instance:
                    print(
                        f"Warning: No JPG files found in {videopath}.  Using empty padding."
                    )
                    jpg_files = [
                        ""
                    ] * 108  #  Pad with empty strings or a default image path

            elif len(jpg_files) > 108:
                jpg_files = jpg_files[:108]  # 截断到108帧
            distance = len(jpg_files) / num_frame
            jpg_files = jpg_files[:: int(distance)]
        data["window_frames"] = self.read_image(jpg_files)

        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    cot = f.read()
                    data["cot"] = cot
            except OSError as e:
                print(f"打开文件时出错: {e}")
                cot = ""
                data["cot"] = ""  # 默认空字符串
        else:
            print(f"未找到文件: {filepath}")
            cot = ""
            data["cot"] = ""  # 默认空字符串
        max_seq_len = 256
        encoded_cot = self.tokzr(
            cot,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=max_seq_len,  # Or a different reasonable max length for CoT
        )

        masked_batch = masking(encoded_cot["input_ids"], self.tokzr, p_mask=0.15)
        txt_input_ids = encoded_cot["input_ids"].squeeze(0)
        txt_mask = encoded_cot["input_ids"].squeeze(0)
        # pad_mask = (encoded_cot["attention_mask"] != self.tokzr.pad_token_id).long()

        return (
            data["window_frames"],
            txt_input_ids,
            txt_mask,
            video_data,
            data["target"],
            data["aqa_class"],
        )

    # return img, txt, mask, vid
    def __len__(self):
        return len(self.datalist)

    def collate_batch(self, inputs):
        img, txt, mask, vid, score, aqa_class = map(list, unzip(inputs))
        all_imgs = torch.stack(img, dim=0)
        all_txts = torch.stack(txt, dim=0)
        all_masks = torch.stack(mask, dim=0)
        all_score = torch.stack(score, dim=0)
        all_aqa_class = torch.stack(aqa_class, dim=0)
        batch = {
            "img": all_imgs,
            "txt": all_txts,
            "mask": all_masks,
            "img_keys": vid,
            "gt_score": all_score,
            "gt_class": all_aqa_class,
        }

        return batch

    def get_caption_file_in_coco_format(self):
        # # for evaluation
        # cap_file_coco_format = find_file_path_in_yaml(
        #     self.cfg.get("caption_coco_format", None), self.root
        # )
        # if cap_file_coco_format:
        #     return cap_file_coco_format
        # test_split = op.basename(self.yaml_file).split(".")[0]
        return op.join("./", "test" + "_caption_coco_format.pth")
