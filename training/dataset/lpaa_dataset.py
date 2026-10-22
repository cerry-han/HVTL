"""
LPAA dataset with optional offline SBI image-text augmentation.

Place at: training/dataset/lpaa_dataset.py
"""

import hashlib
import json
import os
from copy import deepcopy

import cv2
import numpy as np
import torch
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from transformers import CLIPTokenizerFast


class LPAADataset(DeepfakeAbstractBaseDataset):

    LAYERS = [3, 6, 9, 12]

    REAL_TEXT_TEMPLATES = {
        "L3": [
            "The face region shows clear natural texture with consistent fine details.",
            "The facial appearance contains coherent texture and naturally preserved details.",
            "The face exhibits stable low-level texture without abnormal blurring.",
        ],
        "L6": [
            "The facial components have natural structure and consistent geometry.",
            "The face shows normal proportions and structurally coherent facial features.",
            "The facial contours and component arrangement appear geometrically natural.",
        ],
        "L9": [
            "The face boundary has smooth natural transitions without blending artifacts.",
            "The facial edges remain coherent with no unnatural blending boundary.",
            "The face shows consistent color transitions and naturally formed boundaries.",
        ],
        "L12": [
            "This is an authentic face with natural texture, structure, and boundaries.",
            "The face appears genuine with coherent details and no visible manipulation artifacts.",
            "This facial image shows globally consistent and natural visual characteristics.",
        ],
    }

    def __init__(self, config, mode="train"):
        super().__init__(config, mode)

        clip_name = config.get(
            "clip_model_name", "openai/clip-vit-base-patch16"
        )
        desc_dir = config.get("pfig_desc_dir", "")
        json_path = os.path.join(desc_dir, "descriptions.json")

        print(f"[LPAADataset] 加载 CLIP tokenizer: {clip_name}")
        self.tokenizer = CLIPTokenizerFast.from_pretrained(clip_name)
        self.max_length = 77

        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as handle:
                self.descriptions = json.load(handle)
            print(f"[LPAADataset] 加载描述 {len(self.descriptions)} 条")
        else:
            print(f"[LPAADataset] 警告：{json_path} 不存在，将使用空描述")
            self.descriptions = {}

        self.base_length = len(self.data_dict["image"])
        self.sbi_samples = []
        if mode == "train" and config.get("use_sbi_pfig", False):
            self._load_sbi_samples(config)

    def _load_sbi_samples(self, config):
        dataset_dir = config.get(
            "sbi_dataset_dir", "./data/sbi"
        )
        desc_path = config.get(
            "sbi_pfig_desc_path",
            os.path.join(dataset_dir, "descriptions.json"),
        )
        include_real = bool(config.get("sbi_include_real", True))
        repeat = max(1, int(config.get("sbi_repeat", 1)))

        if not os.path.isfile(desc_path):
            raise FileNotFoundError(f"SBI description file not found: {desc_path}")

        with open(desc_path, "r", encoding="utf-8") as handle:
            entries = json.load(handle)

        samples = []
        missing = 0
        for fake_key, entry in entries.items():
            fake_path = entry.get("fake_key", fake_key)
            real_path = entry.get("real_key")
            fake_descs = entry.get("descriptions", {})
            mask_path = os.path.join(
                dataset_dir, "mask", os.path.basename(fake_path)
            )

            if not os.path.isfile(fake_path):
                missing += 1
                continue
            samples.append(
                {
                    "image_path": fake_path,
                    "mask_path": mask_path,
                    "label": 1,
                    "descriptions": fake_descs,
                    "source": "sbi_fake",
                }
            )

            if include_real and real_path and os.path.isfile(real_path):
                samples.append(
                    {
                        "image_path": real_path,
                        "mask_path": None,
                        "label": 0,
                        "descriptions": self._real_descriptions(real_path),
                        "source": "sbi_real",
                    }
                )

        self.sbi_samples = samples * repeat
        fake_count = sum(s["label"] == 1 for s in self.sbi_samples)
        real_count = sum(s["label"] == 0 for s in self.sbi_samples)
        print(
            "[LPAADataset] SBI augmentation: "
            f"fake={fake_count}, real={real_count}, repeat={repeat}, "
            f"missing={missing}, total_train={self.base_length + len(self.sbi_samples)}"
        )

    def _real_descriptions(self, image_path):
        digest = hashlib.sha1(image_path.encode("utf-8")).digest()
        descriptions = {}
        for offset, layer in enumerate(self.LAYERS):
            key = f"L{layer}"
            choices = self.REAL_TEXT_TEMPLATES[key]
            descriptions[key] = choices[digest[offset] % len(choices)]
        return descriptions

    def _tokenize(self, text):
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0)

    def _empty_token(self):
        return torch.zeros(self.max_length, dtype=torch.long)

    def _tokenize_layers(self, layer_descs):
        text_tokens = {}
        for layer in self.LAYERS:
            key = f"L{layer}"
            desc = layer_descs.get(key)
            text_tokens[key] = (
                self._tokenize(desc) if desc else self._empty_token()
            )
        return text_tokens

    def _load_sbi_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load SBI image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        size = int(self.config["resolution"])
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)

    def _load_sbi_mask(self, path):
        if not self.config.get("with_mask", False):
            return None
        size = int(self.config["resolution"])
        if not path or not os.path.isfile(path):
            return np.zeros((size, size, 1), dtype=np.float32)
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_LINEAR)
        return np.expand_dims(mask.astype(np.float32) / 255.0, axis=2)

    def _getitem_sbi(self, sample):
        image = self._load_sbi_image(sample["image_path"])
        mask = self._load_sbi_mask(sample["mask_path"])

        if self.mode == "train" and self.config["use_data_augmentation"]:
            image, _, mask = self.data_aug(image, None, mask)
        else:
            image, mask = deepcopy(image), deepcopy(mask)

        image = self.normalize(self.to_tensor(image))
        if mask is not None:
            mask = torch.from_numpy(np.asarray(mask, dtype=np.float32))

        return (
            image,
            int(sample["label"]),
            None,
            mask,
            self._tokenize_layers(sample["descriptions"]),
        )

    def __getitem__(self, index):
        if index >= self.base_length:
            return self._getitem_sbi(self.sbi_samples[index - self.base_length])

        image, label, landmark, mask = super().__getitem__(index)
        image_path = self.image_list[index]

        entry = self.descriptions.get(image_path)
        layer_descs = entry["descriptions"] if entry else {}
        text_tokens = self._tokenize_layers(layer_descs)

        return image, label, landmark, mask, text_tokens

    def __len__(self):
        return self.base_length + len(self.sbi_samples)

    @staticmethod
    def collate_fn(batch):
        return lpaa_collate_fn(batch)


def lpaa_collate_fn(batch):
    images, labels, landmarks, masks, text_tokens_list = zip(*batch)

    images = torch.stack(images, dim=0)
    labels = torch.LongTensor(labels)

    try:
        if not any(
            lm is None or (isinstance(lm, list) and None in lm)
            for lm in landmarks
        ):
            landmarks = torch.stack(
                [
                    torch.tensor(lm) if not isinstance(lm, torch.Tensor) else lm
                    for lm in landmarks
                ],
                dim=0,
            )
        else:
            landmarks = None
    except Exception:
        landmarks = None

    try:
        if not any(
            mask is None or (isinstance(mask, list) and None in mask)
            for mask in masks
        ):
            masks = torch.stack(
                [
                    torch.tensor(mask)
                    if not isinstance(mask, torch.Tensor)
                    else mask
                    for mask in masks
                ],
                dim=0,
            )
        else:
            masks = None
    except Exception:
        masks = None

    text_tokens = {
        f"L{layer}": torch.stack(
            [tokens[f"L{layer}"] for tokens in text_tokens_list], dim=0
        )
        for layer in [3, 6, 9, 12]
    }

    return {
        "image": images,
        "label": labels,
        "landmark": landmarks,
        "mask": masks,
        "text_tokens": text_tokens,
    }
