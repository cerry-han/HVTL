"""Balanced LPAA training dataset for VLFFD/PFIG and offline SBI text.

The epoch is split into four equally sized groups:
  1. original FF++ real images with synthetic real descriptions;
  2. original FF++ fake images with VLFFD/PFIG descriptions;
  3. offline SBI real images with synthetic real descriptions;
  4. offline SBI fake images with SBI-PFIG descriptions.

This gives an exact 50:50 source ratio and an exact 50:50 class ratio while
keeping the number of samples per epoch equal to the original LPAA dataset.
"""

import hashlib
import json
import os
import random
from copy import deepcopy

import cv2
import numpy as np
import torch

from dataset.lpaa_dataset import LPAADataset, lpaa_collate_fn


class LPAAHybridTextDataset(LPAADataset):
    LAYERS = (3, 6, 9, 12)

    REAL_TEXT_TEMPLATES = {
        "L3": (
            "The face region shows clear natural texture with consistent fine details.",
            "The facial appearance contains coherent texture and naturally preserved details.",
            "The face exhibits stable low-level texture without abnormal blurring.",
        ),
        "L6": (
            "The facial components have natural structure and consistent geometry.",
            "The face shows normal proportions and structurally coherent facial features.",
            "The facial contours and component arrangement appear geometrically natural.",
        ),
        "L9": (
            "The face boundary has smooth natural transitions without blending artifacts.",
            "The facial edges remain coherent with no unnatural blending boundary.",
            "The face shows consistent color transitions and naturally formed boundaries.",
        ),
        "L12": (
            "This is an authentic face with natural texture, structure, and boundaries.",
            "The face appears genuine with coherent details and no visible manipulation artifacts.",
            "This facial image shows globally consistent and natural visual characteristics.",
        ),
    }

    def __init__(self, config, mode="train"):
        parent_config = dict(config)
        # Prevent the previous append-style SBI loader from changing this dataset.
        parent_config["use_sbi_pfig"] = False
        super().__init__(parent_config, mode)
        self.config = parent_config

        if mode != "train":
            raise ValueError("LPAAHybridTextDataset is only intended for training")

        self.base_length = len(self.image_list)
        labels = list(getattr(self, "label_list", []))
        if len(labels) != self.base_length:
            labels = list(self.data_dict.get("label", []))
        if len(labels) != self.base_length:
            raise RuntimeError(
                "Cannot align labels with the original LPAA image list: "
                f"images={self.base_length}, labels={len(labels)}"
            )

        base_real = []
        base_fake = []
        missing_base_text = 0
        for index, (path, label) in enumerate(zip(self.image_list, labels)):
            if int(label) == 0:
                base_real.append(index)
                continue
            entry = self.descriptions.get(path)
            descriptions = entry.get("descriptions", {}) if entry else {}
            if self._has_any_text(descriptions):
                base_fake.append(index)
            else:
                missing_base_text += 1

        sbi_real, sbi_fake, missing_sbi = self._load_offline_sbi(config)
        pools = {
            "vlffd_real": base_real,
            "vlffd_fake": base_fake,
            "sbi_real": sbi_real,
            "sbi_fake": sbi_fake,
        }
        empty = [name for name, values in pools.items() if not values]
        if empty:
            raise RuntimeError(f"Hybrid dataset has empty pools: {empty}")

        requested_size = int(config.get("hybrid_epoch_size", 0) or 0)
        if requested_size <= 0:
            requested_size = self.base_length
        group_size = requested_size // 4
        if group_size < 1:
            raise ValueError(f"hybrid_epoch_size is too small: {requested_size}")

        seed = int(config.get("hybrid_seed", config.get("manualSeed", 1024)))
        rng = random.Random(seed)
        schedule = []
        for source, values in pools.items():
            for value in self._balanced_take(values, group_size, rng):
                schedule.append((source, value))
        rng.shuffle(schedule)
        self.schedule = schedule

        print(
            "[LPAAHybridTextDataset] exact mix: "
            f"VLFFD/PFIG={2 * group_size}, SBI={2 * group_size}, "
            f"real={2 * group_size}, fake={2 * group_size}, "
            f"epoch_size={len(self.schedule)}"
        )
        print(
            "[LPAAHybridTextDataset] available pools: "
            f"vlffd_real={len(base_real)}, vlffd_fake={len(base_fake)}, "
            f"sbi_real={len(sbi_real)}, sbi_fake={len(sbi_fake)}, "
            f"missing_base_text={missing_base_text}, missing_sbi={missing_sbi}"
        )

    @staticmethod
    def _has_any_text(descriptions):
        return any(
            isinstance(descriptions.get(f"L{layer}"), str)
            and bool(descriptions[f"L{layer}"].strip())
            for layer in (3, 6, 9, 12)
        )

    @staticmethod
    def _balanced_take(values, count, rng):
        """Select count values reproducibly, cycling only when necessary."""
        values = list(values)
        selected = []
        while len(selected) < count:
            current = values[:]
            rng.shuffle(current)
            selected.extend(current[: count - len(selected)])
        return selected

    def _load_offline_sbi(self, config):
        dataset_dir = config.get(
            "hybrid_sbi_dataset_dir",
            "./data/sbi",
        )
        description_path = config.get(
            "hybrid_sbi_desc_path",
            os.path.join(dataset_dir, "descriptions.json"),
        )
        if not os.path.isfile(description_path):
            raise FileNotFoundError(
                f"SBI description file not found: {description_path}"
            )

        with open(description_path, "r", encoding="utf-8") as handle:
            entries = json.load(handle)

        from dataset.lpaa_sbi_split_guard import validate_sbi_entries
        validate_sbi_entries(entries, config)

        real_samples = []
        fake_samples = []
        missing = 0
        for fake_key, entry in entries.items():
            fake_path = entry.get("fake_key", fake_key)
            real_path = entry.get("real_key")
            if not real_path:
                real_path = os.path.join(
                    dataset_dir, "real", os.path.basename(fake_path)
                )
            mask_path = os.path.join(
                dataset_dir, "mask", os.path.basename(fake_path)
            )
            descriptions = entry.get("descriptions", {})
            if (
                not os.path.isfile(fake_path)
                or not os.path.isfile(real_path)
                or not self._has_any_text(descriptions)
            ):
                missing += 1
                continue

            fake_samples.append(
                {
                    "image_path": fake_path,
                    "mask_path": mask_path,
                    "label": 1,
                    "descriptions": descriptions,
                }
            )
            real_samples.append(
                {
                    "image_path": real_path,
                    "mask_path": None,
                    "label": 0,
                    "descriptions": self._real_descriptions(real_path),
                }
            )
        return real_samples, fake_samples, missing

    def _real_descriptions(self, image_path):
        digest = hashlib.sha1(image_path.encode("utf-8")).digest()
        result = {}
        for offset, layer in enumerate(self.LAYERS):
            key = f"L{layer}"
            choices = self.REAL_TEXT_TEMPLATES[key]
            result[key] = choices[digest[offset] % len(choices)]
        return result

    def _tokenize_layers_local(self, descriptions):
        tokens = {}
        for layer in self.LAYERS:
            key = f"L{layer}"
            text = descriptions.get(key)
            tokens[key] = self._tokenize(text) if text else self._empty_token()
        return tokens

    def _load_disk_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        size = int(self.config["resolution"])
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)

    def _load_disk_mask(self, path):
        if not self.config.get("with_mask", False):
            return None
        size = int(self.config["resolution"])
        if not path or not os.path.isfile(path):
            return np.zeros((size, size, 1), dtype=np.float32)
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return np.zeros((size, size, 1), dtype=np.float32)
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_LINEAR)
        return np.expand_dims(mask.astype(np.float32) / 255.0, axis=2)

    def _getitem_sbi(self, sample):
        image = self._load_disk_image(sample["image_path"])
        mask = self._load_disk_mask(sample["mask_path"])

        if self.config.get("use_data_augmentation", False):
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
            self._tokenize_layers_local(sample["descriptions"]),
        )

    def _getitem_vlffd(self, base_index, use_real_text):
        sample = LPAADataset.__getitem__(self, base_index)
        if not use_real_text:
            return sample
        image, label, landmark, mask, _ = sample
        descriptions = self._real_descriptions(self.image_list[base_index])
        return (
            image,
            label,
            landmark,
            mask,
            self._tokenize_layers_local(descriptions),
        )

    def __getitem__(self, index):
        source, value = self.schedule[index]
        if source == "vlffd_real":
            return self._getitem_vlffd(value, use_real_text=True)
        if source == "vlffd_fake":
            return self._getitem_vlffd(value, use_real_text=False)
        return self._getitem_sbi(value)

    def __len__(self):
        return len(self.schedule)

    @staticmethod
    def collate_fn(batch):
        return lpaa_collate_fn(batch)

