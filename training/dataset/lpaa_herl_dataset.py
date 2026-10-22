"""LPAA hybrid dataset with hierarchical evidence ranking prompts."""

import hashlib

import torch

from dataset.lpaa_hybrid_text_dataset import LPAAHybridTextDataset
from dataset.lpaa_dataset import lpaa_collate_fn


class LPAAHERLDataset(LPAAHybridTextDataset):
    """Build positive, hard-evidence, and natural-evidence text tuples."""

    REGIONS = ("eyes", "nose", "mouth", "face")
    HARD_CANDIDATE_COUNT = 3

    def _descriptions_for_index(self, index):
        source, value = self.schedule[index]
        if source == "vlffd_real":
            return self._real_descriptions(self.image_list[value])
        if source == "vlffd_fake":
            path = self.image_list[value]
            entry = self.descriptions.get(path)
            return entry.get("descriptions", {}) if entry else {}
        return value.get("descriptions", {})

    @staticmethod
    def _extract_region(text):
        lowered = text.lower()
        if "mouth" in lowered or "lip" in lowered:
            return "mouth"
        if "eyes" in lowered or "eye" in lowered:
            return "eyes"
        if "nose" in lowered:
            return "nose"
        return "face"

    @staticmethod
    def _region_label(region):
        return {
            "eyes": "eye region",
            "nose": "nose region",
            "mouth": "mouth region",
            "face": "face region",
        }[region]

    def _ordered_alternative_regions(self, current, sample_key, layer):
        regions = [region for region in self.REGIONS if region != current]
        digest = hashlib.sha1(
            f"{sample_key}|{layer}|herl-candidates".encode("utf-8")
        ).digest()
        shift = digest[0] % len(regions)
        return regions[shift:] + regions[:shift]

    def _region_evidence_text(self, layer, positive, region_name):
        region = self._region_label(region_name)
        lowered = positive.lower()
        if layer == 3:
            if "blur" in lowered:
                return (
                    f"The {region} shows weak blur and reduced local "
                    "fine-grained detail."
                )
            return (
                f"The {region} contains subtle low-level texture "
                "inconsistency."
            )
        if layer == 6:
            return (
                f"The {region} contains mild structural inconsistency "
                "relative to natural facial geometry."
            )
        if layer == 9:
            if "color" in lowered or "illumination" in lowered:
                return (
                    f"The {region} shows a weak color or illumination "
                    "transition inconsistency."
                )
            return (
                f"The {region} boundary contains a subtle blending or "
                "edge-transition artifact."
            )
        return (
            f"The {region} contains weak but forgery-related visual "
            "inconsistency."
        )

    @staticmethod
    def _generic_evidence_text(layer):
        if layer == 3:
            return (
                "A facial region contains subtle texture degradation or "
                "loss of fine detail."
            )
        if layer == 6:
            return (
                "A facial component shows mild geometric or structural "
                "inconsistency."
            )
        if layer == 9:
            return (
                "A local facial transition contains weak blending, color, "
                "or boundary inconsistency."
            )
        return (
            "The face contains multiple weak but forgery-related visual "
            "inconsistencies."
        )

    @staticmethod
    def _related_evidence_text(layer):
        if layer == 3:
            return (
                "Local facial texture appears slightly noisy, blurry, or "
                "less detailed than expected."
            )
        if layer == 6:
            return (
                "The facial contour shows a small deformation or local "
                "misalignment."
            )
        if layer == 9:
            return (
                "The skin transition shows a faint double edge or "
                "illumination mismatch."
            )
        return (
            "Several facial areas provide weak but coherent manipulation "
            "evidence."
        )

    def _hard_evidence_texts(self, layer, positive, current, sample_key):
        alternatives = self._ordered_alternative_regions(
            current, sample_key, layer
        )
        return [
            self._region_evidence_text(layer, positive, alternatives[0]),
            self._generic_evidence_text(layer),
            self._related_evidence_text(layer),
        ]

    def _natural_evidence_text(self, layer, current):
        region = self._region_label(current)
        if layer == 3:
            return (
                f"The {region} has natural texture with consistently "
                "preserved fine details."
            )
        if layer == 6:
            return (
                f"The {region} has coherent structure and geometrically "
                "natural contours."
            )
        if layer == 9:
            return (
                f"The {region} has smooth natural boundaries with "
                "consistent color and illumination transitions."
            )
        return (
            f"The {region} and the overall face are coherent, natural, "
            "and free of manipulation artifacts."
        )

    def _evidence_rank_tokens(self, index, label, descriptions):
        candidates_by_layer = {}
        negative_by_layer = {}
        sample_key = str(self.schedule[index][1])

        for layer in self.LAYERS:
            key = f"L{layer}"
            positive = descriptions.get(key)
            is_valid_fake = (
                int(label) == 1
                and isinstance(positive, str)
                and bool(positive.strip())
            )
            if not is_valid_fake:
                candidates_by_layer[key] = torch.stack(
                    [self._empty_token() for _ in range(self.HARD_CANDIDATE_COUNT)]
                )
                negative_by_layer[key] = self._empty_token()
                continue

            current = self._extract_region(positive)
            hard_texts = self._hard_evidence_texts(
                layer, positive, current, sample_key
            )
            candidates_by_layer[key] = torch.stack(
                [self._tokenize(text) for text in hard_texts]
            )
            negative_by_layer[key] = self._tokenize(
                self._natural_evidence_text(layer, current)
            )

        return candidates_by_layer, negative_by_layer

    def __getitem__(self, index):
        image, label, landmark, mask, text_tokens = super().__getitem__(index)
        descriptions = self._descriptions_for_index(index)
        hard_candidates, natural_negative = self._evidence_rank_tokens(
            index, label, descriptions
        )
        return (
            image,
            label,
            landmark,
            mask,
            text_tokens,
            hard_candidates,
            natural_negative,
        )

    @staticmethod
    def collate_fn(batch):
        data_dict = lpaa_collate_fn([sample[:5] for sample in batch])
        candidate_list = [sample[5] for sample in batch]
        negative_list = [sample[6] for sample in batch]

        data_dict["evidence_rank_tokens"] = {
            "hard_candidates": {
                f"L{layer}": torch.stack(
                    [tokens[f"L{layer}"] for tokens in candidate_list], dim=0
                )
                for layer in (3, 6, 9, 12)
            },
            "natural_negative": {
                f"L{layer}": torch.stack(
                    [tokens[f"L{layer}"] for tokens in negative_list], dim=0
                )
                for layer in (3, 6, 9, 12)
            },
        }
        return data_dict
