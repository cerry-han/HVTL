"""Fixed text-to-depth controls, with the original PFSA supervision retained."""
import logging
import random

LAYERS = (3, 6, 9, 12)
ORIGINAL_TEXT = '_mapping_original_text_tokens'


def make_mapping(strategy, seed=1024):
    if strategy == 'original':
        sources = list(LAYERS)
    elif strategy == 'reverse':
        sources = list(reversed(LAYERS))
    elif strategy == 'shared':
        sources = [12] * 4
    elif strategy == 'random':
        # Independent RNG: the permutation must not consume the training RNG.
        rng = random.Random(seed)
        sources = list(LAYERS)
        rng.shuffle(sources)
        # Exclude the two named controls; partial fixed points remain allowed.
        while sources in (list(LAYERS), list(reversed(LAYERS))):
            rng.shuffle(sources)
    else:
        raise ValueError('Unknown strategy: ' + str(strategy))
    return dict(zip(('L%d' % v for v in LAYERS),
                    ('L%d' % v for v in sources)))


def remap_batch(data, mapping):
    if ORIGINAL_TEXT in data:
        raise ValueError('Batch has already been remapped')
    def remap(values):
        missing = set(mapping.values()) - set(values)
        if missing:
            raise KeyError('Missing source text layers: %s' % sorted(missing))
        return {target: values[source] for target, source in mapping.items()}
    result = dict(data)
    result[ORIGINAL_TEXT] = data['text_tokens']
    result['text_tokens'] = remap(data['text_tokens'])
    rank = dict(data['evidence_rank_tokens'])
    for name in ('hard_candidates', 'natural_negative'):
        rank[name] = remap(rank[name])
    result['evidence_rank_tokens'] = rank
    return result


def register_detector():
    from detectors import DETECTOR
    from detectors.lpaa_crossattn_pfsa_detector import LPAACrossAttnPFSADetector

    @DETECTOR.register_module(module_name='hvtl_mapping_ablation')
    class MappingDetector(LPAACrossAttnPFSADetector):
        def __init__(self, config):
            super().__init__(config)
            if tuple(self.adapter_layers) != LAYERS:
                raise ValueError('This control requires B/16 layers 3,6,9,12')
            self.semantic_mapping = make_mapping(
                config['mapping_strategy'], config['mapping_seed'])
            logging.getLogger(__name__).info(
                '[Mapping] %s; PFSA reads ORIGINAL text; weights stay at visual depths',
                self.semantic_mapping)
            print('[Mapping]', self.semantic_mapping, 'PFSA=original', flush=True)

        def get_losses(self, data_dict, pred_dict):
            if 'text_tokens' not in data_dict:
                if self.training:
                    raise KeyError('Training batch is missing text_tokens')
                return super().get_losses(data_dict, pred_dict)
            mapped = remap_batch(data_dict, self.semantic_mapping)
            return super().get_losses(mapped, pred_dict)

        def _compute_pfsa_mono_loss(self, data_dict, pred_dict):
            original = dict(data_dict)
            original['text_tokens'] = data_dict.get(
                ORIGINAL_TEXT, data_dict.get('text_tokens', {}))
            return super()._compute_pfsa_mono_loss(original, pred_dict)

    return MappingDetector
