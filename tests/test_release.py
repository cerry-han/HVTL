import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    spec = importlib.util.spec_from_file_location(Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mapping = load('experiments/mapping.py')
metrics = load('evaluation/video_eval_metrics.py')


class ReleaseTests(unittest.TestCase):
    def test_manifest(self):
        entries = json.loads((ROOT / 'docs/source_manifest.json').read_text())
        self.assertGreaterEqual(len(entries), 11)
        self.assertEqual(len(entries), len({entry['path'] for entry in entries}))
        for entry in entries:
            data = (ROOT / entry['path']).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), entry['sha256'])

    def test_python_syntax(self):
        for path in ROOT.rglob('*.py'):
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))

    def test_named_mappings(self):
        self.assertEqual(list(mapping.make_mapping('reverse').values()),
                         ['L12', 'L9', 'L6', 'L3'])
        self.assertEqual(set(mapping.make_mapping('shared').values()), {'L12'})
        original = mapping.make_mapping('original')
        self.assertTrue(all(k == v for k, v in original.items()))

    def test_random_mapping_rng_isolation(self):
        state = random.getstate()
        result = mapping.make_mapping('random')
        self.assertEqual(state, random.getstate())
        self.assertEqual(result, mapping.make_mapping('random'))
        self.assertEqual(set(result), set(result.values()))
        self.assertNotEqual(result, mapping.make_mapping('original'))
        self.assertNotEqual(result, mapping.make_mapping('reverse'))

    def test_remap_preserves_original(self):
        tokens = {key: key for key in ('L3', 'L6', 'L9', 'L12')}
        data = {'text_tokens': tokens,
                'evidence_rank_tokens': {'hard_candidates': tokens,
                                         'natural_negative': tokens}}
        result = mapping.remap_batch(data, mapping.make_mapping('reverse'))
        self.assertEqual(data['text_tokens']['L3'], 'L3')
        self.assertEqual(result['text_tokens']['L3'], 'L12')
        self.assertIs(result[mapping.ORIGINAL_TEXT], tokens)
        with self.assertRaises(ValueError):
            mapping.remap_batch(result, mapping.make_mapping('reverse'))

    def test_video_mean(self):
        rows = [dict(frame_path=str(i), video_id='v', label=1, probability=p)
                for i, p in enumerate([0.2, 0.8])]
        self.assertEqual(metrics.aggregate(rows),
                         [dict(video_id='v', label=1, frames=2, probability=0.5)])

    def test_invalid_predictions(self):
        row = dict(frame_path='frame', video_id='v', label=1, probability=0.5)
        for rows in ([], [row, row], [dict(row, probability=float('nan'))],
                     [dict(row, probability=1.1)],
                     [row, dict(row, frame_path='other', label=0)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                metrics.aggregate(rows)


if __name__ == '__main__':
    unittest.main()
