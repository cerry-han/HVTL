"""Frame/video evaluation using the original training-time test loader."""
import argparse
import csv
import hashlib
import json
import random
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from video_eval_metrics import aggregate


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--train-config', default='configs/empty_defaults.yaml')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--datasets', nargs='+', default=[
        'Celeb-DF-v1', 'Celeb-DF-v2', 'DFDCP', 'DFDC', 'UADFV'])
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error('Invalid batch size or worker count')
    if len(set(args.datasets)) != len(args.datasets):
        parser.error('Duplicate dataset names')
    if any('/' in name or '\\' in name or name in ('.', '..') for name in args.datasets):
        parser.error('Dataset names must not contain path separators')
    import numpy as np
    import torch
    import yaml
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    root = Path.cwd()
    sys.path.insert(0, str(root / 'training'))
    from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
    from detectors import DETECTOR

    class IdentifiedDataset(DeepfakeAbstractBaseDataset):
        def collect_img_and_label_for_one_dataset(self, name):
            images, labels, videos = super().collect_img_and_label_for_one_dataset(name)
            self.video_ids = list(videos)
            return images, labels, videos

        def __getitem__(self, index):
            sample = super().__getitem__(index)
            return sample, str(self.image_list[index]), self.video_ids[index]

        @staticmethod
        def collate_fn(batch):
            samples, paths, videos = zip(*batch)
            return DeepfakeAbstractBaseDataset.collate_fn(samples), paths, videos

    def read_yaml(path):
        with open(path, encoding='utf-8') as handle:
            return yaml.safe_load(handle)

    # Match train.py: detector config first, then training defaults, except labels.
    config = read_yaml(args.config)
    defaults = read_yaml(args.train_config)
    if 'label_dict' in config:
        defaults['label_dict'] = config['label_dict']
    config.update(defaults)
    # Use the same explicit metadata directory as training.
    config.update(ddp=False, cuda=True, local_rank=0, manualSeed=args.seed,
                  use_data_augmentation=False)
    if config.get('video_mode', False):
        raise ValueError('Expected a frame model, not video_mode=True')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; activate dfbench_new and check GPU')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = bool(config.get('cudnn', False))

    output = Path(args.output or ('outputs/supplementary/frame_video/' +
                                 datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    digest = hashlib.sha256()
    with checkpoint.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    metadata = dict(checkpoint=str(checkpoint), sha256=digest.hexdigest(),
                    args=vars(args), config=config, torch_version=torch.__version__,
                    gpu=torch.cuda.get_device_name(0), aggregation='mean fake probability',
                    sampling='Unmodified repository test dataset sampling',
                    status='incomplete')
    def save_metadata():
        (output / 'metadata.json').write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    save_metadata()
    # Construct test sets before the model, as in train.py. Capture IDs before shuffling is lost.
    datasets = {}
    for name in args.datasets:
        one_config = deepcopy(config)
        one_config['test_dataset'] = name
        ds = IdentifiedDataset(config=one_config, mode='test')
        if len(ds.video_ids) != len(ds):
            raise ValueError('Video/frame alignment failure: ' + name)
        if any(not isinstance(p, str) for p in ds.image_list):
            raise ValueError('Expected individual frame paths')
        datasets[name] = ds
    model = DETECTOR[config['model_name']](config)
    state = torch.load(str(checkpoint), map_location='cpu')
    if not isinstance(state, dict):
        raise ValueError('Expected a checkpoint state dictionary')
    for key in ('state_dict', 'model_state_dict', 'model', 'net'):
        if isinstance(state.get(key), dict):
            state = state[key]
            break
    state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    del state
    model.to('cuda:0').eval()
    summary = []
    for name, ds in datasets.items():
        print('Evaluating {}: {} frames, {} videos'.format(
            name, len(ds), len(set(ds.video_ids))), flush=True)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, drop_last=False,
                            collate_fn=ds.collate_fn, pin_memory=True)
        rows = []
        partial = output / (name + '_frames.partial.csv')
        fields = ['frame_path', 'video_id', 'label', 'probability']
        with partial.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            with torch.inference_mode():
                for index, (batch, paths, videos) in enumerate(loader):
                    labels = (batch['label'] != 0).long()
                    batch['label'] = labels
                    batch = {k: v.to('cuda:0', non_blocking=True) if torch.is_tensor(v) else v
                             for k, v in batch.items()}
                    prob = model(batch, inference=True)['prob'].detach().cpu().numpy()
                    if prob.ndim == 2 and prob.shape[1] == 1:
                        prob = prob[:, 0]
                    if prob.shape != (len(paths),):
                        raise ValueError('Expected one fake probability per frame, got ' + str(prob.shape))
                    for path, video, label, score in zip(paths, videos, labels.tolist(), prob):
                        row = dict(frame_path=path, video_id=video, label=label,
                                   probability=float(score))
                        rows.append(row)
                        writer.writerow(row)
                    if index % 50 == 0:
                        handle.flush()
                        print('{}: {}/{} frames'.format(name, len(rows), len(ds)), flush=True)
        if len(rows) != len(ds):
            raise ValueError('Incomplete frame predictions')
        video_rows = aggregate(rows)
        frame_auc = float(roc_auc_score([r['label'] for r in rows], [r['probability'] for r in rows]))
        video_auc = float(roc_auc_score([r['label'] for r in video_rows], [r['probability'] for r in video_rows]))
        partial.rename(output / (name + '_frames.csv'))
        write_csv(output / (name + '_videos.csv'), video_rows)
        counts = [r['frames'] for r in video_rows]
        result = dict(dataset=name, frames=len(rows), videos=len(video_rows),
                      min_frames_per_video=min(counts), max_frames_per_video=max(counts),
                      frame_auc=frame_auc, video_auc=video_auc)
        summary.append(result)
        write_csv(output / 'summary.csv', summary)
        print(json.dumps(result), flush=True)
        if hasattr(ds, 'env'):
            ds.env.close()
    summary.append(dict(dataset='macro_average', frames=sum(r['frames'] for r in summary),
                        videos=sum(r['videos'] for r in summary), min_frames_per_video='',
                        max_frames_per_video='', frame_auc=float(np.mean([r['frame_auc'] for r in summary])),
                        video_auc=float(np.mean([r['video_auc'] for r in summary]))))
    write_csv(output / 'summary.csv', summary)
    metadata['status'] = 'complete'
    save_metadata()
    print('Saved results to ' + str(output.resolve()), flush=True)
    print('Compare frame_auc with the original log before using video_auc in the paper.', flush=True)


if __name__ == '__main__':
    main()
