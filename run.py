"""Portable entry point; help and dependency inspection do not initialize CUDA."""
import argparse
from datetime import datetime
import importlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
PATH_KEYS = ('dataset_json_folder', 'lmdb_dir', 'rgb_dir', 'pfig_desc_dir',
             'hybrid_sbi_dataset_dir', 'hybrid_sbi_desc_path', 'log_dir')
DEPENDENCIES = ('yaml', 'torch', 'torchvision', 'transformers', 'numpy', 'scipy',
                'sklearn', 'cv2', 'PIL', 'lmdb', 'albumentations', 'tensorboard', 'tqdm')


def resolve_paths(config, overrides, base):
    unknown = set(overrides) - set(PATH_KEYS)
    if unknown:
        raise ValueError('Unknown path keys: ' + ', '.join(sorted(unknown)))
    config = dict(config)
    config.update(overrides)
    for key in PATH_KEYS:
        if key not in config:
            raise ValueError('Missing path setting: ' + key)
        path = Path(config[key]).expanduser()
        config[key] = str((base / path).resolve() if not path.is_absolute() else path.resolve())
    return config


def check_data(config, training=True):
    required = [Path(config['dataset_json_folder']) / (name + '.json')
                for name in config['test_dataset']]
    if config.get('lmdb'):
        required += [Path(config['lmdb_dir']) / (name + '_lmdb') / 'data.mdb'
                     for name in config['test_dataset']]
    if training:
        required += [Path(config['dataset_json_folder']) / 'FaceForensics++.json',
                     Path(config['pfig_desc_dir']) / 'descriptions.json',
                     Path(config['hybrid_sbi_desc_path']),
                     Path(config['hybrid_sbi_dataset_dir']) / 'manifest.jsonl']
        if config.get('lmdb'):
            required.append(Path(config['lmdb_dir']) / 'FaceForensics++_lmdb/data.mdb')
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError('Required data are missing:\n' + '\n'.join(missing))
    if training:
        spec = importlib.util.spec_from_file_location(
            'split_guard', ROOT / 'training/dataset/lpaa_sbi_split_guard.py')
        guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)
        entries = json.loads(Path(config['hybrid_sbi_desc_path']).read_text(encoding='utf-8'))
        guard.validate_sbi_entries(entries, config)
        missing_images = []
        for key, entry in entries.items():
            for field, value in (('fake_key', entry.get('fake_key', key)),
                                 ('real_key', entry.get('real_key'))):
                if not value or not Path(value).is_file():
                    missing_images.append((key, field, value))
        if missing_images:
            raise FileNotFoundError('SBI image paths need relocation; first examples: ' +
                                    repr(missing_images[:5]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['doctor', 'check', 'smoke', 'train', 'evaluate'])
    parser.add_argument('--config', default=str(ROOT / 'configs/main.yaml'))
    parser.add_argument('--paths', help='JSON path overrides; relative paths resolve beside this file')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--checkpoint', help='Trusted local checkpoint for evaluation')
    parser.add_argument('--batch-size', type=int, default=16, help='Evaluation batch size only')
    parser.add_argument('--smoke-batch-size', type=int, default=2)
    args = parser.parse_args()
    if args.checkpoint:
        args.checkpoint = str(Path(args.checkpoint).resolve())
    if args.batch_size < 1 or args.smoke_batch_size < 1:
        parser.error('Batch sizes must be positive')
    if args.command == 'doctor':
        missing = [name for name in DEPENDENCIES if importlib.util.find_spec(name) is None]
        print(json.dumps({'python': sys.version, 'missing_modules': missing}, indent=2))
        if missing:
            raise SystemExit(1)
        # Actual imports expose binary/API incompatibilities that metadata cannot detect.
        for name in DEPENDENCIES:
            importlib.import_module(name)
        sys.path.insert(0, str(ROOT / 'training'))
        importlib.import_module('detectors')
        importlib.import_module('dataset.lpaa_herl_dataset')
        importlib.import_module('trainer.trainer')
        print('Runtime imports passed. No model or dataset loaded; no GPU test performed.')
        return
    import yaml
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    overrides = json.loads(Path(args.paths).read_text(encoding='utf-8')) if args.paths else {}
    base = Path(args.paths).resolve().parent if args.paths else ROOT
    config = resolve_paths(config, overrides, base)
    if not config.get('lmdb'):
        raise ValueError('This release entry point currently requires the verified LMDB pipeline')
    if config['dataset_type'] != 'lpaa_herl' or config['model_name'] != 'lpaa_crossattn_pfsa':
        raise ValueError('This entry point currently supports the corrected full main model only')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / 'training'))
    check_data(config, training=args.command != 'evaluate')
    if args.command == 'check':
        print('Required paths and SBI source split verified. No training started.')
        return
    if args.command == 'evaluate' and (not args.checkpoint or not Path(args.checkpoint).is_file()):
        raise ValueError('Provide an existing trusted --checkpoint')
    folder = ROOT / 'outputs' / 'launches' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True, exist_ok=False)
    resolved = folder / 'config.yaml'
    resolved.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    if args.command in ('train', 'smoke'):
        sys.argv = ['train.py', '--detector_path', str(resolved), '--no-save_feat']
        if args.command == 'train':
            runpy.run_path(str(ROOT / 'training/train.py'), run_name='__main__')
        else:
            import torch
            namespace = runpy.run_path(str(ROOT / 'training/train.py'), run_name='release_smoke')
            smoke = dict(config, train_batchSize=args.smoke_batch_size, workers=0,
                         ddp=False, local_rank=0)
            namespace['init_seed'](smoke)
            loader = namespace['prepare_training_data'](smoke)
            model = namespace['DETECTOR'][smoke['model_name']](smoke).cuda().train()
            if hasattr(model, 'on_epoch_start'):
                model.on_epoch_start(0)
            model.epoch = 0
            def to_device(value):
                if torch.is_tensor(value):
                    return value.cuda()
                if isinstance(value, dict):
                    return {key: to_device(item) for key, item in value.items()}
                if isinstance(value, list):
                    return [to_device(item) for item in value]
                if isinstance(value, tuple):
                    return tuple(to_device(item) for item in value)
                return value
            batch = to_device(next(iter(loader)))
            losses = model.get_losses(batch, model(batch))
            if not torch.isfinite(losses['overall']).all():
                raise RuntimeError('Non-finite smoke loss')
            losses['overall'].backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(grad).all() for grad in grads):
                raise RuntimeError('Missing or non-finite smoke gradients')
            print('One training batch passed forward/backward; no optimizer step or checkpoint saved.')
    else:
        sys.path.insert(0, str(ROOT / 'evaluation'))
        sys.argv = ['evaluate.py', '--config', str(resolved), '--checkpoint', args.checkpoint,
                    '--batch-size', str(args.batch_size)]
        runpy.run_path(str(ROOT / 'evaluation/evaluate.py'), run_name='__main__')


if __name__ == '__main__':
    main()
