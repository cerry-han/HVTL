"""Validate offline SBI samples against the active FF++ real-video splits."""
import json
from pathlib import Path


def norm(value):
    if not isinstance(value, str) or not value:
        raise ValueError('Missing or invalid source path')
    return value.replace('\\', '/').strip('/')


def real_video_splits(metadata_path, compression='c23'):
    data = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
    real = data['FaceForensics++']['FF-real']
    splits = {}
    for split in ('train', 'val', 'test'):
        videos = set()
        for record in real[split][compression].values():
            for frame in record['frames']:
                videos.add(norm(frame).rsplit('/', 1)[0])
        if not videos:
            raise ValueError('Empty real-video split: ' + split)
        splits[split] = videos
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if splits[a] & splits[b]:
            raise ValueError('Overlapping real-video paths: %s/%s' % (a, b))
    return splits


def membership(source, splits):
    video = norm(source).rsplit('/', 1)[0]
    return '+'.join(s for s in ('train', 'val', 'test') if video in splits[s]) or 'unmatched'


def load_manifest(path):
    entries = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = norm(record['fake_path'])
        if key in entries:
            raise ValueError('Duplicate fake path in manifest: ' + key)
        entries[key] = record
    return entries


def verify_entry(key, entry, records):
    fake = norm(entry.get('fake_key', key))
    if fake != norm(key):
        raise ValueError('Description key differs from fake_key: ' + str(key))
    record = records[fake]
    if (norm(entry['source_key']) != norm(record['source_key'])
            or norm(entry['real_key']) != norm(record['real_path'])
            or int(record['source_label']) != 0):
        raise ValueError('Description/manifest provenance mismatch: ' + str(key))
    return record


def validate_sbi_entries(entries, config):
    root = Path(__file__).resolve().parents[2]
    folder = Path(config['dataset_json_folder'])
    if not folder.is_absolute():
        folder = root / folder
    splits = real_video_splits(folder / 'FaceForensics++.json', config.get('compression', 'c23'))
    dataset = Path(config['hybrid_sbi_dataset_dir'])
    records = load_manifest(dataset / 'manifest.jsonl')
    if not entries:
        raise ValueError('Empty SBI description database')
    for key, entry in entries.items():
        record = verify_entry(key, entry, records)
        split = membership(record['source_key'], splits)
        if split != 'train':
            raise ValueError('SBI split guard rejected %s: source belongs to %s' % (key, split))
    print('[SBI split guard] Verified %d train-only pairs against active metadata' % len(entries))
