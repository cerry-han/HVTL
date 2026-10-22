"""Validated video aggregation, independent of model dependencies."""
import math


def aggregate(rows):
    groups = {}
    seen = set()
    for row in rows:
        path, video = row['frame_path'], row['video_id']
        label, score = int(row['label']), float(row['probability'])
        if not video or not path or label not in (0, 1):
            raise ValueError('Invalid identity or binary label')
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Prediction is not a finite probability')
        if path in seen:
            raise ValueError('Duplicate frame: ' + path)
        seen.add(path)
        group = groups.setdefault(video, {'label': label, 'scores': []})
        if group['label'] != label:
            raise ValueError('Conflicting labels in video: ' + video)
        group['scores'].append(score)
    if not groups:
        raise ValueError('No predictions')
    return [dict(video_id=v, label=g['label'], frames=len(g['scores']),
                 probability=math.fsum(g['scores']) / len(g['scores']))
            for v, g in sorted(groups.items())]
