"""Download DailyDialog and prepare small, short English question/reply splits.

Source: https://huggingface.co/datasets/ConvLab/dailydialog
Paper: https://aclanthology.org/I17-1099/
Mirror license: CC BY-NC-SA 4.0 (noncommercial educational use).
Uses the official dialogue splits, then extracts adjacent utterance pairs.
Default: 5000 train / 500 validation / 500 test pairs; no new dependencies.
"""
import argparse
from io import BytesIO
import json
from pathlib import Path
import random
from urllib.request import urlopen
from zipfile import ZipFile

URL = 'https://huggingface.co/datasets/ConvLab/dailydialog/resolve/745c179/data.zip'
OUT = Path(__file__).resolve().parents[1] / 'data' / 'dailydialog'


def prepare(dialogues, limits):
    splits = {name: [] for name in limits}
    for dialogue in dialogues:
        split = dialogue['data_split']
        split = 'validation' if split in ('val', 'valid', 'validation') else split
        if split not in splits:
            continue
        turns = dialogue['turns']
        for first, second in zip(turns, turns[1:]):
            question = ' '.join(first['utterance'].split())
            answer = ' '.join(second['utterance'].split())
            # Keep complete short examples rather than cutting off replies.
            if 1 <= len(question) <= 96 and 1 <= len(answer) <= 150:
                splits[split].append(dict(question=question, answer=answer,
                                         dialogue_id=dialogue['dialogue_id']))
    for name, rows in splits.items():
        random.Random(42).shuffle(rows)
        splits[name] = rows[:limits[name]]
        if not splits[name]:
            raise ValueError(f'No pairs found for split {name}')
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-pairs', type=int, default=5000)
    parser.add_argument('--eval-pairs', type=int, default=500)
    args = parser.parse_args()
    if min(args.train_pairs, args.eval_pairs) < 1:
        parser.error('pair counts must be positive')
    with urlopen(URL, timeout=60) as response:
        archive = response.read()
    with ZipFile(BytesIO(archive)) as zipped:
        dialogues = json.loads(zipped.read('data/dialogues.json'))
    splits = prepare(dialogues, dict(train=args.train_pairs,
                     validation=args.eval_pairs, test=args.eval_pairs))
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = OUT / f'{name}.json'
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'{name}: {len(rows)} pairs -> {path}')
    (OUT / 'SOURCE.txt').write_text(
        f'DailyDialog\nPaper: https://aclanthology.org/I17-1099/\nMirror: {URL}\n'
        'License (mirror): CC BY-NC-SA 4.0\nOfficial splits retained. Seed=42.\n',
        encoding='utf-8')


if __name__ == '__main__':
    main()
