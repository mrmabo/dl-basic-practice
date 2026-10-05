"""Download ~1.1 MB public English Shakespeare text, no extra dependencies.

Source: Karpathy char-rnn's Tiny Shakespeare corpus. Shakespeare's original
works are public domain; see the source repository for corpus provenance.
"""
from pathlib import Path
from urllib.request import urlopen

URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
OUT = Path(__file__).resolve().parents[1] / 'data' / 'tiny_shakespeare' / 'input.txt'


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(URL, timeout=60) as response:
        data = response.read()
    text = data.decode('utf-8')
    if len(text) < 10000 or 'ROMEO' not in text:
        raise ValueError('Downloaded content does not look like Tiny Shakespeare')
    OUT.write_text(text, encoding='utf-8')
    print(f'Saved {len(data):,} bytes to {OUT}')


if __name__ == '__main__':
    main()
