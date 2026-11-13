"""Restore losslessly compressed dataset CSVs and verify original file hashes."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='Check every dataset without writing restored CSV files.')
    args = parser.parse_args()
    records = json.loads((ROOT / 'Datasets/DATASET_MANIFEST.json').read_text(encoding='utf-8'))
    for item in records:
        stored = ROOT / item['repository_path']
        if digest(stored) != item['repository_sha256']:
            raise ValueError(f'Archive/file checksum mismatch: {stored}')
        if item['action'] == 'lossless_gzip':
            target = ROOT / item['source_path']
            if args.verify_only:
                result = hashlib.sha256()
                size = 0
                with gzip.open(stored, 'rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        result.update(block)
                        size += len(block)
                if result.hexdigest() != item['source_sha256'] or size != item['source_bytes']:
                    raise ValueError(f'Original data checksum mismatch: {stored}')
            elif target.exists():
                if digest(target) != item['source_sha256']:
                    raise ValueError(f'Existing file differs; refusing to overwrite: {target}')
            else:
                with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.restoring', delete=False) as out:
                    temp = Path(out.name)
                    try:
                        with gzip.open(stored, 'rb') as stream:
                            shutil.copyfileobj(stream, out)
                    except BaseException:
                        temp.unlink(missing_ok=True)
                        raise
                try:
                    if digest(temp) != item['source_sha256'] or temp.stat().st_size != item['source_bytes']:
                        raise ValueError(f'Restored data checksum mismatch: {stored}')
                    temp.replace(target)
                finally:
                    temp.unlink(missing_ok=True)
        print(f'OK {item["source_path"]}')
    print(f'Verified {len(records)} dataset files.')


if __name__ == '__main__':
    main()
