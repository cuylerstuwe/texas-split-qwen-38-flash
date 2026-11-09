#!/usr/bin/env python3
"""Check a downloaded Qwen3.8-Flash-Next-Sushi-4bpw pack against host/model-manifest.json (names and
sizes at the pinned revision) and build the model directory texas-split serves: a directory of
symlinks to the download, except config.json, which is host/model-config-native-rope.json: the
pack's config with the model's native rope (rope_type "default", 262,144 positions) instead of YaRN
x4. Sushi gets the same override on its command line; the 5090 stage server reads config.json itself
and pins its exact bytes.

  make_model_view.py DOWNLOAD_DIR VIEW_DIR        (VIEW_DIR's basename must be the model name)
  make_model_view.py --check DOWNLOAD_DIR         verify only
"""
import hashlib, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL_CONFIG_SHA = '08a0cdca437ea94e6e16281d0c884bca670a3bb923cb39293fb0b048d3ca7576'   # the pack's config.json
NATIVE_CONFIG_SHA = '2446916c7a09b7a66bfb3eba2a19d62ded0af083b66e7c20b891bd82bfdde96a'     # pinned by the stage server


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(download):
    m = json.loads((HERE / 'model-manifest.json').read_text())
    bad = []
    for f in m['files']:
        p = download / f['path']
        if not p.is_file():
            bad.append(f"missing {f['path']}")
        elif p.stat().st_size != f['size']:
            bad.append(f"size {f['path']}: {p.stat().st_size} != {f['size']}")
    if (download / 'config.json').is_file() and sha(download / 'config.json') != ORIGINAL_CONFIG_SHA:
        bad.append('config.json is not the pinned revision\'s')
    return m, bad


def main():
    args = sys.argv[1:]
    only_check = args[:1] == ['--check']
    if only_check:
        args = args[1:]
    if len(args) != (1 if only_check else 2):
        sys.exit(__doc__)
    download = Path(args[0]).expanduser().resolve()
    m, bad = check(download)
    if bad:
        print(f'{download} does not match {m["repo"]}@{m["revision"][:10]} ({len(bad)} problems):', file=sys.stderr)
        for b in bad[:20]:
            print('  ' + b, file=sys.stderr)
        sys.exit(1)
    print(f'model download OK: {len(m["files"])} files, {m["bytes"] / 2**30:.1f} GiB ({m["repo"]}@{m["revision"][:10]})')
    if only_check:
        return
    view = Path(args[1]).expanduser().absolute()
    if view.name != m['name']:
        sys.exit(f'the view directory must be named {m["name"]} (Sushi reports the directory name as the model id)')
    if view.resolve() == download:
        sys.exit('the view directory must differ from the download directory')
    view.mkdir(parents=True, exist_ok=True)
    for f in m['files']:
        src, dst = download / f['path'], view / f['path']
        dst.parent.mkdir(parents=True, exist_ok=True)
        if f['path'] == 'config.json':
            continue
        if dst.is_symlink() or dst.exists():
            if dst.is_symlink() and Path(os.readlink(dst)) == src:
                continue
            dst.unlink()
        dst.symlink_to(src)
    native = HERE / 'model-config-native-rope.json'
    if sha(native) != NATIVE_CONFIG_SHA:
        sys.exit(f'{native} is not the pinned native-rope config')
    tmp = view / 'config.json.tmp'
    tmp.write_bytes(native.read_bytes())
    tmp.replace(view / 'config.json')
    print(f'model directory: {view} (symlinks + native-rope config.json)')


if __name__ == '__main__':
    main()
