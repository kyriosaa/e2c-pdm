# Export recorded sessions from data/raw to a removable drive, with checksums.
#
# This machine only records; the analysis machine is elsewhere, so every session
# makes the trip as ~4.5 GB of opaque binary. A truncated vibration.bin or a
# single flipped byte does not announce itself -- it surfaces later as a
# "transient" in the spectrogram that looks exactly like an impulsive fault.
# So every file is SHA-256'd as it is copied and the digests travel with the
# data in export_manifest.json.
#
#   python export_sessions.py E:/pdm_transfer            # copy everything new
#   python export_sessions.py E:/pdm_transfer --only healthy24V_..._20260719_010905
#   python export_sessions.py E:/pdm_transfer --skip-tests
#   python export_sessions.py E:/pdm_transfer --verify   # re-hash, copy nothing
#
# On the analysis machine, point --verify at the drive (or at the copy made from
# it) before deleting anything from the source:
#
#   python export_sessions.py E:/pdm_transfer --verify
#
# Files already present at the destination with a matching size and digest are
# skipped, so re-running after an interrupted transfer resumes where it stopped.

import os
import sys
import json
import time
import shutil
import hashlib
import argparse

RAW_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', '..', 'data', 'raw')
MANIFEST_NAME = 'export_manifest.json'
MANIFEST_VERSION = 1
CHUNK = 8 * 1024 * 1024        # 8 MiB: large enough that USB throughput, not
                               # syscall overhead, is the limit

# Written by data_catcher.py. A session missing any of these is incomplete and
# is refused rather than silently half-exported.
REQUIRED = ('session.json', 'vibration.bin', 'audio.wav',
            'status.csv', 'packet_index.csv')


# ---------------------------------------------------------------- helpers
def human(n):
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024 or unit == 'TB':
            return f'{n:.1f} {unit}' if unit != 'B' else f'{n} B'
        n /= 1024


def sha256_file(path, progress=None):
    h = hashlib.sha256()
    done = 0
    with open(path, 'rb') as f:
        while True:
            buf = f.read(CHUNK)
            if not buf:
                break
            h.update(buf)
            done += len(buf)
            if progress:
                progress(done)
    return h.hexdigest()


def copy_and_hash(src, dst, progress=None):
    """Copy through a .part file, hashing the bytes actually written.

    The digest comes from the write path, so it certifies what landed on the
    destination -- not what we hoped we read. The rename is the commit point:
    an interrupted run leaves a .part behind and never a short file that looks
    complete.
    """
    h = hashlib.sha256()
    part = dst + '.part'
    done = 0
    with open(src, 'rb') as fi, open(part, 'wb') as fo:
        while True:
            buf = fi.read(CHUNK)
            if not buf:
                break
            fo.write(buf)
            h.update(buf)
            done += len(buf)
            if progress:
                progress(done)
        fo.flush()
        os.fsync(fo.fileno())
    if os.path.exists(dst):
        os.remove(dst)
    os.replace(part, dst)
    return h.hexdigest()


def clear_line():
    sys.stdout.write('\r' + ' ' * 100 + '\r')


def bar(label, done, total, started):
    pct = done / total * 100 if total else 100.0
    rate = done / max(1e-9, time.time() - started)
    eta = (total - done) / rate if rate > 0 else 0
    sys.stdout.write(f'\r    {label:<18} {pct:5.1f}%  {human(rate)}/s  '
                     f'ETA {int(eta // 60):02d}:{int(eta % 60):02d}   ')
    sys.stdout.flush()


def load_manifest(dest):
    path = os.path.join(dest, MANIFEST_NAME)
    if not os.path.isfile(path):
        return {'manifest_version': MANIFEST_VERSION, 'sessions': {}}
    with open(path) as f:
        m = json.load(f)
    if m.get('manifest_version') != MANIFEST_VERSION:
        sys.exit(f'{path}: unsupported manifest_version {m.get("manifest_version")}')
    return m


def save_manifest(dest, manifest):
    # Written after every session so an interrupted transfer keeps the digests
    # of the sessions that did complete.
    path = os.path.join(dest, MANIFEST_NAME)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def discover(root, only, skip_tests):
    if not os.path.isdir(root):
        sys.exit(f'no such directory: {root}')
    names = sorted(d for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d)))
    if only:
        missing = [n for n in only if n not in names]
        if missing:
            sys.exit(f'not found in {root}: {", ".join(missing)}')
        names = [n for n in names if n in only]
    if skip_tests:
        names = [n for n in names if not n.startswith('test_')]
    return names


# ---------------------------------------------------------------- verify
def verify(dest, manifest, only):
    sessions = manifest['sessions']
    if only:
        sessions = {k: v for k, v in sessions.items() if k in only}
    if not sessions:
        sys.exit(f'{MANIFEST_NAME} in {dest} lists no sessions to verify')

    bad, checked = [], 0
    for name in sorted(sessions):
        print(f'  {name}')
        for fname, meta in sorted(sessions[name]['files'].items()):
            path = os.path.join(dest, name, fname)
            if not os.path.isfile(path):
                print(f'    {fname:<18} MISSING')
                bad.append(f'{name}/{fname} missing')
                continue
            size = os.path.getsize(path)
            if size != meta['size']:
                print(f'    {fname:<18} SIZE {size} != {meta["size"]}')
                bad.append(f'{name}/{fname} wrong size')
                continue
            t0 = time.time()
            got = sha256_file(path, lambda d, t=size, s=t0, l=fname: bar(l, d, t, s))
            ok = got == meta['sha256']
            clear_line()
            print(f'    {fname:<18} {"ok" if ok else "CHECKSUM MISMATCH"}')
            if not ok:
                bad.append(f'{name}/{fname} checksum mismatch')
            checked += 1

    print()
    if bad:
        print(f'FAILED: {len(bad)} problem(s) across {checked} file(s)')
        for b in bad:
            print(f'  - {b}')
        sys.exit(1)
    print(f'All {checked} file(s) verified against {MANIFEST_NAME}.')


# ---------------------------------------------------------------- export
def export(root, dest, names, manifest, recheck):
    os.makedirs(dest, exist_ok=True)

    plan = []
    for name in names:
        sdir = os.path.join(root, name)
        missing = [f for f in REQUIRED if not os.path.isfile(os.path.join(sdir, f))]
        if missing:
            print(f'  skip {name}: incomplete ({", ".join(missing)})')
            continue
        files = sorted(f for f in os.listdir(sdir)
                       if os.path.isfile(os.path.join(sdir, f)))
        plan.append((name, sdir, files))

    if not plan:
        sys.exit('nothing to export')

    total = sum(os.path.getsize(os.path.join(d, f))
                for _n, d, fs in plan for f in fs)
    free = shutil.disk_usage(dest).free
    print(f'\n{len(plan)} session(s), {human(total)} total')
    print(f'destination {dest}: {human(free)} free')
    if free < total:
        # Already-copied files are skipped below, so this is a warning rather
        # than a hard stop -- a resumed transfer legitimately needs less space.
        print(f'  [warn] free space is less than the full payload; '
              f'unchanged files will be skipped, but this may still run out')

    copied_bytes = skipped_bytes = 0
    for name, sdir, files in plan:
        ddir = os.path.join(dest, name)
        os.makedirs(ddir, exist_ok=True)
        entry = manifest['sessions'].setdefault(name, {'files': {}})
        print(f'\n  {name}')

        for fname in files:
            src = os.path.join(sdir, fname)
            dst = os.path.join(ddir, fname)
            size = os.path.getsize(src)
            known = entry['files'].get(fname)

            if (not recheck and known and known['size'] == size
                    and os.path.isfile(dst) and os.path.getsize(dst) == size):
                print(f'    {fname:<18} skip (already exported)')
                skipped_bytes += size
                continue

            t0 = time.time()
            digest = copy_and_hash(
                src, dst, lambda d, t=size, s=t0, l=fname: bar(l, d, t, s))
            clear_line()
            elapsed = time.time() - t0
            print(f'    {fname:<18} {human(size)} in {elapsed:.0f}s  '
                  f'{digest[:12]}...')
            entry['files'][fname] = {'size': size, 'sha256': digest}
            copied_bytes += size

        entry['exported_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                              time.gmtime())
        save_manifest(dest, manifest)

    print(f'\nCopied {human(copied_bytes)}'
          f'{f", skipped {human(skipped_bytes)} already present" if skipped_bytes else ""}.')
    print(f'Manifest: {os.path.join(dest, MANIFEST_NAME)}')
    print('\nOn the analysis machine, before deleting anything here:')
    print(f'  python export_sessions.py <copied-location> --verify')


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description='Copy sessions from data/raw to a drive, with checksums.')
    ap.add_argument('dest', help='destination directory (e.g. E:/pdm_transfer)')
    ap.add_argument('--source', default=RAW_ROOT,
                    help='session root (default: repo data/raw)')
    ap.add_argument('--only', nargs='+', metavar='SESSION',
                    help='export/verify only these session directory names')
    ap.add_argument('--skip-tests', action='store_true',
                    help='exclude test_* sessions')
    ap.add_argument('--verify', action='store_true',
                    help='re-hash the destination against its manifest; copy nothing')
    ap.add_argument('--recheck', action='store_true',
                    help='re-copy and re-hash even files the manifest already covers')
    args = ap.parse_args()

    manifest = load_manifest(args.dest)

    if args.verify:
        print(f'Verifying {args.dest} against {MANIFEST_NAME}')
        verify(args.dest, manifest, args.only)
        return

    root = os.path.abspath(args.source)
    names = discover(root, args.only, args.skip_tests)
    if not names:
        sys.exit(f'no session directories in {root}')
    print(f'Source: {root}')
    export(root, args.dest, names, manifest, args.recheck)


if __name__ == '__main__':
    main()
