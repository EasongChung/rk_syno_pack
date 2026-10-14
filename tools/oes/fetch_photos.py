#!/usr/bin/env python3
"""拉取真实照片做校准/回归集（picsum.photos，失败容忍）.

用法：
  python fetch_photos.py --dest calib --prefix oes-calib --count 300
  python fetch_photos.py --dest eval --prefix oes-eval --count 200
要求成功数 >= --min-ok（默认 50），否则 exit 1。
"""
import argparse
import os
import time
import urllib.request

URL = 'https://picsum.photos/seed/%s-%d/800/600'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', required=True)
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--count', type=int, default=300)
    ap.add_argument('--min-ok', type=int, default=50)
    ap.add_argument('--timeout', type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.dest, exist_ok=True)
    ok = 0
    for i in range(args.count):
        dst = os.path.join(args.dest, '%s-%04d.jpg' % (args.prefix, i))
        if os.path.exists(dst) and os.path.getsize(dst) > 4096:
            ok += 1
            continue
        try:
            req = urllib.request.Request(
                URL % (args.prefix, i), headers={'User-Agent': 'oes-ci/1.0'})
            with urllib.request.urlopen(req, timeout=args.timeout) as r, \
                    open(dst, 'wb') as f:
                f.write(r.read())
            if os.path.getsize(dst) > 4096:
                ok += 1
            if i % 50 == 0:
                print('fetch %d/%d ok=%d' % (i, args.count, ok), flush=True)
            time.sleep(0.1)
        except Exception as e:
            print('skip %d: %s' % (i, e), flush=True)
    print('FETCH DONE ok=%d/%d -> %s' % (ok, args.count, args.dest), flush=True)
    if ok < args.min_ok:
        raise SystemExit('too few images: %d < %d' % (ok, args.min_ok))


if __name__ == '__main__':
    main()
