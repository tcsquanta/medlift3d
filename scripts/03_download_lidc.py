#!/usr/bin/env python
"""Step 3 - download only the selected LIDC-IDRI CT series from TCIA.

Uses the public NBIA REST API (no account, no Data Retriever app):
  https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID=...
Files go to <paths.lidc_root>/<SeriesInstanceUID>/*.dcm. Re-running skips
finished series, so it is safe to interrupt. Parallel downloads (--workers)
are much faster than one at a time.

Alternative: open work/lidc_subset.tcia with NBIA Data Retriever and point
paths.lidc_root at the download folder (any folder layout works; step 4
searches for the series).

    python scripts/03_download_lidc.py --workers 4
"""
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from _common import base_parser, setup

from medlift3d.data.lidc import download_series
from medlift3d.io import Paths
from medlift3d.utils import Checker


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg, _ = setup(args)
    paths = Paths(cfg)
    ck = Checker("Step 3: download LIDC-IDRI series")
    with open(paths.p("lidc_series.txt")) as f:
        uids = [l.strip() for l in f if l.strip()]
    if args.limit:
        uids = uids[: args.limit]
    os.makedirs(paths.lidc_root, exist_ok=True)
    ok, failed = 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_series, u, paths.lidc_root): u for u in uids}
        for i, fu in enumerate(as_completed(futs), 1):
            u = futs[fu]
            try:
                d = fu.result()
                n = len([x for x in os.listdir(d) if x.endswith(".dcm")])
                ok += 1
                print(f"[{i}/{len(uids)}] {u[-12:]}: {n} slices")
            except Exception as e:
                failed.append(u)
                print(f"[{i}/{len(uids)}] {u[-12:]}: FAILED {e}")
    ck.check("series downloaded", ok == len(uids), f"{ok}/{len(uids)}")
    if failed:
        with open(paths.p("lidc_failed.txt"), "w") as f:
            f.write("\n".join(failed) + "\n")
        ck.info("failed list", paths.p("lidc_failed.txt"))
    ck.finish(paths.out("03_download"))


if __name__ == "__main__":
    main()
