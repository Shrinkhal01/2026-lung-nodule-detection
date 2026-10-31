"""Phase 0-1 · F (Fetch): check and resume the LIDC-IDRI CT download.

  python scripts/01_fetch.py --check       # what is done / partial / missing (safe while a download runs)
  python scripts/01_fetch.py --download    # fetch only missing or partial series, in batches, logging MB/s

Do NOT run --download while another download into the same folder is still running.
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lungdet.fetch.inventory import scan_series_folders, series_status  # noqa: E402


def load_index():
    from idc_index import IDCClient
    client = IDCClient.client()
    cols = "PatientID, SeriesInstanceUID, series_size_MB"
    try:
        df = client.sql_query(f"SELECT {cols}, instanceCount FROM index "
                              "WHERE collection_id='lidc_idri' AND Modality='CT' ORDER BY PatientID")
    except Exception:  # older index without instanceCount -> 'non-empty folder' counts as done
        df = client.sql_query(f"SELECT {cols} FROM index "
                              "WHERE collection_id='lidc_idri' AND Modality='CT' ORDER BY PatientID")
        df["instanceCount"] = 0
    return client, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=os.path.expandvars("$SCRATCH/lidc_raw"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--batch", type=int, default=25, help="series per download call")
    ap.add_argument("--log", default=os.path.expandvars("$SCRATCH/logs/fetch_log.csv"))
    args = ap.parse_args()

    client, df = load_index()
    found = scan_series_folders(args.raw_dir)
    status = {u: series_status(found.get(u, 0), int(n)) for u, n in zip(df.SeriesInstanceUID, df.instanceCount)}
    done = [u for u, s in status.items() if s == "done"]
    todo = [u for u, s in status.items() if s != "done"]
    gb_done = df[df.SeriesInstanceUID.isin(done)].series_size_MB.sum() / 1024
    gb_all = df.series_size_MB.sum() / 1024
    n_partial = sum(s == "partial" for s in status.values())
    print(f"series: {len(done)} done, {n_partial} partial, {len(todo) - n_partial} missing, of {len(df)}")
    print(f"size:   {gb_done:.1f} of {gb_all:.1f} GB done")

    if not args.download or not todo:
        return
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    new_log = not os.path.exists(args.log)
    sizes = dict(zip(df.SeriesInstanceUID, df.series_size_MB))
    with open(args.log, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_log:
            w.writerow(["time", "n_series", "MB", "seconds", "MB_per_s"])
        for i in range(0, len(todo), args.batch):
            batch = todo[i:i + args.batch]
            mb = sum(sizes[u] for u in batch)
            t = time.time()
            client.download_dicom_series(seriesInstanceUID=batch, downloadDir=args.raw_dir)
            s = time.time() - t
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), len(batch), f"{mb:.0f}", f"{s:.0f}", f"{mb / s:.2f}"])
            fh.flush()
            print(f"batch {i // args.batch + 1}: {len(batch)} series, {mb:.0f} MB, {mb / s:.2f} MB/s")


if __name__ == "__main__":
    main()
