"""Kaggle CPU concurrency probe: record start/end times of a fixed 240 s sleep."""
import datetime as dt
import json
import os
import time

start = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
time.sleep(240)
end = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
os.makedirs("/kaggle/working/probe", exist_ok=True)
json.dump({"start_utc": start, "end_utc": end, "cpu_count": os.cpu_count()},
          open("/kaggle/working/probe/times.json", "w"))
print(start, end, flush=True)
