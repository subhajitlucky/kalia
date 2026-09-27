"""Fetch a running Kaggle kernel's live log.

`kaggle kernels logs` only returns output once a session has finished, which is
useless for monitoring a multi-session run. The Python client's
`kernels_logs_stream` proxies the same SSE feed the browser uses, so a run in
progress can be read directly.

Usage:
    .venv/bin/python tools/kaggle_live_log.py kalia-train-v020 --seconds 90
    .venv/bin/python tools/kaggle_live_log.py kalia-train-v020 --seconds 60 \
        --grep "step |val loss|checkpoint"

Progress bars (the 566 MB checkpoint uploads) are dropped by default: they
dominate the feed and carry no information.
"""

from __future__ import annotations

import argparse
import re
import sys
import time

NOISE = (
    "ckpt.pt",
    "New Data Upload",
    "Processing Files",
    "MB/s",
    "B /",
    "it/s]",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", help="kernel slug, e.g. kalia-train-v020")
    parser.add_argument("--owner", default="subhajitlucky")
    parser.add_argument("--seconds", type=float, default=60.0, help="how long to listen")
    parser.add_argument("--grep", default=None, help="only print lines containing this substring")
    parser.add_argument("--dedup", action="store_true", help="drop consecutive duplicate lines")
    args = parser.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    started = time.time()
    printed = 0
    last = None
    try:
        for event in api.kernels_logs_stream(f"{args.owner}/{args.kernel}"):
            for raw in event.get("data", "").splitlines():
                line = raw.replace("\r", " ").strip()
                if not line or any(token in line for token in NOISE):
                    continue
                if re.fullmatch(r"[=\-|\s]*", line):
                    continue
                if args.dedup and line == last:
                    continue
                if args.grep and args.grep not in line:
                    last = line
                    continue
                print(line, flush=True)
                printed += 1
                last = line
            if time.time() - started > args.seconds:
                break
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # noqa: BLE001 - the SSE feed drops mid-run; report, don't crash
        print(f"[stream ended: {type(exc).__name__}: {exc}]", file=sys.stderr)

    print(f"[{printed} lines in {time.time() - started:.0f}s]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
