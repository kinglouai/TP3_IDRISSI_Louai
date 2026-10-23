"""Attendre que le serveur MLflow réponde avant de lancer DVC."""

from __future__ import annotations

import argparse
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


def wait_for_server(url: str, retries: int = 30, interval: float = 1.0) -> bool:
    for attempt in range(1, retries + 1):
        try:
            with urlopen(url, timeout=3):
                print(f"MLflow server accessible: {url}")
                return True
        except (OSError, URLError):
            if attempt < retries:
                time.sleep(interval)
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5000/")
    parser.add_argument("--retries", type=int, default=30)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    if wait_for_server(args.url, retries=args.retries, interval=args.interval):
        return 0
    print(
        f"MLflow server inaccessible après {args.retries} tentatives : {args.url}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
