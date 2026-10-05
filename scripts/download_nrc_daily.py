"""Download historical daily NRC reactor status reports."""

import argparse
import hashlib
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "nrc" / "daily"

BASE_URL = (
    "https://www.nrc.gov/documents-reports/document-collections/"
    "events-reports-associated-with/power-reactor-status-reports"
)

DEFAULT_START = date(1999, 1, 1)
DEFAULT_END = (
    datetime.now(ZoneInfo("America/Chicago")).date() - timedelta(days=1)
)


def download_reports(start, end, delay):
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    failure_path = RAW_DIR / (
        f"failures_{start.isoformat()}_{end.isoformat()}.json"
    )
    failures = []
    downloaded = 0
    skipped = 0

    def save_failures():
        failure_path.write_text(
            json.dumps(failures, indent=2) + "\n",
            encoding="utf-8",
        )

    day = start

    try:
        with requests.Session() as session:
            while day <= end:
                stamp = day.strftime("%Y%m%d")
                url = f"{BASE_URL}/{day.year}/{stamp}ps"

                folder = RAW_DIR / str(day.year)
                destination = folder / f"{stamp}ps.html"
                metadata_path = folder / f"{stamp}ps.metadata.json"

                if destination.exists() and metadata_path.exists():
                    skipped += 1
                    day += timedelta(days=1)
                    continue

                try:
                    response = session.get(url, timeout=60)

                    if response.status_code in (403, 429):
                        message = (
                            f"NRC returned {response.status_code}. "
                            "Stopping; completed downloads are preserved."
                        )
                        failures.append({
                            "date": day.isoformat(),
                            "url": url,
                            "error": message,
                        })
                        raise SystemExit(message)

                    response.raise_for_status()
                    content = response.content
                    text = response.text.lower()

                    # Basic screening; table parsing is a separate step.
                    if not content or not all(
                        field in text
                        for field in ("scrams", "unit", "power")
                    ):
                        raise ValueError(
                            "Response does not look like a reactor report"
                        )

                    folder.mkdir(parents=True, exist_ok=True)

                    temporary = destination.with_suffix(".html.part")
                    temporary.write_bytes(content)
                    temporary.replace(destination)

                    metadata = {
                        "report_date": day.isoformat(),
                        "source_url": url,
                        "resolved_url": response.url,
                        "retrieved_at_utc": datetime.now(
                            timezone.utc
                        ).isoformat(),
                        "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }

                    metadata_temporary = metadata_path.with_suffix(
                        ".json.part"
                    )
                    metadata_temporary.write_text(
                        json.dumps(metadata, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    metadata_temporary.replace(metadata_path)

                    downloaded += 1
                    print(f"Downloaded {day}", flush=True)

                except (requests.RequestException, ValueError) as error:
                    failures.append({
                        "date": day.isoformat(),
                        "url": url,
                        "error": str(error),
                    })
                    save_failures()
                    print(f"Failed {day}: {error}", flush=True)

                day += timedelta(days=1)
                time.sleep(delay)

    finally:
        save_failures()
        print(
            f"\nDownloaded: {downloaded}; "
            f"skipped: {skipped}; failed: {len(failures)}",
            flush=True,
        )

    if failures:
        raise SystemExit(f"Review failed downloads in {failure_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        default=DEFAULT_START,
        help="First date, YYYY-MM-DD (default: 1999-01-01)",
    )
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        default=DEFAULT_END,
        help="Last date, YYYY-MM-DD (default: yesterday in Chicago)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds between requests (default: 2)",
    )
    args = parser.parse_args()

    if args.end < args.start:
        parser.error("--end must be on or after --start")
    if args.delay < 0:
        parser.error("--delay must be nonnegative")

    print(f"Downloading {args.start} through {args.end}", flush=True)
    print(f"Saving to {RAW_DIR}", flush=True)
    download_reports(args.start, args.end, args.delay)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Run again to resume.")
        raise SystemExit(130)