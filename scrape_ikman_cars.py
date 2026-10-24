"""Scrape Sri Lankan used-car ads from ikman.lk into a CSV (coursework use only).

Polite by design: plain listing/ad URLs only (allowed by robots.txt), fixed delay
between requests, and a resumable cache so an interrupted run continues where it left off.

Usage:
    python3 scrape_ikman_cars.py --max-ads 3000 --delay 1.5
    # MLOps: collect a fresh batch of ads not yet in the training data
    python3 scrape_ikman_cars.py --max-ads 250 --max-pages 60 --exclude-csv ikman_cars.csv \\
        --cache data/fresh_raw.jsonl --out-csv data/fresh_batch.csv
"""
import argparse
import csv
import json
import re
import time
from pathlib import Path

import requests

BASE = "https://ikman.lk"
LIST_URL = BASE + "/en/ads/sri-lanka/cars"
HEADERS = {"User-Agent": "Mozilla/5.0 (academic coursework scraper)"}
OUT_DIR = Path(__file__).parent
CACHE = OUT_DIR / "ikman_cars_raw.jsonl"
CSV_OUT = OUT_DIR / "ikman_cars.csv"

FIELDS = ["slug", "title", "price_lkr", "brand", "model", "model_year", "condition",
          "transmission", "body", "fuel_type", "engine_cc", "mileage_km", "location",
          "is_verified", "is_member", "posted_date", "description"]


def get_initial_data(url, session):
    for attempt in range(4):
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    html = r.content.decode("utf-8", errors="replace")
    i = html.find("window.initialData = ")
    if i < 0:
        raise ValueError("initialData not found")
    data, _ = json.JSONDecoder().raw_decode(html[i + len("window.initialData = "):])
    return data


def to_int(text):
    digits = re.sub(r"[^\d]", "", text or "")
    return int(digits) if digits else None


def parse_ad(data, price_text):
    ad = data["adDetail"]["data"]["ad"]
    props = {p["key"]: p["value"] for p in ad.get("properties", [])}
    loc = ad.get("location") or {}
    return {
        "slug": ad["slug"],
        "title": ad.get("title"),
        "price_lkr": to_int(price_text),
        "brand": props.get("brand"),
        "model": props.get("model"),
        "model_year": to_int(props.get("model_year")),
        "condition": props.get("condition"),
        "transmission": props.get("transmission"),
        "body": props.get("body"),
        "fuel_type": props.get("fuel_type"),
        "engine_cc": to_int(props.get("engine_capacity")),
        "mileage_km": to_int(props.get("mileage")),
        "location": loc.get("name"),
        "is_verified": ad.get("isVerified"),
        "is_member": ad.get("isMember"),
        "posted_date": ad.get("adDate"),
        "description": (ad.get("description") or "").replace("\n", " ")[:500],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-ads", type=int, default=3000)
    ap.add_argument("--delay", type=float, default=1.5, help="seconds between requests")
    ap.add_argument("--max-pages", type=int, default=500)
    ap.add_argument("--cache", default=str(CACHE), help="raw jsonl cache (resumable)")
    ap.add_argument("--out-csv", default=str(CSV_OUT))
    ap.add_argument("--exclude-csv", default=None, help="skip ads whose slug is already in this CSV")
    args = ap.parse_args()
    cache, csv_out = Path(args.cache), Path(args.out_csv)
    cache.parent.mkdir(parents=True, exist_ok=True); csv_out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if cache.exists():
        with cache.open() as f:
            done = {json.loads(line)["slug"] for line in f if line.strip()}
    known = set()
    if args.exclude_csv:
        with open(args.exclude_csv, newline="", encoding="utf-8") as f:
            known = {row["slug"] for row in csv.DictReader(f)}
    done |= known
    saved = len(done) - len(known)
    print(f"Resuming with {saved} ads already saved; skipping {len(known)} known ads")

    session = requests.Session()
    with cache.open("a") as out:
        for page in range(1, args.max_pages + 1):
            if saved >= args.max_ads:
                break
            try:
                listing = get_initial_data(f"{LIST_URL}?page={page}", session)
            except Exception as e:
                print(f"page {page}: listing failed ({e}); stopping")
                break
            time.sleep(args.delay)
            ads = listing["serp"]["ads"]["data"]["ads"]
            if not ads:
                print(f"page {page}: no ads, stopping")
                break
            for item in ads:
                slug = item["slug"]
                if slug in done:
                    continue
                if saved >= args.max_ads:
                    break
                try:
                    data = get_initial_data(f"{BASE}/en/ad/{slug}", session)
                    row = parse_ad(data, item.get("price"))
                except Exception as e:
                    print(f"  skip {slug}: {e}")
                    time.sleep(args.delay)
                    continue
                out.write(json.dumps(row) + "\n")
                out.flush()
                done.add(slug)
                saved += 1
                if saved % 25 == 0:
                    print(f"  {saved} ads saved (page {page})")
                time.sleep(args.delay)

    rows = [json.loads(line) for line in cache.open() if line.strip()]
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {csv_out}")


if __name__ == "__main__":
    main()
