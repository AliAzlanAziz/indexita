#!/usr/bin/env python3
"""
Fetch ~11,477 subreddit records from the Social Rise API with pagination,
respect a 150 req/min rate limit, and save to a single JSON array file.
Additional rule: if X-RateLimit-Remaining == 1, wait 70s before next request.
"""
# https://social-rise.com/blog/list-of-nsfw-subreddits

import json
import math
import time
from collections import deque
from datetime import datetime, timedelta, timezone

import requests
from requests.adapters import HTTPAdapter, Retry

# ------------------------- Configuration -------------------------

OUTPUT_PATH = "nsfw_subreddits_via_social_rise.json"
TARGET_TOTAL = 11477   # <- per your note
PER_PAGE = 100
START_PAGE = 1

# Rate limiting: at most 150 requests per 60 seconds
MAX_REQ_PER_WINDOW = 150
WINDOW_SECONDS = 60

# Backoff for transient errors (429/5xx)
MAX_RETRIES_PER_REQUEST = 5
BACKOFF_BASE_SECONDS = 1.5

# Optional: user agent is nice for servers/logs
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"


BASE_URL = "https://app.social-rise.com/api/subreddits"
BASE_PARAMS = {
    "per_page": str(PER_PAGE),
    "sort_by": "subscribers",
    "sort_dir": "desc",
    "filters[0][field]": "status",
    "filters[0][operator]": "<>",
    "filters[0][value]": "noexist",
    "filters[0][type]": "1",
    "filters[1][field]": "nsfw",
    "filters[1][operator]": "=",
    "filters[1][value]": "0",
    "filters[1][type]": "2",
    "filters[2][field]": "name",
    "filters[2][operator]": "not like",
    "filters[2][value]": r"u\_%",
    "filters[2][type]": "1",
    "filters[3][field]": "status",
    "filters[3][operator]": "<>",
    "filters[3][value]": "banned",
    "filters[3][type]": "1",
    "search_fields[]": "name",
    "search_value": "",
}

# ------------------------- Helpers -------------------------

class MinuteRateLimiter:
    """Sliding-window limiter: <= N events per WINDOW_SECONDS."""
    def __init__(self, max_events: int, window_seconds: int):
        self.max_events = max_events
        self.window = timedelta(seconds=window_seconds)
        self.events = deque()

    def wait_for_slot(self):
        now = datetime.now(timezone.utc)
        while self.events and (now - self.events[0]) > self.window:
            self.events.popleft()
        if len(self.events) >= self.max_events:
            wait_s = (self.window - (now - self.events[0])).total_seconds()
            if wait_s > 0:
                time.sleep(wait_s)
        self.events.append(datetime.now(timezone.utc))


def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=0)
    adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def fetch_page(session: requests.Session, page: int, limiter: MinuteRateLimiter) -> dict:
    params = dict(BASE_PARAMS)
    params["page"] = str(page)
    attempt = 0

    while True:
        attempt += 1
        limiter.wait_for_slot()

        try:
            resp = session.get(BASE_URL, params=params, timeout=30)
            rate_remaining = resp.headers.get("X-RateLimit-Remaining")

            # If remaining == 1, pause 70s before proceeding further
            if rate_remaining is not None:
                try:
                    remaining = int(rate_remaining)
                    if remaining == 1:
                        print("Rate limit nearly exhausted (1 remaining). Sleeping 70s...")
                        time.sleep(70)
                except ValueError:
                    pass

            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code in (429, 502, 503, 504, 520, 521, 522):
                if attempt > MAX_RETRIES_PER_REQUEST:
                    raise RuntimeError(f"HTTP {resp.status_code} after {attempt-1} retries")
                sleep_s = (BACKOFF_BASE_SECONDS ** attempt)
                time.sleep(sleep_s)
                continue
            else:
                resp.raise_for_status()

        except (requests.Timeout, requests.ConnectionError) as e:
            if attempt > MAX_RETRIES_PER_REQUEST:
                raise RuntimeError(f"Network error after {attempt-1} retries: {e}") from e
            sleep_s = (BACKOFF_BASE_SECONDS ** attempt)
            time.sleep(sleep_s)
            continue


def main():
    session = make_session()
    limiter = MinuteRateLimiter(MAX_REQ_PER_WINDOW, WINDOW_SECONDS)

    all_items = []
    seen_ids = set()

    page = START_PAGE
    pages_estimate = math.ceil(TARGET_TOTAL / PER_PAGE)  # ~115 pages
    print(f"Fetching ~{TARGET_TOTAL} records across ~{pages_estimate} pages...")

    while True:
        data = fetch_page(session, page, limiter)

        items = data.get("data", [])
        if not isinstance(items, list):
            raise RuntimeError(f"Unexpected payload shape on page {page}: {type(items)}")

        for it in items:
            it_id = it.get("id")
            if it_id is None or it_id not in seen_ids:
                all_items.append(it)
                if it_id is not None:
                    seen_ids.add(it_id)

        print(f"Page {page}: received {len(items)} items | total collected: {len(all_items)}")

        # Stop if last page or we hit the target total
        if len(items) < PER_PAGE:
            print("Reached a short (last) page; stopping.")
            break
        if len(all_items) >= TARGET_TOTAL:
            print(f"Reached target total ({TARGET_TOTAL}); stopping.")
            all_items = all_items[:TARGET_TOTAL]
            break

        page += 1

    # Save a single JSON array
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_items)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
