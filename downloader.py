#!/usr/bin/env python3
"""
downloader.py
=============

Generic, multi-site file downloader with "only download new files" behaviour.

- Reads a list of sources from config.json (CKAN open-data portals, or plain
  web pages with linked files).
- Keeps a manifest (downloaded_manifest.json) of everything already fetched.
- On every run, for every source, it figures out which files are currently
  offered, skips anything already present (either recorded in the manifest
  OR already sitting in the destination folder with the same filename), and
  downloads only what's new.
- Designed to be triggered on a schedule (cron / Task Scheduler / Colab
  scheduled runs) once a week, see README.md for setup.

Google Colab:
    When this script (and config.json) live in a Google Drive folder and are
    run inside Colab, it auto-mounts Drive and stores downloads/manifest/log
    under config["drive_root"] instead of a local folder, so everything
    persists across Colab sessions. Outside Colab it behaves exactly as
    before (paths resolve relative to the current working directory).

Usage:
    python3 downloader.py                # run all enabled sources
    python3 downloader.py --source anac_partecipanti   # run just one
    python3 downloader.py --dry-run      # show what would be downloaded
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse, unquote

import requests
from bs4 import BeautifulSoup

# A realistic browser User-Agent. Several government open-data portals (ANAC
# included) run a WAF that rejects requests that look automated, so we
# identify as a normal browser and send the usual companion headers.
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
}

CONFIG_PATH = Path(__file__).with_name("config.json")


# --------------------------------------------------------------------------- #
# Google Colab / Drive helpers
# --------------------------------------------------------------------------- #

def in_colab() -> bool:
    """True when running inside a Google Colab runtime."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def mount_drive_if_colab() -> None:
    """
    If running in Colab, mount Google Drive at /content/drive (no-op if it's
    already mounted). Outside Colab this does nothing, so the script still
    runs unmodified on a normal machine.
    """
    if not in_colab():
        return
    drive_mountpoint = Path("/content/drive")
    if (drive_mountpoint / "MyDrive").exists():
        return  # already mounted
    from google.colab import drive  # type: ignore
    drive.mount(str(drive_mountpoint))


def resolve_drive_path(config: dict, key: str, default: str) -> Path:
    """
    Resolve a path from config.

    - In Google Colab, relative paths are anchored under config["drive_root"].
    - On Windows/local execution, the default download location is
      ~/Downloads/WeeklyDownloader so files remain on the laptop instead of
      being written into the temporary GitHub checkout.
    - An absolute path in config.json always wins.
    - Other relative paths remain relative to the current working directory.
    """
    raw = config.get(key)

    if raw is None:
        if key == "download_folder":
            if os.name == "nt" and not in_colab():
                return Path.home() / "Downloads" / "WeeklyDownloader"
        raw = default

    p = Path(raw)
    if p.is_absolute():
        return p

    drive_root = config.get("drive_root")
    if drive_root and in_colab():
        return Path(drive_root) / p

    # For the default local download location, keep it stable on the laptop.
    if key == "download_folder" and raw == "./downloads" and os.name == "nt":
        return Path.home() / "Downloads" / "WeeklyDownloader"

    return p.resolve()


# --------------------------------------------------------------------------- #
# Setup / helpers
# --------------------------------------------------------------------------- #

def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(path: Path, manifest: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=True)


def setup_logging(log_file: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def url_key(url: str) -> str:
    """Stable identifier for a URL, used as the manifest key."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def safe_filename_from_url(url: str, fallback_ext: str = "") -> str:
    name = unquote(Path(urlparse(url).path).name)
    if not name:
        name = "file" + fallback_ext
    return name


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(BROWSER_HEADERS)
    return session


# --------------------------------------------------------------------------- #
# Source handlers: each returns a list of dicts: {"url": ..., "filename": ...}
# --------------------------------------------------------------------------- #

def list_files_ckan(session: requests.Session, source: dict, timeout: int) -> list:
    """
    CKAN open-data portals (dati.anticorruzione.it, dati.gov.it, most
    regional/city Italian portals, and many others worldwide run CKAN).

    Calls package_show for the given dataset_id and returns every resource's
    direct download URL.
    """
    base_url = source["base_url"].rstrip("/")
    dataset_id = source["dataset_id"]
    endpoint = f"{base_url}/api/3/action/package_show"
    method = source.get("http_method", "post").lower()

    headers = dict(session.headers)
    headers["Referer"] = f"{base_url}/dataset/{dataset_id}"

    if method == "post":
        resp = session.post(endpoint, json={"id": dataset_id}, headers=headers, timeout=timeout)
    else:
        resp = session.get(endpoint, params={"id": dataset_id}, headers=headers, timeout=timeout)

    if "application/json" not in resp.headers.get("Content-Type", ""):
        raise RuntimeError(
            f"Expected JSON from {endpoint} but got '{resp.headers.get('Content-Type')}' "
            f"(status {resp.status_code}). The site's WAF may be blocking this request. "
            f"First 200 chars of response: {resp.text[:200]!r}"
        )

    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"CKAN API returned an error for dataset '{dataset_id}': {data}")

    files = []
    for res in data["result"].get("resources", []):
        res_url = res.get("url")
        if not res_url:
            continue
        fmt = (res.get("format") or "").lower().strip()
        ext = f".{fmt}" if fmt else ""
        name = res.get("name") or safe_filename_from_url(res_url, ext)
        # Make sure the filename actually has an extension.
        if "." not in Path(name).name:
            name = f"{name}{ext}"
        files.append({"url": res_url, "filename": name})
    return files


def list_files_html_links(session: requests.Session, source: dict, timeout: int) -> list:
    """
    Generic handler for a plain web page that links directly to downloadable
    files (e.g. <a href="report_2024.csv">). Filters links by extension.
    """
    page_url = source["page_url"]
    extensions = tuple(e.lower() for e in source.get("file_extensions", []))

    resp = session.get(page_url, timeout=timeout)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    files = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full_url = requests.compat.urljoin(page_url, href)
        if not full_url.lower().endswith(extensions):
            continue
        if full_url in seen:
            continue
        seen.add(full_url)
        files.append({"url": full_url, "filename": safe_filename_from_url(full_url)})
    return files


SOURCE_HANDLERS = {
    "ckan": list_files_ckan,
    "html_links": list_files_html_links,
}


# --------------------------------------------------------------------------- #
# Download logic
# --------------------------------------------------------------------------- #

def download_file(session: requests.Session, url: str, dest: Path, timeout: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    with session.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
    tmp_path.rename(dest)


def process_source(session: requests.Session, source: dict, base_download_folder: Path,
                    manifest: dict, timeout: int, dry_run: bool) -> None:
    name = source["name"]
    stype = source.get("type")
    handler = SOURCE_HANDLERS.get(stype)
    if handler is None:
        logging.error("[%s] Unknown source type '%s' - skipping.", name, stype)
        return

    dest_folder = base_download_folder / source.get("subfolder", name)
    dest_folder.mkdir(parents=True, exist_ok=True)

    manifest.setdefault(name, {})

    try:
        candidates = handler(session, source, timeout)
    except Exception as exc:
        logging.error("[%s] Could not list files: %s", name, exc)
        return

    logging.info("[%s] Found %d file(s) listed at the source.", name, len(candidates))

    new_count = 0
    for item in candidates:
        url, filename = item["url"], item["filename"]
        key = url_key(url)
        dest_path = dest_folder / filename

        already_in_manifest = key in manifest[name]
        already_on_disk = dest_path.exists()

        if already_in_manifest:
            logging.info("[%s] SKIPPED - already recorded in manifest: %s", name, filename)
            continue

        if already_on_disk:
            # A file with the same filename is already present on the laptop.
            # Record it in the manifest so future runs skip it too.
            logging.info("[%s] SKIPPED - file already exists: %s", name, filename)
            manifest[name][key] = {
                "url": url,
                "filename": filename,
                "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "recovered_existing_file": True,
            }
            continue

        new_count += 1
        if dry_run:
            logging.info("[%s] (dry-run) Would download: %s -> %s", name, url, dest_path)
            continue

        logging.info("[%s] Downloading new file: %s", name, filename)
        try:
            download_file(session, url, dest_path, timeout)
        except Exception as exc:
            logging.error("[%s] Failed to download %s: %s", name, url, exc)
            continue

        manifest[name][key] = {
            "url": url,
            "filename": filename,
            "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    if new_count == 0:
        logging.info("[%s] No new files - up to date.", name)
    else:
        logging.info("[%s] Done: %d new file(s) processed.", name, new_count)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly multi-site new-file downloader")
    parser.add_argument("--source", help="Only run the source with this 'name' from config.json")
    parser.add_argument("--dry-run", action="store_true", help="List what would be downloaded, don't save anything")
    args = parser.parse_args()

    mount_drive_if_colab()

    config = load_config()

    base_download_folder = resolve_drive_path(config, "download_folder", "./downloads")
    manifest_path = resolve_drive_path(config, "manifest_file", "./downloaded_manifest.json")
    log_path = resolve_drive_path(config, "log_file", "./downloader.log")

    base_download_folder.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    setup_logging(str(log_path))

    timeout = config.get("request_timeout", 60)

    manifest = load_manifest(manifest_path)
    session = make_session()

    sources = config.get("sources", [])
    if args.source:
        sources = [s for s in sources if s.get("name") == args.source]
        if not sources:
            logging.error("No source named '%s' found in config.json", args.source)
            sys.exit(1)

    logging.info("=== Run started (%d source(s) to check) ===", len(sources))
    for source in sources:
        if not source.get("enabled", True):
            logging.info("[%s] Disabled - skipping.", source["name"])
            continue
        process_source(session, source, base_download_folder, manifest, timeout, args.dry_run)

    if not args.dry_run:
        save_manifest(manifest_path, manifest)
    logging.info("=== Run finished ===")


if __name__ == "__main__":
    main()
