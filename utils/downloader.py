"""Download a recording or transcript from a link (Google Drive or direct URL)."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import gdown
import requests

CHUNK_SIZE = 1 << 20
TIMEOUT_S = 60


def _is_google_drive(url: str) -> bool:
    host = urlparse(url).netloc
    return "drive.google.com" in host or "docs.google.com" in host


def _filename_from_response(response: requests.Response, url: str) -> str:
    disposition = response.headers.get("content-disposition", "")
    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', disposition)
    if match:
        return unquote(match.group(1))
    return Path(urlparse(url).path).name or "download"


def download(url: str, dest_dir: str | Path) -> Path:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if _is_google_drive(url):
        output = gdown.download(url=url, output=f"{dest_dir}/", quiet=True, fuzzy=True)
        if not output:
            raise RuntimeError(f"Google Drive download failed (is the link shared publicly?): {url}")
        return Path(output)

    with requests.get(url, stream=True, timeout=TIMEOUT_S) as response:
        response.raise_for_status()
        path = dest_dir / _filename_from_response(response, url)
        with path.open("wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                f.write(chunk)
    return path
