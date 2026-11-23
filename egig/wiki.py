"""Wikipedia + Wikimedia Commons client used by Agent 2 to fetch facts and reference photos.
Both APIs are free and need no key."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from PIL import Image

PHOTO_MIMES = ("image/jpeg", "image/png", "image/webp")


@dataclass
class WikiPage:
    title: str
    url: str
    extract: str
    image_url: Optional[str]


@dataclass
class CommonsImage:
    title: str
    url: str
    page_url: str
    width: int
    height: int


def user_agent(contact: str) -> str:
    """Wikimedia's required format: tool/version (contact) library/version."""
    return f"EntityGroundedImageGen/0.1 ({contact or 'no contact set'}) python-httpx/{httpx.__version__}"


class WikiClient:
    def __init__(self, contact: str, lang: str = "en", timeout: float = 20.0,
                 http: Optional[httpx.Client] = None, thumb_width: int = 960):
        self.lang = lang
        self.thumb_width = thumb_width
        self.contact = contact
        self._warned = False
        if not contact and http is None:
            print("[wiki] WIKI_CONTACT is empty: Wikimedia will likely refuse requests (HTTP 403). "
                  "Set WIKI_CONTACT in .env to your email or a URL.")
        self.http = http or httpx.Client(
            headers={"User-Agent": user_agent(contact)}, timeout=timeout, follow_redirects=True
        )

    # ------------------------------------------------------------------ search
    def find_page(self, query: str) -> Optional[WikiPage]:
        """Top Wikipedia search hit that is not a disambiguation page, with its intro text
        and lead image (as a ~960 px thumbnail)."""
        data = self._get_json(
            f"https://{self.lang}.wikipedia.org/w/api.php",
            {
                "action": "query", "format": "json", "formatversion": "2",
                "generator": "search", "gsrsearch": query, "gsrlimit": "5",
                "prop": "pageimages|extracts|pageprops|info",
                "piprop": "thumbnail|name", "pithumbsize": str(self.thumb_width),
                "exintro": "1", "explaintext": "1", "exlimit": "5",
                "inprop": "url", "redirects": "1",
            },
        )
        pages = sorted(data.get("query", {}).get("pages", []), key=lambda p: p.get("index", 999))
        for p in pages:
            if "disambiguation" in (p.get("pageprops") or {}):
                continue
            return WikiPage(
                title=p.get("title", query),
                url=p.get("fullurl", ""),
                extract=(p.get("extract") or "").strip(),
                image_url=(p.get("thumbnail") or {}).get("source"),
            )
        return None

    def commons_images(self, query: str, limit: int = 8) -> List[CommonsImage]:
        """Photos on Wikimedia Commons whose file page matches the query."""
        data = self._get_json(
            "https://commons.wikimedia.org/w/api.php",
            {
                "action": "query", "format": "json", "formatversion": "2",
                "generator": "search", "gsrsearch": f"{query} filetype:bitmap",
                "gsrnamespace": "6", "gsrlimit": str(limit),
                "prop": "imageinfo", "iiprop": "url|mime|size", "iiurlwidth": str(self.thumb_width),
            },
        )
        out: List[CommonsImage] = []
        pages = sorted(data.get("query", {}).get("pages", []), key=lambda p: p.get("index", 999))
        for p in pages:
            info = (p.get("imageinfo") or [{}])[0]
            if info.get("mime") not in PHOTO_MIMES or int(info.get("width", 0)) < 400:
                continue
            out.append(CommonsImage(
                title=p.get("title", ""),
                url=info.get("thumburl") or info.get("url", ""),
                page_url=info.get("descriptionurl", ""),
                width=int(info.get("width", 0)),
                height=int(info.get("height", 0)),
            ))
        return out

    # ---------------------------------------------------------------- download
    def download_image(self, url: str, dest: Path, min_side: int = 256) -> Optional[Path]:
        try:
            resp = self._get(url)
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as err:  # network error, 404, not an image...
            print(f"[wiki] skip {url[:80]}: {err}")
            return None
        if min(img.size) < min_side:
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest, format="JPEG", quality=92)
        return dest

    # ------------------------------------------------------------------- http
    def _get(self, url: str, params: Optional[Dict[str, str]] = None, retries: int = 2) -> httpx.Response:
        for attempt in range(retries + 1):
            resp = self.http.get(url, params=params)
            if resp.status_code in (429, 500, 502, 503) and attempt < retries:
                time.sleep(2.0 * (attempt + 1))
                continue
            if resp.status_code == 403 and not self._warned:
                self._warned = True
                print("[wiki] HTTP 403 from Wikimedia: their robot policy requires contact info in the "
                      "User-Agent. Set WIKI_CONTACT (email or URL) in .env / Kaggle Secrets.")
            resp.raise_for_status()
            return resp
        raise RuntimeError("unreachable")

    def _get_json(self, url: str, params: Dict[str, str]) -> dict:
        try:
            return self._get(url, params).json()
        except Exception as err:
            print(f"[wiki] request failed ({url}): {err}")
            return {}
