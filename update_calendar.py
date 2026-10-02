#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VK Medveščak (škola plivanja i vaterpola) -> javni iCalendar feed za Svetice.

Namjena:
- pokreće se jednom dnevno u GitHub Actions
- dohvaća službenu naslovnicu VK Medveščak
- nalazi aktualni tjedni raspored i sliku rasporeda
- OCR-om nalazi red "ŠKOLA PLIVANJA I VATERPOLA"
- uzima SAMO termine koji u ćeliji sadrže "SVETICE"
- generira medvescak-svetice.ics
- ako OCR nije dovoljno pouzdan, završava greškom i NE prepisuje postojeći .ics

Važno: feed ne sadrži ime djeteta; objavljuje samo javni raspored kluba.
"""

from __future__ import annotations

import io
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import cv2
import numpy as np
import pytesseract
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageEnhance, ImageOps

HOME_URL = "https://www.vk-medvescak.hr/"
OUT = Path(__file__).resolve().parent / "medvescak-svetice.ics"
TZID = "Europe/Zagreb"
EVENT_TITLE = "VK Medveščak – Svetice (škola)"
CAL_NAME = "VK Medveščak – Svetice"

DAYS = ["PON", "UTO", "SRI", "ČET", "PET", "SUB", "NED"]
ALIASES = {
    "PON": ("PON",),
    "UTO": ("UTO",),
    "SRI": ("SRI", "SRIJ"),
    "ČET": ("ČET", "CET"),
    "PET": ("PET",),
    "SUB": ("SUB",),
    "NED": ("NED",),
}

UA = {"User-Agent": "Mozilla/5.0 MedvescakCalendar/2.0"}


@dataclass(frozen=True)
class Session:
    day_offset: int
    start: str
    end: str
    raw: str


def ascii_norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.upper())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def fetch_html() -> str:
    r = requests.get(HOME_URL, timeout=30, headers=UA)
    r.raise_for_status()
    return r.text


def parse_week_start_and_image(html: str) -> Tuple[datetime, str]:
    soup = BeautifulSoup(html, "html.parser")
    visible = soup.get_text(" ", strip=True)

    # npr. "Datum: 28.9.-4.10. 2026."
    m = re.search(
        r"Datum:\s*(\d{1,2})\.(\d{1,2})\.\s*-\s*(\d{1,2})\.(\d{1,2})\.\s*(\d{4})\.",
        visible,
        re.I,
    )
    if not m:
        raise RuntimeError("Nije pronađen datum aktualnog rasporeda.")

    d1, mo1, d2, mo2, year = map(int, m.groups())
    week_start = datetime(year, mo1, d1)

    candidates = []
    for img in soup.find_all("img"):
        src = img.get("data-src") or img.get("data-lazy-src") or img.get("src")
        if not src:
            continue
        blob = " ".join([
            img.get("alt") or "",
            img.get("title") or "",
            src,
        ])
        n = ascii_norm(blob)
        score = 0
        if "RASPORED" in n:
            score += 10
        if "PAGE0001" in n:
            score += 3
        if "UPLOADS" in n:
            score += 1
        if score:
            candidates.append((score, urljoin(HOME_URL, src)))

    if not candidates:
        raise RuntimeError("Nije pronađena slika rasporeda.")

    candidates.sort(key=lambda x: x[0], reverse=True)
    return week_start, candidates[0][1]


def fetch_image(url: str) -> Image.Image:
    r = requests.get(url, timeout=30, headers=UA)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def prep(img: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(img)
    gray = ImageEnhance.Contrast(gray).enhance(2.0)
    # dovoljno veliko za pouzdaniji OCR
    w, h = gray.size
    if w < 2400:
        scale = 2400 / w
        gray = gray.resize((int(w * scale), int(h * scale)))
    return gray


def tsv(img: Image.Image):
    return pytesseract.image_to_data(
        img,
        lang="hrv+eng",
        config="--psm 6",
        output_type=pytesseract.Output.DICT,
    )


def word_center(data, i: int) -> Tuple[int, int]:
    x = int(data["left"][i]) + int(data["width"][i]) // 2
    y = int(data["top"][i]) + int(data["height"][i]) // 2
    return x, y


def find_day_centers(data) -> Dict[str, int]:
    hits: Dict[str, Tuple[int, int]] = {}
    for i, text in enumerate(data["text"]):
        n = ascii_norm(text or "")
        if not n:
            continue
        x, y = word_center(data, i)
        for day, aliases in ALIASES.items():
            if any(n == ascii_norm(a) for a in aliases):
                # prvi/gornji uvjerljiv header
                if day not in hits or y < hits[day][1]:
                    hits[day] = (x, y)

    if len(hits) < 5:
        raise RuntimeError(f"Nedovoljno prepoznatih zaglavlja dana: {sorted(hits)}")
    return {d: xy[0] for d, xy in hits.items()}


def find_school_row_y(data) -> int:
    tokens = []
    for i, text in enumerate(data["text"]):
        n = ascii_norm(text or "")
        if not n:
            continue
        x, y = word_center(data, i)
        if (
            "SKOLA" in n
            or "PLIVAN" in n
            or "VATERPOL" in n
            or n in {"2016", "MLADI"}
        ):
            tokens.append((x, y, n))

    # grupiranje po y području
    if not tokens:
        raise RuntimeError("Nije pronađen red škole plivanja i vaterpola.")

    buckets: Dict[int, List[str]] = {}
    ys: Dict[int, List[int]] = {}
    for x, y, n in tokens:
        b = round(y / 35) * 35
        buckets.setdefault(b, []).append(n)
        ys.setdefault(b, []).append(y)

    scored = []
    for b, vals in buckets.items():
        joined = " ".join(vals)
        score = 0
        if "SKOLA" in joined:
            score += 4
        if "PLIVAN" in joined:
            score += 3
        if "VATERPOL" in joined:
            score += 3
        if "2016" in joined:
            score += 1
        scored.append((score, int(np.median(ys[b])), joined))

    scored.sort(reverse=True)
    if not scored or scored[0][0] < 4:
        raise RuntimeError("Red škole nije prepoznat s dovoljnom sigurnošću.")
    return scored[0][1]


def horizontal_grid_lines(img: Image.Image) -> List[int]:
    arr = np.array(img)
    _, bw = cv2.threshold(arr, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, arr.shape[1] // 18), 1))
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(horiz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ys = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > arr.shape[1] * 0.25:
            ys.append(y + h // 2)
    ys = sorted(set(ys))

    # spoji linije udaljene tek nekoliko px
    merged = []
    for y in ys:
        if not merged or y - merged[-1] > 8:
            merged.append(y)
        else:
            merged[-1] = (merged[-1] + y) // 2
    return merged


def row_bounds(img: Image.Image, row_y: int) -> Tuple[int, int]:
    lines = horizontal_grid_lines(img)
    above = [y for y in lines if y < row_y]
    below = [y for y in lines if y > row_y]
    if above and below:
        y1, y2 = max(above), min(below)
        if y2 - y1 >= 35:
            return y1 + 2, y2 - 2

    # fallback
    h = img.height
    half = max(70, int(h * 0.055))
    return max(0, row_y - half), min(h, row_y + half)


def day_bounds(centers: Dict[str, int], width: int) -> Dict[str, Tuple[int, int]]:
    ordered = sorted(((d, centers[d]) for d in DAYS if d in centers), key=lambda x: x[1])
    xs = [x for _, x in ordered]
    result = {}
    for i, (d, x) in enumerate(ordered):
        left = 0 if i == 0 else (xs[i - 1] + x) // 2
        right = width if i == len(xs) - 1 else (x + xs[i + 1]) // 2
        result[d] = (max(0, left), min(width, right))
    return result


def parse_times(raw: str) -> Optional[Tuple[str, str]]:
    s = raw.replace("\n", " ")
    # najčešći oblici 18-19, 18.00-19.00, 18:00 – 19:00
    m = re.search(
        r"\b(\d{1,2})(?:[:.](\d{2}))?\s*[-–—]\s*"
        r"(\d{1,2})(?:[:.](\d{2}))?\b",
        s,
    )
    if not m:
        return None
    h1, m1, h2, m2 = m.groups()
    h1, h2 = int(h1), int(h2)
    m1, m2 = int(m1 or 0), int(m2 or 0)
    if not (0 <= h1 <= 23 and 0 <= h2 <= 23 and 0 <= m1 <= 59 and 0 <= m2 <= 59):
        return None
    return f"{h1:02d}:{m1:02d}", f"{h2:02d}:{m2:02d}"


def extract_sessions(img: Image.Image) -> List[Session]:
    work = prep(img)
    data = tsv(work)
    centers = find_day_centers(data)
    ry = find_school_row_y(data)
    y1, y2 = row_bounds(work, ry)
    bounds = day_bounds(centers, work.width)

    sessions: List[Session] = []
    for offset, day in enumerate(DAYS):
        if day not in bounds:
            continue
        x1, x2 = bounds[day]
        # mala margina da se ne pokupi tekst susjedne ćelije
        margin = max(3, int((x2 - x1) * 0.03))
        cell = work.crop((x1 + margin, y1, x2 - margin, y2))
        raw = pytesseract.image_to_string(cell, lang="hrv+eng", config="--psm 6").strip()
        n = ascii_norm(raw)

        # korisnik želi samo Svetice; Šalata se namjerno ignorira
        if "SVET" not in n:
            continue

        times = parse_times(raw)
        if not times:
            raise RuntimeError(f"Svetice su prepoznate za {day}, ali satnica nije: {raw!r}")
        start, end = times
        sessions.append(Session(offset, start, end, raw))

    if not sessions:
        raise RuntimeError(
            "Nije pronađen nijedan pouzdan termin škole na Sveticama. "
            "Postojeći kalendar ostaje netaknut."
        )
    return sessions


def ics_escape(s: str) -> str:
    return (
        s.replace("\\", "\\\\")
         .replace(";", "\\;")
         .replace(",", "\\,")
         .replace("\r\n", "\\n")
         .replace("\n", "\\n")
    )


def fold(line: str, limit: int = 73) -> str:
    # jednostavan RFC5545 folding, dovoljno za naše ASCII/UTF-8 kratke linije
    if len(line) <= limit:
        return line
    parts = [line[:limit]]
    line = line[limit:]
    while line:
        parts.append(" " + line[: limit - 1])
        line = line[limit - 1 :]
    return "\r\n".join(parts)


def make_ics(week_start: datetime, sessions: List[Session], source_url: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Medvescak Svetice Calendar//HR//",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        f"X-WR-CALNAME:{ics_escape(CAL_NAME)}",
        f"X-WR-TIMEZONE:{TZID}",
        "REFRESH-INTERVAL;VALUE=DURATION:PT6H",
        "X-PUBLISHED-TTL:PT6H",
    ]

    for s in sessions:
        date = week_start + timedelta(days=s.day_offset)
        ymd = date.strftime("%Y%m%d")
        start = s.start.replace(":", "") + "00"
        end = s.end.replace(":", "") + "00"
        uid = f"vk-medvescak-svetice-{ymd}@calendar.local"
        desc = f"Automatski preuzeto sa službenog rasporeda VK Medveščak. Izvor: {source_url}"
        lines.extend([
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"DTSTAMP:{now}",
            f"DTSTART;TZID={TZID}:{ymd}T{start}",
            f"DTEND;TZID={TZID}:{ymd}T{end}",
            f"SUMMARY:{ics_escape(EVENT_TITLE)}",
            "LOCATION:Bazeni Svetice, Zagreb",
            f"DESCRIPTION:{ics_escape(desc)}",
            f"URL:{source_url}",
            "STATUS:CONFIRMED",
            "TRANSP:OPAQUE",
            "END:VEVENT",
        ])

    lines.append("END:VCALENDAR")
    return "\r\n".join(fold(x) for x in lines) + "\r\n"


def main() -> int:
    html = fetch_html()
    week_start, image_url = parse_week_start_and_image(html)
    img = fetch_image(image_url)
    sessions = extract_sessions(img)
    content = make_ics(week_start, sessions, HOME_URL)

    # tek nakon uspješnog parsiranja prepisuje izlaz
    OUT.write_text(content, encoding="utf-8", newline="")
    print(f"Tjedan: {week_start.date()}")
    print(f"Slika: {image_url}")
    for s in sessions:
        d = week_start + timedelta(days=s.day_offset)
        print(f"{d.date()} {s.start}-{s.end} Svetice")
    print(f"Zapisano: {OUT}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"GREŠKA: {exc}", file=sys.stderr)
        raise
