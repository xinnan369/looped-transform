#!/usr/bin/env python3
"""Stage CI wheels from the single locked mirror before an offline install."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

MIRROR = "mirrors.aliyun.com"
MAX_MATRIX_BYTES = 1_000_000_000
LOCK = Path(__file__).resolve().parents[1] / "configs" / "ci_cpu_wheels.json"


def curl(url, options):
    return subprocess.run(
        ["curl", "--disable", "--noproxy", "*", "-4", "--proto", "=https", "--fail", "--silent",
         "--show-error", "--connect-timeout", "15", *options, url],
        stdout=subprocess.PIPE, text=True, check=True,
    ).stdout.strip()


def verify(wheel, path):
    if path.stat().st_size != wheel["bytes"]:
        raise ValueError(f"wrong byte count for {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if wheel["sha256"] and actual != wheel["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {path.name}")
    return actual


def download(wheel, directory):
    url = urlsplit(wheel["url"])
    file = unquote(url.path.rsplit("/", 1)[-1])
    if url.scheme != "https" or url.netloc != MIRROR or file != wheel["file"] or not file.endswith(".whl"):
        raise ValueError(f"invalid mirror wheel: {wheel['url']}")

    path = directory / file
    if path.exists():
        print(f"Reuse {file} {wheel['bytes']} bytes sha256={verify(wheel, path)}", flush=True)
        return

    probe = curl(wheel["url"], [
        "--max-time", "30", "--range", "0-0", "--max-filesize", "1", "--output", "/dev/null",
        "--dump-header", "-", "--write-out", "%{http_code}\n%{remote_ip}",
    ])
    *headers, code, remote_ip = probe.splitlines()
    expected_range = f"content-range: bytes 0-0/{wheel['bytes']}"
    if code != "206" or expected_range not in [line.lower() for line in headers]:
        raise ValueError(f"unexpected probe response for {file}: {probe}")
    routes = json.loads(subprocess.check_output(["ip", "-j", "route", "get", remote_ip], text=True))
    device = routes[0].get("dev", "")
    if not re.fullmatch(r"(?:eth|en)[a-z0-9]+", device):
        raise ValueError(f"could not verify direct runner interface for {remote_ip}: {device}")

    print(f"Fetch {file} {wheel['bytes']} bytes via {MIRROR} {remote_ip} {device}", flush=True)
    partial = path.with_suffix(".part")
    progress = ["--no-silent", "--progress-bar"] if wheel["bytes"] >= 32 * 1024**2 else []
    status = curl(wheel["url"], [
        *progress,
        "--max-time", "900", "--resolve", f"{MIRROR}:443:{remote_ip}", "--max-filesize", str(wheel["bytes"]),
        "--output", str(partial), "--write-out", "%{http_code}",
    ])
    if status != "200":
        raise ValueError(f"mirror download returned HTTP {status} for {file}")
    digest = verify(wheel, partial)
    partial.replace(path)
    print(f"Verified {file} {wheel['bytes']} bytes sha256={digest} {datetime.now(timezone.utc).isoformat()}", flush=True)


def main():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    planned_bytes = sum(wheel["bytes"] + 1 for wheels in lock.values() for wheel in wheels)
    if planned_bytes > MAX_MATRIX_BYTES:
        raise ValueError(f"CI matrix wheel transfer exceeds its 1 GB limit: {planned_bytes}")
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Locked CI matrix transfer limit {planned_bytes} bytes", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda wheel: download(wheel, directory), lock[tag]))


if __name__ == "__main__":
    main()
