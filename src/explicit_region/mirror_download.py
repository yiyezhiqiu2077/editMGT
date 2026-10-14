"""Pinned public Hugging Face downloads through a mirror, without env proxies."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import time

import requests


class MirrorDownloader:
    def __init__(self, endpoint="https://hf-mirror.com", reserve_bytes=0):
        if endpoint.rstrip("/") != "https://hf-mirror.com":
            raise ValueError("this experiment requires hf-mirror.com")
        self.endpoint = endpoint.rstrip("/")
        self.reserve_bytes = reserve_bytes

    @staticmethod
    def session():
        session = requests.Session()
        session.trust_env = False
        return session

    @staticmethod
    def validate_revision(revision):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("immutable commit required")

    def metadata(self, repo, revision, *, dataset=True):
        self.validate_revision(revision)
        kind = "datasets" if dataset else "models"
        with self.session() as session:
            response = session.get(f"{self.endpoint}/api/{kind}/{repo}/revision/{revision}",
                                   params={"blobs": "true"}, timeout=(20, 90))
            response.raise_for_status()
            value = response.json()
        if value.get("sha") != revision:
            raise RuntimeError("mirror resolved a different revision")
        return value

    def download(self, repo, revision, entry, root, *, dataset=True):
        self.validate_revision(revision)
        name = entry["rfilename"]
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe mirror path")
        path = Path(root) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        expected_size = entry.get("size")
        lfs_sha = entry.get("lfs", {}).get("sha256")
        blob_sha = entry.get("blobId")

        def verify(candidate):
            if expected_size is not None and candidate.stat().st_size != expected_size:
                raise RuntimeError(f"download size mismatch: {name}")
            digest = hashlib.sha256() if lfs_sha else hashlib.sha1()
            if not lfs_sha:
                digest.update(f"blob {candidate.stat().st_size}\0".encode())
            with candidate.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            expected = lfs_sha or blob_sha
            if expected and digest.hexdigest() != expected:
                raise RuntimeError(f"download content hash mismatch: {name}")
            return path

        if path.exists():
            return verify(path)
        temporary = path.with_name(path.name + ".partial")
        prefix = "datasets/" if dataset else ""
        url = f"{self.endpoint}/{prefix}{repo}/resolve/{revision}/{name}"
        for attempt in range(4):
            try:
                offset = temporary.stat().st_size if temporary.exists() else 0
                if shutil.disk_usage(path.parent).free < self.reserve_bytes + max(0, (expected_size or 0) - offset):
                    raise RuntimeError("download would consume reserved training disk space")
                with self.session() as session:
                    response = session.get(url, headers={"Range": f"bytes={offset}-"} if offset else {},
                                           stream=True, timeout=(30, 120))
                    response.raise_for_status()
                    if offset and response.status_code == 206:
                        if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                            raise RuntimeError("incorrect mirror range response")
                        mode = "ab"
                    else:
                        mode = "wb"
                    with temporary.open(mode) as handle:
                        for block in response.iter_content(1024 * 1024):
                            if block:
                                handle.write(block)
                verify(temporary)
                temporary.replace(path)
                return path
            except (requests.RequestException, OSError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        raise AssertionError("unreachable")

    def many(self, repo, revision, entries, root, *, dataset=True, workers=4):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda entry: self.download(repo, revision, entry, root,
                                                            dataset=dataset), entries))
