"""CPU-only plots and a retryable GitHub PNG publisher, independent of training."""
import json
import math
import os
from pathlib import Path
import re
import subprocess
import urllib.error
import urllib.request


def read_committed(path, stage):
    rows = {}
    if not Path(path).exists():
        return []
    with Path(path).open() as handle:
        for line in handle:
            if not line.endswith("\n"):
                break  # An incomplete concurrent append is never committed.
            record = json.loads(line)
            step = record.get("global_step")
            if not isinstance(step, int) or step <= 0:
                continue
            if stage == "dimo" and record.get("committed") is not True:
                continue
            keys = ("global_step", "committed", "objective", "sample_mean_ce", "grad_norm_pre_clip",
                    "loss_dimo", "loss_aux", "distribution_energy", "student_grad_norm", "aux_grad_norm", "dimo_gradient_norm")
            rows[step] = {key: record[key] for key in keys if key in record}
    return [rows[k] for k in sorted(rows)]


def _series(rows, key):
    points = [(r["global_step"], r.get(key)) for r in rows]
    return [(step, float(value)) for step, value in points if value is not None and math.isfinite(float(value))]


def plot_curves(rows, output, *, stage, step, kind, window=50):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = (("sample_mean_ce",) if stage == "e3" else ("loss_dimo", "loss_aux", "distribution_energy")) if kind == "loss" else (
        ("grad_norm_pre_clip",) if stage == "e3" else ("student_grad_norm", "aux_grad_norm", "dimo_gradient_norm"))
    selected = [row for row in rows if row["global_step"] <= step]
    figure, axes = plt.subplots(len(keys), 1, figsize=(9, 3*len(keys)), squeeze=False)
    objectives = {row.get("objective", "squared") for row in selected} if stage == "dimo" else {"CE"}
    if len(objectives) > 1:
        raise RuntimeError("MONITOR_CANNOT_MIX_OBJECTIVES_IN_ONE_RUN")
    objective = next(iter(objectives))
    for axis, key in zip(axes[:, 0], keys):
        points = _series(selected, key)
        axis.set_title(f"{key} ({objective} objective)" if key == "loss_dimo" else key)
        if points:
            x, y = zip(*points)
            axis.plot(x, y, alpha=.65, linewidth=.7, label="measured")
            if kind == "loss":
                smooth = [float(np.mean(y[max(0, i-window+1):i+1])) for i in range(len(y))]
                axis.plot(x, smooth, label=f"moving mean ({window} measured points)")
            axis.legend()
        else:
            axis.text(.5, .5, "NOT_COLLECTED", transform=axis.transAxes, ha="center")
        axis.set_xlabel("committed optimizer step")
        axis.grid(alpha=.2)
    figure.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=110)
    plt.close(figure)


class GitHubPNGPublisher:
    def __init__(self, repository, tag, sha, *, timeout=20):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("invalid repository")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", tag) or not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError("invalid tag or fixed code SHA")
        self.repository, self.tag, self.sha, self.timeout = repository, tag, sha, timeout
        self.token = os.environ.get("GH_TOKEN")
        self.release = None

    def request(self, url, *, method="GET", data=None, content_type="application/json"):
        if not self.token:
            raise PermissionError("MONITOR_GITHUB_AUTH_UNAVAILABLE")
        request = urllib.request.Request(url, data=data, method=method,
            headers={"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
                     "Content-Type": content_type, "User-Agent": "editMGT-curves-monitor"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())

    def publish(self, paths):
        names = [Path(p).name for p in paths]
        if (len(paths) != 2 or not re.fullmatch(r"loss_step_\d{6}\.png", names[0])
                or not re.fullmatch(r"gradient_step_\d{6}\.png", names[1]) or names[0][10:] != names[1][14:]):
            raise ValueError("MONITOR_ASSET_WHITELIST_ONLY_MATCHING_LOSS_AND_GRADIENT_PNG")
        if any(Path(p).read_bytes()[:8] != b"\x89PNG\r\n\x1a\n" for p in paths):
            raise ValueError("MONITOR_ASSET_NOT_PNG")
        if not self.token:
            # gh handles its stored authentication; no secrets are captured.
            found = subprocess.run(["gh", "release", "view", self.tag, "--repo", self.repository],
                timeout=self.timeout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if found.returncode:
                subprocess.run(["gh", "release", "create", self.tag, "--repo", self.repository,
                    "--target", self.sha, "--prerelease", "--title", self.tag, "--notes", "Loss and gradient curves only."],
                    check=True, timeout=self.timeout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            metadata = subprocess.run(["gh", "api", f"repos/{self.repository}/releases/tags/{self.tag}"],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=self.timeout)
            release = json.loads(metadata.stdout)
            if release.get("target_commitish") != self.sha:
                raise RuntimeError("MONITOR_RELEASE_CODE_IDENTITY_MISMATCH")
            response = subprocess.run(["gh", "api", "--paginate", "--slurp",
                f"repos/{self.repository}/releases/{release['id']}/assets?per_page=100"],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=self.timeout)
            existing = {asset["name"] for page in json.loads(response.stdout) for asset in page}
            # Assets are immutable; API path handles partial-upload idempotence.
            for path in paths:
                if Path(path).name not in existing:
                    subprocess.run(["gh", "release", "upload", self.tag, str(path), "--repo", self.repository],
                        check=True, timeout=self.timeout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        base = f"https://api.github.com/repos/{self.repository}"
        try:
            self.release = self.request(f"{base}/releases/tags/{self.tag}")
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            self.release = self.request(f"{base}/releases", method="POST", data=json.dumps({
                "tag_name": self.tag, "target_commitish": self.sha, "name": self.tag,
                "prerelease": True, "body": "Loss and gradient curves only."}).encode())
        if self.release.get("target_commitish") not in (self.sha,):
            raise RuntimeError("MONITOR_RELEASE_CODE_IDENTITY_MISMATCH")
        existing = set()
        page = 1
        while True:
            assets = self.request(f"{base}/releases/{self.release['id']}/assets?per_page=100&page={page}")
            existing.update(a["name"] for a in assets)
            if len(assets) < 100:
                break
            page += 1
        for path in paths:
            if Path(path).name in existing:
                continue
            url = self.release["upload_url"].split("{")[0] + "?name=" + Path(path).name
            self.request(url, method="POST", data=Path(path).read_bytes(), content_type="image/png")


class Monitor:
    def __init__(self, log, output, *, stage, publisher=None, max_pending=128):
        self.log, self.output, self.stage = Path(log), Path(output), stage
        self.publisher, self.max_pending = publisher, max_pending
        self.output.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output / "upload_cursor.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"uploaded": [], "attempts": {}, "pending": []}

    def poll(self, retry_limit=3):
        rows = read_committed(self.log, self.stage)
        milestones = [r["global_step"] for r in rows if r["global_step"] % 500 == 0]
        pending = sorted(set(self.state["pending"]) | (set(milestones) - set(self.state["uploaded"])))
        # Keep most recent pending pairs. Older plots can be regenerated from
        # authoritative local scalars after a long outage.
        pending = pending[-self.max_pending:]
        self.state["pending"] = pending
        keep = {f"{kind}_step_{s:06d}.png" for s in pending for kind in ("loss", "gradient")}
        for path in self.output.glob("*_step_*.png"):
            if path.name not in keep:
                path.unlink()  # Only owned, reproducible monitor PNG cache.
        for step in pending[:]:
            paths = [self.output / f"{kind}_step_{step:06d}.png" for kind in ("loss", "gradient")]
            for kind, path in zip(("loss", "gradient"), paths):
                if not path.exists():
                    plot_curves(rows, path, stage=self.stage, step=step, kind=kind)
            attempts = self.state["attempts"].get(str(step), 0)
            if self.publisher is None or attempts >= retry_limit:
                continue
            try:
                self.publisher.publish(paths)
                self.state["uploaded"].append(step)
                self.state["pending"].remove(step)
            except Exception as exc:
                # Record class, never request headers, tokens or private URLs.
                self.state["attempts"][str(step)] = attempts + 1
                self.state["last_error_class"] = type(exc).__name__
        # Limit bytes as well as pair count. Only these owned reproducible PNGs
        # can be evicted; logs and model artifacts are never touched.
        cached = sorted(self.output.glob("*_step_*.png"), key=lambda p: p.name)
        cache_bytes = sum(p.stat().st_size for p in cached)
        for old_step in sorted(pending):
            if cache_bytes <= 256*1024**2:
                break
            for kind in ("loss", "gradient"):
                path = self.output / f"{kind}_step_{old_step:06d}.png"
                if path.exists():
                    cache_bytes -= path.stat().st_size
                    path.unlink()
        temporary = self.state_path.with_suffix(".partial")
        temporary.write_text(json.dumps(self.state, sort_keys=True) + "\n")
        os.replace(temporary, self.state_path)
        return self.state
