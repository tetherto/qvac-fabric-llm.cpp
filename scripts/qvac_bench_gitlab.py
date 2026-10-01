#!/usr/bin/env python3
"""Pull qvac-bench results from GitLab CI job artifacts into the dashboards DB.

The benchmark pipeline (.gitlab-ci.yml on alf/benchmark-scripts of
gitlab.collabora.com/ml/llama.cpp) uploads bench-workdir/results as the
artifact of every benchmark job. This script
lists the newest pipelines of a ref, downloads the archive of each finished
job, extracts the results directory and pushes it through qvac_bench_db with
the job URL as run_url:

    GITLAB_TOKEN=<read_api token> DATABASE_URL=postgres://... \\
        qvac_bench_gitlab.py [--ref REF] [--limit N] [--dry-run]

Jobs already pushed, or whose artifact expired, are recorded in a state file
and skipped next time; upserts are idempotent, so --force is always safe. Only
the stdlib is used here; psycopg comes in through qvac_bench_db.
"""

import argparse
import datetime
import json
import logging
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from qvac_bench_db import DbPreflightError, collect_run_rows, load_json, make_pusher

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PREFIX = "bench-workdir/results/"
FINISHED = ("success", "failed")
LIST_ERRORS = (urllib.error.URLError, OSError, ValueError)
JOB_ERRORS = LIST_ERRORS + (zipfile.BadZipFile,)


def log(label: str, msg: str) -> None:
    logger.info("[%s] %s", label, msg)


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class GitLab:
    """Minimal GitLab REST v4 client scoped to one project."""

    def __init__(self, url: str, project: str, token: "str | None") -> None:
        self.base = f"{url.rstrip('/')}/api/v4/projects/{urllib.parse.quote(project, safe='')}"
        self.token = token

    def request(self, path: str, params: "dict | None" = None) -> urllib.request.Request:
        url = f"{self.base}/{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url)
        if self.token:
            req.add_header("PRIVATE-TOKEN", self.token)
        return req

    def iter_list(self, path: str, params: "dict | None" = None):
        """Items of a list endpoint, following X-Next-Page."""
        params = dict(params or {}, per_page=100)
        page = 1
        while page:
            params["page"] = page
            with urllib.request.urlopen(self.request(path, params), timeout=60) as resp:
                items = json.load(resp)
                page = int(resp.headers.get("X-Next-Page") or 0)
            yield from items

    def download(self, path: str, target: Path) -> None:
        with urllib.request.urlopen(self.request(path), timeout=600) as resp:
            with target.open("wb") as f:
                shutil.copyfileobj(resp, f)


def has_archive(job: dict) -> bool:
    return any(a.get("file_type") == "archive" for a in job.get("artifacts") or [])


def extract_results(archive: Path, dest: Path) -> int:
    """Copy the bench-workdir/results/ entries of a job archive into dest; returns the file count."""
    root = dest.resolve()
    count = 0
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = info.filename[2:] if info.filename.startswith("./") else info.filename
            if info.is_dir() or not name.startswith(RESULTS_PREFIX):
                continue
            target = (root / name[len(RESULTS_PREFIX):]).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"unsafe path in archive: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    return count


def stems(results_dir: Path) -> list:
    return [p.with_name(p.name[:-len(".status")]) for p in sorted(results_dir.glob("*.status"))]


def job_record(job: dict, pipeline_id: int) -> dict:
    return {
        "pipeline": pipeline_id,
        "name": job["name"],
        "web_url": job["web_url"],
        "finished_at": job.get("finished_at"),
    }


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def process_job(gl: GitLab, job: dict, pipeline_id: int, pusher, workdir: Path) -> dict:
    """Download, extract and push one job's artifact; returns its state record.

    pusher=None is a dry run: rows are counted, nothing is pushed. Download and
    extraction errors propagate; push errors are counted in the record.
    """
    label = f"job {job['id']} ({job['name']})"
    record = job_record(job, pipeline_id)
    archive = workdir / "artifacts.zip"
    try:
        gl.download(f"jobs/{job['id']}/artifacts", archive)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        log("gitlab", f"{label}: artifact expired (404)")
        return record | {"expired": True, "seen_at": now()}
    results = workdir / "results"
    results.mkdir(exist_ok=True)
    files = extract_results(archive, results)
    archive.unlink()
    if not files:
        log("gitlab", f"{label}: no results in archive, skipping")
        return record | {"files": 0, "seen_at": now()}
    found = stems(results)
    if pusher is None:
        for stem in found:
            log("dry-run", f"{stem.name}: {len(collect_run_rows(stem, job['web_url']))} row(s)")
        log("dry-run", f"{label}: {files} file(s), {len(found)} run(s) from {job['web_url']}")
        return record | {"files": files, "runs": len(found)}
    failures = pusher.push_results_dir(results, run_url=job["web_url"])
    log("gitlab", f"{label}: {len(found)} run(s), {failures} push failure(s)")
    return record | {"files": files, "runs": len(found), "failures": failures, "pushed_at": now()}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Push the bench-workdir/results artifacts of GitLab CI benchmark "
        "jobs to the dashboards database (idempotent; safe to re-run).",
        epilog="GITLAB_TOKEN must hold a token with read_api scope for private projects.",
    )
    p.add_argument("--gitlab-url", default=os.environ.get("GITLAB_URL", "https://gitlab.collabora.com"))
    p.add_argument("--project", default="ml/llama.cpp", help="project path (default: %(default)s)")
    p.add_argument("--ref", default="alf/benchmark-scripts", help="branch whose pipelines to scan (default: %(default)s)")
    p.add_argument("--stage", default="benchmark", help="only jobs of this pipeline stage (default: %(default)s)")
    p.add_argument("--limit", type=int, default=10, help="scan only the newest N pipelines, 0 = all (default: %(default)s)")
    p.add_argument(
        "--state",
        type=Path,
        default=REPO_ROOT / "bench-workdir" / "gitlab-pushed.json",
        help="JSON file of jobs already handled (default: bench-workdir/gitlab-pushed.json)",
    )
    p.add_argument("--force", action="store_true", help="re-push jobs recorded in the state file")
    p.add_argument("--dry-run", action="store_true", help="download and count rows, push nothing, keep the state file as is")
    p.add_argument("--keep", type=Path, metavar="DIR", help="extract into DIR/<job_id>/results/ instead of a temp dir")
    p.add_argument(
        "--db-url",
        default=os.environ.get("DATABASE_URL"),
        help="Postgres/TimescaleDB URL (default: $DATABASE_URL)",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    token = os.environ.get("GITLAB_TOKEN")
    if not token:
        log("gitlab", "GITLAB_TOKEN is not set; only public projects are reachable")
    pusher = None
    if not args.dry_run:
        if not args.db_url:
            sys.exit("db: no database URL (pass --db-url or set $DATABASE_URL, or use --dry-run)")
        try:
            pusher = make_pusher(args.db_url)
            pusher.ping()
        except DbPreflightError as e:
            sys.exit(f"db: {e}")

    gl = GitLab(args.gitlab_url, args.project, token)
    state = load_json(args.state) or {}

    pipelines = []
    try:
        for pipeline in gl.iter_list("pipelines", {"ref": args.ref}):
            pipelines.append(pipeline)
            if args.limit > 0 and len(pipelines) >= args.limit:
                break
    except LIST_ERRORS as e:
        sys.exit(f"gitlab: cannot list pipelines of {args.project}@{args.ref}: {e}")
    log("gitlab", f"{len(pipelines)} pipeline(s) on {args.ref}")

    failures = 0
    for pipeline in pipelines:
        try:
            jobs = list(gl.iter_list(f"pipelines/{pipeline['id']}/jobs"))
        except LIST_ERRORS as e:
            log("gitlab", f"pipeline {pipeline['id']}: cannot list jobs: {e}")
            failures += 1
            continue
        for job in jobs:
            if job.get("stage") != args.stage or job["status"] not in FINISHED:
                continue
            key = str(job["id"])
            if key in state and not args.force:
                continue
            label = f"job {job['id']} ({job['name']})"
            if not has_archive(job):
                log("gitlab", f"{label}: no archive artifact (expired?)")
                record = job_record(job, pipeline["id"]) | {"expired": True, "seen_at": now()}
            else:
                try:
                    if args.keep:
                        workdir = args.keep / key
                        workdir.mkdir(parents=True, exist_ok=True)
                        record = process_job(gl, job, pipeline["id"], pusher, workdir)
                    else:
                        with tempfile.TemporaryDirectory(prefix="qvac-bench-gitlab-") as tmp:
                            record = process_job(gl, job, pipeline["id"], pusher, Path(tmp))
                except JOB_ERRORS as e:
                    log("gitlab", f"{label}: {e}")
                    failures += 1
                    continue
            if record.get("failures"):
                # Left out of the state so the next run retries the push.
                failures += 1
                continue
            if not args.dry_run:
                state[key] = record
                save_state(args.state, state)
    if failures:
        sys.exit(f"gitlab: {failures} job(s) failed; they are retried on the next run")


if __name__ == "__main__":
    main()
