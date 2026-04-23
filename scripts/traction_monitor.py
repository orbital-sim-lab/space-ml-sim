#!/usr/bin/env python3
"""Public traction monitor for space-ml-sim.

Pulls signals from PyPI, GitHub, Hacker News, and Reddit, then produces a
short markdown report with week-over-week deltas and actionable
recommendations.

Usage:
    python scripts/traction_monitor.py                 # write report to reports/traction/
    python scripts/traction_monitor.py --print         # print report to stdout
    python scripts/traction_monitor.py --issue         # write a body.md for use with gh issue create

Requires only the Python standard library. For richer GitHub data (traffic,
clones, referrers), set the GITHUB_TOKEN environment variable with `repo`
scope on the target repository. Falls back to public-only signals
otherwise.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PACKAGE_NAME = "space-ml-sim"
GITHUB_OWNER = "orbital-sim-lab"
GITHUB_REPO = "space-ml-sim"
SEARCH_TERMS = ["space-ml-sim", "orbital-sim-lab"]

# Reports are written outside the public repo by default to keep strategy
# notes, referrer data, and recommendation text private. Override with
# --output-dir or the SPACE_ML_SIM_TRACTION_DIR env var.
DEFAULT_REPORT_DIR = Path(
    os.environ.get("SPACE_ML_SIM_TRACTION_DIR")
    or (Path.home() / ".space-ml-sim" / "traction")
)
REQUEST_TIMEOUT_SEC = 15

USER_AGENT = "space-ml-sim-traction-monitor/1.0 (+https://github.com/orbital-sim-lab/space-ml-sim)"


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only)
# ---------------------------------------------------------------------------


def http_get_json(url: str, headers: dict[str, str] | None = None) -> Any:
    """GET a URL and parse the response as JSON. Returns None on failure."""
    req_headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SEC) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body)
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError) as exc:
        print(f"[warn] GET {url} failed: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TractionSnapshot:
    """Immutable record of traction signals at a point in time."""

    generated_at: str
    pypi_downloads_last_day: int | None = None
    pypi_downloads_last_week: int | None = None
    pypi_downloads_last_month: int | None = None
    pypi_real_downloads_last_month: int | None = None
    pypi_bot_share_pct: float | None = None
    github_stars: int | None = None
    github_forks: int | None = None
    github_watchers: int | None = None
    github_open_issues: int | None = None
    github_views_14d: int | None = None
    github_uniques_14d: int | None = None
    github_clones_14d: int | None = None
    github_top_referrers: list[dict[str, Any]] = field(default_factory=list)
    github_top_paths: list[dict[str, Any]] = field(default_factory=list)
    hn_mentions: list[dict[str, Any]] = field(default_factory=list)
    reddit_mentions: list[dict[str, Any]] = field(default_factory=list)
    code_references: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------


def collect_pypi(snapshot: TractionSnapshot) -> None:
    """Pull recent download counts from pypistats.org.

    Also computes a "real downloads" estimate by excluding the `null`
    system category, which captures mirror syncs, SBOM scanners, and
    other bots that don't report a Python runtime fingerprint.
    """
    data = http_get_json(f"https://pypistats.org/api/packages/{PACKAGE_NAME}/recent")
    if data and "data" in data:
        d = data["data"]
        snapshot.pypi_downloads_last_day = d.get("last_day")
        snapshot.pypi_downloads_last_week = d.get("last_week")
        snapshot.pypi_downloads_last_month = d.get("last_month")

    # Segment real vs bot by looking at the system breakdown.
    system_data = http_get_json(
        f"https://pypistats.org/api/packages/{PACKAGE_NAME}/system"
    )
    if system_data and isinstance(system_data.get("data"), list):
        real_total = 0
        bot_total = 0
        for row in system_data["data"]:
            cat = row.get("category")
            dl = row.get("downloads", 0)
            if cat in (None, "null"):
                bot_total += dl
            else:
                real_total += dl
        total = real_total + bot_total
        snapshot.pypi_real_downloads_last_month = real_total
        snapshot.pypi_bot_share_pct = (
            round(100.0 * bot_total / total, 1) if total else None
        )


def collect_github_public(snapshot: TractionSnapshot) -> None:
    """Public GitHub repo metadata — no auth required."""
    data = http_get_json(f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}")
    if data:
        snapshot.github_stars = data.get("stargazers_count")
        snapshot.github_forks = data.get("forks_count")
        snapshot.github_watchers = data.get("subscribers_count")
        snapshot.github_open_issues = data.get("open_issues_count")


def collect_github_traffic(snapshot: TractionSnapshot, token: str) -> None:
    """Traffic endpoints require push access — feed via GITHUB_TOKEN."""
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    base = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}"

    views = http_get_json(f"{base}/traffic/views", headers=headers)
    if views:
        snapshot.github_views_14d = views.get("count")
        snapshot.github_uniques_14d = views.get("uniques")

    clones = http_get_json(f"{base}/traffic/clones", headers=headers)
    if clones:
        snapshot.github_clones_14d = clones.get("count")

    referrers = http_get_json(f"{base}/traffic/popular/referrers", headers=headers)
    if isinstance(referrers, list):
        snapshot.github_top_referrers = referrers[:10]

    paths = http_get_json(f"{base}/traffic/popular/paths", headers=headers)
    if isinstance(paths, list):
        snapshot.github_top_paths = paths[:10]


def collect_hn_mentions(snapshot: TractionSnapshot) -> None:
    """Hacker News search (Algolia) — last 30 days."""
    results: list[dict[str, Any]] = []
    since_ts = int((dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=30)).timestamp())
    for term in SEARCH_TERMS:
        # Quote as a phrase so the hyphen is treated as part of the string.
        quoted = f'"{term}"'
        url = (
            f"http://hn.algolia.com/api/v1/search"
            f"?query={urllib.parse.quote(quoted)}"
            f"&numericFilters=created_at_i>{since_ts}"
            f"&hitsPerPage=20"
        )
        data = http_get_json(url)
        if not data:
            continue
        for hit in data.get("hits", []):
            results.append(
                {
                    "term": term,
                    "title": hit.get("title") or hit.get("story_title"),
                    "url": hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}",
                    "points": hit.get("points", 0),
                    "num_comments": hit.get("num_comments", 0),
                    "author": hit.get("author"),
                    "created_at": hit.get("created_at"),
                }
            )
    # Dedupe by URL
    seen = set()
    unique: list[dict[str, Any]] = []
    for item in results:
        key = item.get("url")
        if key and key not in seen:
            unique.append(item)
            seen.add(key)
    snapshot.hn_mentions = unique


def collect_reddit_mentions(snapshot: TractionSnapshot) -> None:
    """Reddit search JSON — unauthenticated public endpoint."""
    results: list[dict[str, Any]] = []
    for term in SEARCH_TERMS:
        # Reddit interprets `-` as a NOT operator when unquoted, so wrap the
        # term in quotes to search for the literal phrase.
        quoted = f'"{term}"'
        url = (
            f"https://www.reddit.com/search.json?q={urllib.parse.quote(quoted)}"
            f"&sort=new&t=month&limit=25"
        )
        data = http_get_json(url, headers={"User-Agent": USER_AGENT})
        if not data:
            continue
        for child in data.get("data", {}).get("children", []):
            d = child.get("data", {})
            results.append(
                {
                    "term": term,
                    "title": d.get("title"),
                    "subreddit": d.get("subreddit"),
                    "url": f"https://reddit.com{d.get('permalink', '')}",
                    "score": d.get("score", 0),
                    "num_comments": d.get("num_comments", 0),
                    "author": d.get("author"),
                    "created_utc": d.get("created_utc"),
                }
            )
    seen = set()
    unique: list[dict[str, Any]] = []
    for item in results:
        key = item.get("url")
        if key and key not in seen:
            unique.append(item)
            seen.add(key)
    snapshot.reddit_mentions = unique


def collect_code_references(snapshot: TractionSnapshot, token: str | None) -> None:
    """GitHub code search — who else is importing or mentioning the package?"""
    if not token:
        return  # code search requires auth
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    url = (
        "https://api.github.com/search/code?q="
        + urllib.parse.quote(f'"{PACKAGE_NAME}" in:file -repo:{GITHUB_OWNER}/{GITHUB_REPO}')
        + "&per_page=20"
    )
    data = http_get_json(url, headers=headers)
    if not data:
        return
    results: list[dict[str, Any]] = []
    for item in data.get("items", [])[:20]:
        repo = item.get("repository", {})
        results.append(
            {
                "repo_full_name": repo.get("full_name"),
                "repo_url": repo.get("html_url"),
                "file_path": item.get("path"),
                "file_url": item.get("html_url"),
            }
        )
    snapshot.code_references = results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def load_previous_snapshot() -> TractionSnapshot | None:
    """Load the most recent prior snapshot from disk for week-over-week delta."""
    if not DEFAULT_REPORT_DIR.exists():
        return None
    json_reports = sorted(DEFAULT_REPORT_DIR.glob("*.json"))
    if not json_reports:
        return None
    try:
        raw = json.loads(json_reports[-1].read_text())
        return TractionSnapshot(**raw)
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        print(f"[warn] could not parse previous snapshot: {exc}", file=sys.stderr)
        return None


def delta(current: int | None, previous: int | None) -> str:
    """Human-readable delta, e.g. '+42' or '—'."""
    if current is None:
        return "—"
    if previous is None:
        return f"{current}"
    diff = current - previous
    if diff == 0:
        return f"{current} (no change)"
    sign = "+" if diff > 0 else ""
    pct = f" ({sign}{(diff/previous)*100:.0f}%)" if previous else ""
    return f"{current} ({sign}{diff}{pct})"


def build_recommendations(current: TractionSnapshot, previous: TractionSnapshot | None) -> list[str]:
    """Heuristic suggestions based on signals."""
    recs: list[str] = []

    # New HN / Reddit mentions → engage fast
    if current.hn_mentions:
        recs.append(
            f"🔥 **{len(current.hn_mentions)} Hacker News mention(s)** detected. "
            "Reply within 24h; HN rewards fast, substantive author engagement."
        )
    if current.reddit_mentions:
        subs = {m.get("subreddit") for m in current.reddit_mentions if m.get("subreddit")}
        recs.append(
            f"💬 **{len(current.reddit_mentions)} Reddit thread(s)** across {len(subs)} subreddit(s). "
            "Reply with technical depth; pin useful links back to docs/notebooks."
        )

    # Bot-share warning — if inflated, don't celebrate headline numbers
    if current.pypi_bot_share_pct is not None and current.pypi_bot_share_pct > 70:
        recs.append(
            f"🤖 **{current.pypi_bot_share_pct}% of PyPI downloads are bots/mirrors.** "
            f"Real installs last month: ~{current.pypi_real_downloads_last_month}. "
            "Don't quote the inflated headline number when pitching."
        )

    # Download momentum
    prev_week = previous.pypi_downloads_last_week if previous else None
    curr_week = current.pypi_downloads_last_week or 0
    if prev_week is not None and prev_week > 0:
        change = (curr_week - prev_week) / prev_week
        if change >= 0.5:
            recs.append(
                f"📈 **Downloads surged {change*100:.0f}% WoW** ({prev_week} → {curr_week}). "
                "Identify source (see top referrers) and double down."
            )
        elif change <= -0.3 and curr_week < 20:
            recs.append(
                f"📉 **Downloads dropped {abs(change)*100:.0f}% WoW**. "
                "Time for a new touch — blog post, Reddit thread, or outreach email."
            )

    # Stars momentum
    prev_stars = previous.github_stars if previous else None
    curr_stars = current.github_stars or 0
    if prev_stars is not None and (curr_stars - prev_stars) >= 5:
        recs.append(
            f"⭐ **+{curr_stars - prev_stars} stars this cycle** ({prev_stars} → {curr_stars}). "
            "Thank new stargazers via Discussions welcome post; ask what drew them in."
        )

    # Quiet week → push something
    has_mentions = bool(current.hn_mentions or current.reddit_mentions)
    new_stars = (curr_stars - (prev_stars or 0)) if prev_stars is not None else 0
    if not has_mentions and new_stars == 0 and curr_week < 30:
        recs.append(
            "🪶 **Quiet week.** Ship a touch: submit to another awesome-list, post a "
            "notebook demo, or email one researcher you've been meaning to ping."
        )

    # Top referrer insight
    if current.github_top_referrers:
        top = current.github_top_referrers[0]
        recs.append(
            f"🧭 Top traffic source: **{top.get('referrer')}** "
            f"({top.get('count')} views, {top.get('uniques')} unique). "
            "Investigate — is there a post to respond to?"
        )

    # External code references
    if current.code_references:
        recs.append(
            f"🔗 **{len(current.code_references)} external repo(s) reference the package.** "
            "Check if anyone's built something public — reach out, offer a collab or demo slot."
        )

    # Traffic without downloads → conversion gap
    if (
        current.github_views_14d
        and current.github_views_14d > 100
        and (current.pypi_downloads_last_week or 0) < 10
    ):
        recs.append(
            "🧪 **High repo views but low PyPI installs** — conversion gap. "
            "Audit README: is the install command above the fold? Is the quickstart actually 10 lines?"
        )

    if not recs:
        recs.append("Steady state. Keep shipping.")

    return recs


def format_report(current: TractionSnapshot, previous: TractionSnapshot | None) -> str:
    """Render a markdown report."""
    lines: list[str] = []
    lines.append(f"# space-ml-sim — Traction Report ({current.generated_at[:10]})")
    lines.append("")
    lines.append("## Headline numbers")
    lines.append("")
    lines.append("| Metric | Now | Δ vs last report |")
    lines.append("|---|---|---|")
    lines.append(
        f"| PyPI downloads (last day) | {current.pypi_downloads_last_day} | "
        f"{delta(current.pypi_downloads_last_day, previous.pypi_downloads_last_day if previous else None)} |"
    )
    lines.append(
        f"| PyPI downloads (last week) | {current.pypi_downloads_last_week} | "
        f"{delta(current.pypi_downloads_last_week, previous.pypi_downloads_last_week if previous else None)} |"
    )
    lines.append(
        f"| PyPI downloads (last month) | {current.pypi_downloads_last_month} | "
        f"{delta(current.pypi_downloads_last_month, previous.pypi_downloads_last_month if previous else None)} |"
    )
    lines.append(
        f"| PyPI **real** downloads (last month, non-bot) | {current.pypi_real_downloads_last_month} | "
        f"{delta(current.pypi_real_downloads_last_month, previous.pypi_real_downloads_last_month if previous else None)} |"
    )
    lines.append(
        f"| PyPI bot share | {current.pypi_bot_share_pct}% | — |"
    )
    lines.append(
        f"| GitHub stars | {current.github_stars} | "
        f"{delta(current.github_stars, previous.github_stars if previous else None)} |"
    )
    lines.append(
        f"| GitHub forks | {current.github_forks} | "
        f"{delta(current.github_forks, previous.github_forks if previous else None)} |"
    )
    lines.append(
        f"| Repo views (14d) | {current.github_views_14d} | "
        f"{delta(current.github_views_14d, previous.github_views_14d if previous else None)} |"
    )
    lines.append(
        f"| Unique visitors (14d) | {current.github_uniques_14d} | "
        f"{delta(current.github_uniques_14d, previous.github_uniques_14d if previous else None)} |"
    )
    lines.append(
        f"| Clones (14d) | {current.github_clones_14d} | "
        f"{delta(current.github_clones_14d, previous.github_clones_14d if previous else None)} |"
    )
    lines.append("")

    # Recommendations first — most actionable content above the fold
    lines.append("## Recommendations")
    lines.append("")
    for rec in build_recommendations(current, previous):
        lines.append(f"- {rec}")
    lines.append("")

    # Mentions
    if current.hn_mentions:
        lines.append("## Hacker News mentions (last 30 days)")
        lines.append("")
        for m in current.hn_mentions:
            lines.append(
                f"- **{m['title']}** — {m['points']} points, {m['num_comments']} comments "
                f"by {m['author']} ([link]({m['url']}))"
            )
        lines.append("")

    if current.reddit_mentions:
        lines.append("## Reddit mentions (last month)")
        lines.append("")
        for m in current.reddit_mentions:
            lines.append(
                f"- **r/{m['subreddit']}** — {m['title']} "
                f"({m['score']} score, {m['num_comments']} comments) "
                f"by u/{m['author']} ([link]({m['url']}))"
            )
        lines.append("")

    if current.github_top_referrers:
        lines.append("## Top referrers (last 14 days)")
        lines.append("")
        lines.append("| Referrer | Views | Unique |")
        lines.append("|---|---|---|")
        for r in current.github_top_referrers:
            lines.append(f"| {r.get('referrer')} | {r.get('count')} | {r.get('uniques')} |")
        lines.append("")

    if current.github_top_paths:
        lines.append("## Top paths (last 14 days)")
        lines.append("")
        lines.append("| Path | Views | Unique |")
        lines.append("|---|---|---|")
        for p in current.github_top_paths:
            lines.append(f"| `{p.get('path')}` | {p.get('count')} | {p.get('uniques')} |")
        lines.append("")

    if current.code_references:
        lines.append("## External repositories referencing the package")
        lines.append("")
        for c in current.code_references:
            lines.append(
                f"- [{c['repo_full_name']}]({c['repo_url']}) — `{c['file_path']}` "
                f"([view]({c['file_url']}))"
            )
        lines.append("")

    lines.append("---")
    lines.append(f"_Generated at {current.generated_at} by `scripts/traction_monitor.py`._")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def collect_all() -> TractionSnapshot:
    snapshot = TractionSnapshot(
        generated_at=dt.datetime.now(dt.timezone.utc)
        .replace(tzinfo=None)
        .isoformat(timespec="seconds")
        + "Z"
    )
    token = os.environ.get("GITHUB_TOKEN")

    collect_pypi(snapshot)
    collect_github_public(snapshot)
    if token:
        collect_github_traffic(snapshot, token)
        collect_code_references(snapshot, token)
    collect_hn_mentions(snapshot)
    collect_reddit_mentions(snapshot)
    return snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Traction monitor for space-ml-sim")
    parser.add_argument("--print", action="store_true", help="Print report to stdout only")
    parser.add_argument("--issue", action="store_true", help="Also write reports/traction/latest-body.md")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory to write dated report files (default: reports/traction/)",
    )
    args = parser.parse_args(argv)

    previous = load_previous_snapshot()
    current = collect_all()
    report = format_report(current, previous)

    if args.print:
        print(report)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = current.generated_at[:10]
    md_path = args.output_dir / f"{stamp}.md"
    json_path = args.output_dir / f"{stamp}.json"
    md_path.write_text(report)
    json_path.write_text(json.dumps(asdict(current), indent=2))

    latest_body = args.output_dir / "latest-body.md"
    if args.issue:
        latest_body.write_text(report)

    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    if args.issue:
        print(f"Wrote {latest_body}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
