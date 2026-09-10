"""The bundled cron-watcher scripts.

Their whole contract is "print only what is new, print nothing otherwise" —
that is what makes a cron script job stay quiet — so these drive each script
end to end and assert on stdout and exit code.

Fixtures are served over a loopback HTTP server rather than ``file://``: the
watchers only speak ``http(s)`` now, because a watcher that accepts any scheme
``urlopen`` supports is an arbitrary local-file reader (Issue #1065). Loopback
keeps the run offline and the port is picked by the OS.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "agentos"
    / "skills"
    / "bundled"
    / "cron-watchers"
    / "scripts"
)

RSS = """<?xml version="1.0"?><rss><channel>
<item><title>First post</title><link>https://example.com/1</link><guid>1</guid></item>
<item><title>Second post</title><link>https://example.com/2</link><guid>2</guid></item>
</channel></rss>"""

ATOM = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><id>tag:a</id><title>Atom one</title><link href="https://example.com/a"/></entry>
</feed>"""


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTOS_STATE_DIR", str(tmp_path / "state"))
    return tmp_path


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass  # keep pytest's captured stderr about the test, not the server


@pytest.fixture
def base_url(state_dir):
    """Serve ``state_dir`` over loopback HTTP and yield its base URL."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_QuietHandler, directory=str(state_dir)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def _run(script: str, *args: str, env_home: Path) -> subprocess.CompletedProcess[str]:
    # A deliberately small environment, so a watcher cannot reach the
    # developer's own state. Windows is the exception: a child started without
    # the system variables cannot initialise Winsock, and every fetch then dies
    # with `WinError 10106` before it reaches the fixture server. State stays
    # isolated either way — the watermark store reads AGENTOS_STATE_DIR first.
    if os.name == "nt":
        env = dict(os.environ)
        env["USERPROFILE"] = str(env_home)
    else:
        env = {"PATH": "/usr/bin:/bin"}
    env["AGENTOS_STATE_DIR"] = str(env_home / "state")
    env["HOME"] = str(env_home)
    # Loopback must never go through a proxy the runner happens to configure.
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def _feed(tmp_path: Path, base: str, name: str, body: str) -> str:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return f"{base}/{name}"


# ── watch_rss ───────────────────────────────────────────────────────────────


def test_rss_first_run_is_silent(state_dir, base_url):
    url = _feed(state_dir, base_url, "feed.xml", RSS)

    result = _run("watch_rss.py", "--url", url, "--name", "t", env_home=state_dir)

    assert result.returncode == 0
    assert result.stdout == ""


def test_rss_first_run_can_report_everything(state_dir, base_url):
    url = _feed(state_dir, base_url, "feed.xml", RSS)

    result = _run(
        "watch_rss.py", "--url", url, "--name", "t", "--first-run-reports", env_home=state_dir
    )

    assert result.returncode == 0
    assert "First post" in result.stdout
    assert "Second post" in result.stdout


def test_rss_reports_only_what_is_new(state_dir, base_url):
    url = _feed(state_dir, base_url, "feed.xml", RSS)
    _run("watch_rss.py", "--url", url, "--name", "t", env_home=state_dir)

    _feed(
        state_dir,
        base_url,
        "feed.xml",
        RSS.replace(
            "</channel>",
            "<item><title>Third post</title><link>https://example.com/3</link>"
            "<guid>3</guid></item></channel>",
        ),
    )
    result = _run("watch_rss.py", "--url", url, "--name", "t", env_home=state_dir)

    assert result.returncode == 0
    assert "Third post" in result.stdout
    assert "First post" not in result.stdout


def test_rss_unchanged_feed_stays_silent(state_dir, base_url):
    url = _feed(state_dir, base_url, "feed.xml", RSS)
    _run("watch_rss.py", "--url", url, "--name", "t", env_home=state_dir)

    result = _run("watch_rss.py", "--url", url, "--name", "t", env_home=state_dir)

    assert result.returncode == 0
    assert result.stdout == ""


def test_rss_reads_atom_entries(state_dir, base_url):
    url = _feed(state_dir, base_url, "atom.xml", ATOM)

    result = _run(
        "watch_rss.py", "--url", url, "--name", "a", "--first-run-reports", env_home=state_dir
    )

    assert result.returncode == 0
    assert "Atom one" in result.stdout


def test_rss_fails_loudly_on_a_broken_feed(state_dir, base_url):
    url = _feed(state_dir, base_url, "broken.xml", "not xml at all")

    result = _run("watch_rss.py", "--url", url, "--name", "b", env_home=state_dir)

    assert result.returncode == 1
    assert "not valid XML" in result.stderr


def test_watermarks_are_per_name(state_dir, base_url):
    url = _feed(state_dir, base_url, "feed.xml", RSS)
    _run("watch_rss.py", "--url", url, "--name", "one", env_home=state_dir)

    result = _run(
        "watch_rss.py", "--url", url, "--name", "two", "--first-run-reports", env_home=state_dir
    )

    assert "First post" in result.stdout


# ── watch_http_json ─────────────────────────────────────────────────────────


def _events(tmp_path: Path, base: str, items: list[dict]) -> str:
    return _feed(tmp_path, base, "events.json", json.dumps({"data": {"events": items}}))


def test_json_reports_only_new_items(state_dir, base_url):
    url = _events(state_dir, base_url, [{"event_id": "a1", "title": "Deploy finished"}])
    args = ("--url", url, "--name", "j", "--id-field", "event_id", "--items-path", "data.events")
    _run("watch_http_json.py", *args, env_home=state_dir)

    _events(state_dir, base_url, [{"event_id": "a2", "title": "Alert cleared"}])
    result = _run("watch_http_json.py", *args, env_home=state_dir)

    assert result.returncode == 0
    assert result.stdout.strip() == "- Alert cleared"


def test_json_accepts_a_top_level_list(state_dir, base_url):
    url = _feed(state_dir, base_url, "list.json", json.dumps([{"id": "x", "name": "thing"}]))

    result = _run(
        "watch_http_json.py",
        "--url",
        url,
        "--name",
        "l",
        "--first-run-reports",
        env_home=state_dir,
    )

    assert result.returncode == 0
    assert "thing" in result.stdout


def test_json_reports_the_requested_fields(state_dir, base_url):
    url = _events(state_dir, base_url, [{"event_id": "a1", "title": "t", "sev": "high"}])

    result = _run(
        "watch_http_json.py",
        "--url",
        url,
        "--name",
        "f",
        "--id-field",
        "event_id",
        "--items-path",
        "data.events",
        "--field",
        "sev",
        "--first-run-reports",
        env_home=state_dir,
    )

    assert "sev='high'" in result.stdout


def test_json_fails_when_the_path_holds_no_list(state_dir, base_url):
    url = _events(state_dir, base_url, [])

    result = _run(
        "watch_http_json.py",
        "--url",
        url,
        "--name",
        "n",
        "--items-path",
        "data.missing",
        env_home=state_dir,
    )

    assert result.returncode == 1
    assert "Expected a list" in result.stderr


# ── URL scheme guard (Issue #1065) ──────────────────────────────────────────


@pytest.mark.parametrize("script", ["watch_rss.py", "watch_http_json.py"])
def test_watcher_refuses_a_file_url(script, state_dir):
    # urlopen speaks file:// as happily as http://, so an unguarded --url made
    # every watcher an arbitrary local-file reader that reported the contents
    # on each run. Asserted at the real entry point, not just at the helper.
    secret = state_dir / "secret.json"
    secret.write_text('[{"id": "1", "name": "leaked"}]', encoding="utf-8")

    result = _run(script, "--url", secret.as_uri(), "--name", "t", env_home=state_dir)

    assert result.returncode == 1
    assert "must be http:// or https://" in result.stderr
    assert "leaked" not in result.stdout


@pytest.mark.parametrize("url", ["ftp://example.com/x", "data:application/json,[]", "/etc/passwd"])
def test_watcher_refuses_other_non_http_schemes(url, state_dir):
    result = _run("watch_http_json.py", "--url", url, "--name", "t", env_home=state_dir)

    assert result.returncode == 1
    assert "Refusing to fetch" in result.stderr


# ── watch_github ────────────────────────────────────────────────────────────


def test_github_rejects_a_malformed_repo(state_dir):
    result = _run("watch_github.py", "--repo", "not-a-repo", env_home=state_dir)

    assert result.returncode == 1
    assert "owner/name" in result.stderr


def test_github_rejects_an_unknown_scope(state_dir):
    result = _run("watch_github.py", "--repo", "o/n", "--scope", "stars", env_home=state_dir)

    assert result.returncode == 2  # argparse rejects the choice


# ── --limit caps a run, not the backlog (Issue #1674) ───────────────────────


def _rss_with(count: int, offset: int = 0) -> str:
    items = "".join(
        f"<item><title>Post {n}</title><link>https://example.com/{n}</link><guid>{n}</guid></item>"
        for n in range(offset, offset + count)
    )
    return f'<?xml version="1.0"?><rss><channel>{items}</channel></rss>'


def test_rss_reports_the_surplus_on_the_next_run(state_dir, base_url):
    """More new items than ``--limit`` must not be consumed unreported.

    The watermark used to record every fresh id and the caller then printed
    only the first ``--limit`` of them, so a feed that published more than the
    cap between two runs lost the difference for good.
    """
    url = _feed(state_dir, base_url, "feed.xml", _rss_with(1))
    _run("watch_rss.py", "--url", url, "--name", "t", env_home=state_dir)  # adopt silently

    _feed(state_dir, base_url, "feed.xml", _rss_with(6))  # 5 new items, cap of 2
    first = _run("watch_rss.py", "--url", url, "--name", "t", "--limit", "2", env_home=state_dir)
    assert first.returncode == 0
    assert first.stdout.count("Post ") == 2

    reported = first.stdout
    for _ in range(3):
        nxt = _run("watch_rss.py", "--url", url, "--name", "t", "--limit", "2", env_home=state_dir)
        assert nxt.returncode == 0
        reported += nxt.stdout

    # Every one of the five new posts is reported, each exactly once.
    for n in range(1, 6):
        assert reported.count(f"Post {n}\n") == 1, f"Post {n} in {reported!r}"

    # And once the backlog is drained the watcher goes quiet again.
    assert _run("watch_rss.py", "--url", url, "--name", "t", env_home=state_dir).stdout == ""


def test_json_reports_the_surplus_on_the_next_run(state_dir, base_url):
    args = ("--name", "j", "--id-field", "event_id", "--items-path", "data.events")
    url = _events(state_dir, base_url, [{"event_id": "seed", "title": "Seed"}])
    _run("watch_http_json.py", "--url", url, *args, env_home=state_dir)

    _events(
        state_dir,
        base_url,
        [{"event_id": f"e{n}", "title": f"Event {n}"} for n in range(4)],
    )
    first = _run("watch_http_json.py", "--url", url, *args, "--limit", "1", env_home=state_dir)
    assert first.stdout.count("Event ") == 1

    reported = first.stdout
    for _ in range(3):
        reported += _run(
            "watch_http_json.py", "--url", url, *args, "--limit", "1", env_home=state_dir
        ).stdout

    for n in range(4):
        assert reported.count(f"Event {n}\n") == 1, f"Event {n} in {reported!r}"


def test_a_capped_run_records_only_what_it_reported(state_dir):
    """The watermark itself, not just the printed lines.

    ``select_new`` is the shared helper all three watchers commit through, so
    assert on the stored state directly: an id that was held back must not be
    in the file.
    """
    sys.path.insert(0, str(SCRIPTS))
    try:
        import _watermark  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)

    # ``state_dir`` already points AGENTOS_STATE_DIR at this test's tmp_path.
    ids = [f"item-{n:02d}" for n in range(6)]

    _watermark.select_new("cap", ids[:1])  # first run adopts silently
    assert _watermark.load_seen("cap") == ["item-00"]

    assert _watermark.select_new("cap", ids, limit=2) == ["item-01", "item-02"]
    assert _watermark.load_seen("cap") == ["item-00", "item-01", "item-02"]

    assert _watermark.select_new("cap", ids, limit=2) == ["item-03", "item-04"]
    assert _watermark.select_new("cap", ids, limit=2) == ["item-05"]
    assert _watermark.select_new("cap", ids, limit=2) == []
    assert _watermark.load_seen("cap") == ids


def test_the_first_run_adopts_the_whole_feed_despite_a_low_limit(state_dir, base_url):
    """The silent first run stays uncapped.

    Capping it would leave the rest of the page to arrive as "new" on the next
    run — exactly the dump the silent first run exists to prevent.
    """
    url = _feed(state_dir, base_url, "feed.xml", _rss_with(5))

    first = _run("watch_rss.py", "--url", url, "--name", "t", "--limit", "1", env_home=state_dir)
    assert first.stdout == ""

    second = _run("watch_rss.py", "--url", url, "--name", "t", "--limit", "1", env_home=state_dir)
    assert second.stdout == "", "the adopted feed must not resurface as new"
