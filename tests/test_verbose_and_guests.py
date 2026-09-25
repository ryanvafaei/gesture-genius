"""
Guests (a unique id and folder per guest, their own report), the toolbar's
New guest / Main profile, and the verbose log with its summary.
"""

import gzip
import json
from datetime import datetime

import numpy as np
import pytest

from rehab import config, features, guests, storage, verbose
from rehab.Act import SilentSpeaker
from rehab.exercises.base import RepRecord
from rehab.filters import FeatureFilter
from rehab.Think import SessionManager
from helpers import FPS, Clock, answer, returning_profile
from synthetic_hand import IMAGE_SIZE, hand


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    old = config.DATA_DIR
    monkeypatch.setattr(config, "GUESTS_DIR", tmp_path / "data" / "guests")
    config.use_data_dir(tmp_path / "data")
    yield tmp_path / "data"
    config.use_data_dir(old)


@pytest.fixture
def log(tmp_path):
    return storage.SessionLog(tmp_path / "reps.csv", tmp_path / "history.csv", session_id="s1")


def menu_session(log, **kw):
    s = SessionManager(SilentSpeaker(), returning_profile(), log, **kw)
    clock = Clock()
    s.update(features.extract(hand(), clock.t, IMAGE_SIZE), clock.t)
    s.on_key(" ", clock.t)
    answer(s, clock.t)
    assert s.stage == "menu"
    return s, clock


# --- guests -----------------------------------------------------------------------------------

def test_every_new_guest_gets_a_unique_id_and_folder(data_dir):
    first, folder1 = guests.start_guest()
    second, folder2 = guests.start_guest(hand="Right")
    assert first != second and folder1 != folder2
    assert first.startswith("guest-001-") and second.startswith("guest-002-")
    assert folder2.name == second and config.DATA_DIR == folder2
    profile = storage.load_profile()
    assert profile["guest_id"] == second and profile["affected_hand"] == "Right"
    rows = (data_dir / "guests" / "guests.csv").read_text().splitlines()
    assert rows[0].startswith("guest_id") and len(rows) == 3 and second in rows[2]


def test_guest_ids_are_unique_in_the_same_second(tmp_path):
    now = datetime(2026, 9, 25, 14, 30, 12)
    a = guests.new_guest_id(tmp_path, now)
    (tmp_path / a).mkdir()
    b = guests.new_guest_id(tmp_path, now)
    assert a == "guest-001-20260925-143012" and b == "guest-002-20260925-143012"


def test_report_folder_is_named_after_the_user(tmp_path):
    assert guests.report_dir(tmp_path, "guest-003-x").name == "guest-003-x_report"
    assert guests.user_of("20260924-100000") == "guest-20260924-100000"      # older folders
    assert guests.user_of("guest-003-x") == "guest-003-x"


def test_toolbar_new_guest_and_back_to_her(log):
    s, clock = menu_session(log, guest_id="guest-004-20260925-100000")
    items = {i["id"]: i for i in s.view()["toolbar"]["items"]}
    assert set(items) == {"short", "new_guest", "main_profile", "report"}
    assert "guest-004" in items["new_guest"]["status"]
    s.on_key("i", clock.t)
    s.on_key("b", clock.t)
    assert s.switch_user == "main_profile" and s.done


def test_session_row_names_the_user(log):
    s, _ = menu_session(log, guest_id="guest-005-x")
    assert s._session_row(None)["user_id"] == "guest-005-x"
    s2, _ = menu_session(log)
    assert s2._session_row(None)["user_id"] == "eleanor"


def one_session(folder, sid, exertion):
    log = storage.SessionLog(folder / "reps.csv", folder / "history.csv", session_id=sid)
    recs = [RepRecord("grip_release", 1, i, 0, 1, range_high=0.8, raw_high=0.6, hints=0)
            for i in range(1, 4)]
    log.save_summary(log.summarize("grip_release", recs, 1, 10))
    log.save_session({"session_id": sid, "date": "2026-09-25", "total_reps": 3,
                      "exertion": exertion, "enjoyment": 4, "safety_stop": "no"})


def test_a_guests_own_report(data_dir, tmp_path):
    from tools import report
    one_session(data_dir, "a", 2)
    guest = data_dir / "guests" / "guest-001-20260925-100000"
    one_session(guest, "g", 4)
    out = report.build(guest, guests.report_dir(guest, guest.name), user=guest.name)
    assert out.name == "guest-001-20260925-100000_report"
    md = (out / "report.md").read_text()
    assert md.startswith("# Coach data for the report: guest-001-20260925-100000")
    users = (out / "users.csv").read_text()
    assert "guest-001-20260925-100000" in users and "eleanor" not in users


def test_everyones_report_lists_each_guest(data_dir, tmp_path):
    from tools import report
    one_session(data_dir, "a", 2)
    one_session(data_dir / "guests" / "guest-001-20260925-100000", "g1", 4)
    one_session(data_dir / "guests" / "guest-002-20260925-110000", "g2", 5)
    out = report.build(data_dir, tmp_path / "report")
    users = (out / "users.csv").read_text()
    assert "guest-001-20260925-100000" in users and "guest-002-20260925-110000" in users
    assert "guest-guest" not in users
    assert "## Per user" in (out / "report.md").read_text()


def test_report_job_for_one_guest(data_dir):
    from tools.report_job import ReportJob
    guest = data_dir / "guests" / "guest-001-20260925-100000"
    one_session(guest, "g", 4)
    job = ReportJob(guest, guests.report_dir(guest, guest.name), open_when_done=False,
                    user=guest.name)
    assert job.wait(120)[0] == "done"
    assert (guest / f"{guest.name}_report" / "report.md").is_file()


# --- verbose log ------------------------------------------------------------------------------

def test_jsonable():
    out = verbose.jsonable({"a": np.float32(0.123456789), "b": float("nan"), "c": np.arange(2),
                            "d": (1, 2), "e": config.ROOT_DIR})
    assert out == {"a": 0.12346, "b": None, "c": [0, 1], "d": [1, 2], "e": str(config.ROOT_DIR)}
    json.dumps(out)


LIFT = {"index": -25, "middle": -22, "ring": -12, "pinky": -16}


def logged_tapping_session(folder, log):
    """A short finger tapping session through calibration, logged like main --verbose."""
    vlog = verbose.VerboseLog(folder, meta={"user": "test", "session_id": "s1"}, video=True,
                              fps=FPS, mirrored=True)
    speaker = SilentSpeaker()
    speaker.on_spoken = lambda text: vlog.event("speech", text=text)
    s = SessionManager(speaker, returning_profile(), log, trace=vlog.event, short=True)
    clock = Clock()
    frame = np.zeros((72, 128, 3), np.uint8)
    rng = np.random.default_rng(0)
    smoother = FeatureFilter(config.ONE_EURO_MIN_CUTOFF, config.ONE_EURO_BETA,
                             config.ONE_EURO_D_CUTOFF)

    def step(**pose):
        t = clock.tick()
        obs = hand(noise=0.001, rng=rng, back_to_camera=True, **pose)
        f = features.smooth(features.extract(obs, t, IMAGE_SIZE), smoother)      # like main.py
        s.update(f, t)
        vlog.frame(t, frame, [obs], f, s.debug_state(), loop_ms=10.0)

    step()
    s.on_key(" ", clock.t)
    answer(s, clock.t)
    s.on_key(str(config.SESSION_ORDER.index("finger_tapping") + 1), clock.t)
    tick = 0
    for _ in range(int(120 * FPS)):
        st = s.debug_state()
        if s.stage == "calibrating":
            lift = st["calibration"]["step"] == "lift"
            step(**(dict(finger_flex={"index": (-25, 0, 0)}) if lift else {}))
        elif s.stage == "exercise":
            prompted = st["detector"].get("prompted")
            tick = (tick + 1) % int(1.6 * FPS)
            if prompted and tick < int(0.8 * FPS):
                step(finger_flex={prompted: (LIFT[prompted], 0, 0)})
            else:
                step()
        else:
            step()
        if s.stage == "summary":
            break
    s.stop(clock.t)
    vlog.close(summaries=[n for n, _, _ in s.summaries])
    return s, vlog


def test_verbose_log_of_a_session(tmp_path, log):
    folder = tmp_path / "verbose" / "test_s1"
    s, vlog = logged_tapping_session(folder, log)
    assert s.stage == "summary" and s.summaries
    meta = json.loads((folder / "session.json").read_text())
    assert meta["config"]["EXERCISES"]["finger_tapping"]["finger_scale"]["ring"] == 0.7
    assert meta["totals"]["frames"] == vlog.n > 100 and "versions" in meta
    with gzip.open(folder / "frames.jsonl.gz", "rt") as fh:
        rows = [json.loads(line) for line in fh]
    assert len(rows) == vlog.n
    one = [r for r in rows if r["state"]["stage"] == "exercise"][10]
    assert len(one["hands"][0]["world"]) == 21 and "tip_reach" in one["features"]
    assert set(one["state"]["detector"]["scores"]) == set(features.FINGERS)
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    kinds = {e["type"] for e in events}
    assert {"stage", "key", "speech", "exercise", "calibration", "detect", "rep"} <= kinds
    said = [e["text"] for e in events if e["type"] == "speech"]
    assert "How is your hand feeling today?" in said
    detect = [e for e in events if e["type"] == "detect" and e["event"] == "start"]
    assert detect and all(e["correct"] for e in detect)
    assert (folder / "video.mp4").stat().st_size > 0


def test_verbose_summary(tmp_path, log):
    from tools import verbose_summary
    folder = tmp_path / "verbose" / "test_s1"
    logged_tapping_session(folder, log)
    s = verbose_summary.summarize(folder)
    assert s["session"]["frames"] > 100
    assert s["tracking"]["hand_present_share"] == 1.0
    tapping = s["detection"]["finger_tapping"]
    assert set(tapping) == set(features.FINGERS)
    assert all(e["found"] == e["prompts"] and not e["wrong_fingers"] for e in tapping.values())
    assert tapping["index"]["prompted_peak"]["median"] >= 1.0
    assert "finger_tapping/flat" in s["noise_floor"]
    md = verbose_summary.markdown(s)
    assert "## Detection" in md and "## Tracking" in md


def test_replay_of_a_logged_session(tmp_path, log):
    from tools import replay
    folder = tmp_path / "verbose" / "test_s1"
    logged_tapping_session(folder, log)
    timeline, info = replay.replay(folder)
    assert info["calibration"] == "this log's calibration frames"
    assert info["image_size"] == IMAGE_SIZE
    fingers = [finger for _, finger in info["starts"]]
    # the last lift ends the exercise: its frame is logged as the next stage
    assert fingers[:6] == ["index", "middle", "ring", "pinky", "ring", "middle"]
    assert not [text for _, source, text in timeline
                if source == "replay" and text.startswith("quality") and not text.endswith("ok")]
    assert any(source == "log" and text.startswith("start") for _, source, text in timeline)
