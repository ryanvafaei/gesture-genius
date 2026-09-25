import numpy as np
import pytest

from rehab import features
from rehab.calibration import CalibrationRoutine, is_stale
from rehab.exercises.base import Hysteresis, count_speed_peaks, scale
from rehab.exercises.grip_release import GripRelease
from rehab.filters import FeatureFilter, OneEuroFilter
from helpers import Clock, feat
from synthetic_hand import IMAGE_SIZE, hand


# --- features -------------------------------------------------------------

def test_open_hand_is_open_and_palm_faces_camera():
    f = feat()
    assert f.present and f.correct_hand and not f.too_small
    assert all(f.openness[k] > 0.9 for k in features.FINGERS)
    assert f.palm_facing > 0.9
    assert f.quality_problem(need_palm_facing=True) is None


def test_fist_is_closed():
    f = feat(flex=(70, 90, 60))
    assert all(f.openness[k] < 0.2 for k in features.FINGERS)
    assert f.openness_mean < 0.2


@pytest.mark.parametrize("spread", [0.0, 8.0, 15.0])
def test_spread_angles_in_palm_plane(spread):
    f = feat(spread=spread)
    for gap in features.GAP_NAMES:
        assert f.spread[gap] == pytest.approx(spread, abs=0.5)


def test_thumb_touch_distance_is_smallest_for_touched_finger():
    f = feat(thumb_touch="ring")
    assert min(f.thumb_tip_dist, key=f.thumb_tip_dist.get) == "ring"
    assert f.thumb_tip_dist["ring"] < 0.1


def test_lifted_finger_rises_above_palm_plane():
    flat = feat()
    lifted = feat(finger_flex={"index": (-25, 0, 0)})
    assert lifted.tip_height["index"] - flat.tip_height["index"] > 0.2
    assert lifted.tip_height["middle"] == pytest.approx(flat.tip_height["middle"], abs=1e-6)


def test_features_do_not_depend_on_distance_to_camera():
    near = features.extract(hand(), 0, IMAGE_SIZE)
    obs = hand()
    obs.image = obs.image * 0.5      # further away: smaller in the image
    far = features.extract(obs, 0, IMAGE_SIZE)
    assert far.openness == pytest.approx(near.openness)
    assert far.spread == pytest.approx(near.spread)


def test_quality_flags():
    assert features.extract(None, 0, IMAGE_SIZE).quality_problem() == "no_hand"
    right = features.extract(hand(handedness="Right"), 0, IMAGE_SIZE)
    assert right.quality_problem() == "wrong_hand"
    obs = hand()
    obs.image = obs.image * 0.1
    assert features.extract(obs, 0, IMAGE_SIZE).quality_problem() == "too_far"


def test_palm_away_detected():
    obs = hand()
    obs.image[:, 0] = 1.0 - obs.image[:, 0]          # back of the hand towards the camera
    f = features.extract(obs, 0, IMAGE_SIZE)
    assert f.palm_facing < 0
    assert f.quality_problem(need_palm_facing=True) == "palm_away"
    assert f.quality_problem(need_palm_facing=False) is None


def test_choose_hand_prefers_affected_hand():
    left, right = hand(), hand(handedness="Right")
    assert features.choose_hand([right, left], "Left") is left
    assert features.choose_hand([right], "Left") is right
    assert features.choose_hand([], "Left") is None


def _at(obs, dx):
    obs.image = obs.image + np.array([dx, 0.0, 0.0])
    return obs


def test_pair_hands_uses_position_when_labels_agree():
    # MediaPipe labels both hands "Left": in the mirrored image her left is on the left
    a, b = _at(hand(), -0.3), _at(hand(), 0.25)
    for order in ([a, b], [b, a]):
        left, right = features.pair_hands(order, "Left", mirror=True)
        assert left.image[0][0] < right.image[0][0]
        assert (left.handedness, right.handedness) == ("Left", "Right")
    affected, other = features.pair_hands([a, b], "Right", mirror=True)
    assert affected.image[0][0] > other.image[0][0] and affected.handedness == "Right"
    left, right = features.pair_hands([a, b], "Left", mirror=False)
    assert left.image[0][0] > right.image[0][0]


def test_pair_hands_one_hand_and_duplicates():
    left = hand()
    assert features.pair_hands([left], "Left") == (left, None)
    assert features.pair_hands([], "Left") == (None, None)
    twice = _at(hand(handedness="Right"), 0.005)                # the same hand detected twice
    affected, other = features.pair_hands([twice, left], "Left")
    assert other is None and affected is not None


def test_both_hands_are_drawn():
    from rehab.Act import Display
    f = features.extract(_at(hand(), -0.25), 0, IMAGE_SIZE)
    f.other = features.extract(_at(hand(handedness="Right"), 0.2), 0, IMAGE_SIZE)
    frame = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), np.uint8)
    Display(screen=None).render(frame, {}, f)
    for h in (f, f.other):
        x, y = h.image_points[0].astype(int)
        assert frame[y - 3:y + 4, x - 3:x + 4].any()


def test_smoothing_reduces_jitter():
    rng = np.random.default_rng(1)
    filt = FeatureFilter(min_cutoff=1.0, beta=0.05)
    raw, smooth = [], []
    for i in range(120):
        t = i / 30
        f = features.extract(hand(noise=0.002, rng=rng), t, IMAGE_SIZE)
        raw.append(f.openness_mean)
        features.smooth(f, filt)
        smooth.append(f.openness_mean)
    assert np.std(smooth[30:]) < 0.5 * np.std(raw[30:])


# --- filters ----------------------------------------------------------------

def test_one_euro_smooths_still_signal_and_follows_movement():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.5)
    still = [f(0.5 + rng.normal(0, 0.02), i / 30) for i in range(60)]
    assert np.std(still[20:]) < 0.01
    # fast step: should be most of the way there within 0.2 s
    out = [f(1.0, (60 + i) / 30) for i in range(6)]
    assert out[-1] > 0.85


def test_one_euro_arrays():
    f = OneEuroFilter()
    out = f(np.array([1.0, 2.0]), 0.0)
    assert out.shape == (2,)


# --- hysteresis & helpers ----------------------------------------------------

def test_hysteresis_does_not_flicker():
    h = Hysteresis(high=0.7, low=0.3, gap=0.1)
    zones = [h.update(v) for v in (0.5, 0.71, 0.65, 0.69, 0.62, 0.71, 0.59, 0.3, 0.35, 0.39, 0.41)]
    assert zones == ["mid", "high", "high", "high", "high", "high", "mid", "low", "low", "low", "mid"]


def test_scale_handles_reversed_and_tiny_ranges():
    assert scale(20, 120, 20) == pytest.approx(1.0)
    assert scale(0.5, 0.5, 0.5) == pytest.approx(0.0)


def test_speed_peaks_smooth_vs_jerky():
    t = np.linspace(0, 2, 60)
    smooth = 0.5 - 0.5 * np.cos(np.pi * t / 2)
    jerky = np.concatenate([np.linspace(0, 0.3, 15), np.full(15, 0.3),
                            np.linspace(0.3, 0.6, 15), np.full(15, 0.6)])
    assert count_speed_peaks(t, smooth) == 1
    assert count_speed_peaks(t, jerky) == 2


# --- calibration ---------------------------------------------------------------

def test_calibration_records_personal_range():
    clock = Clock()
    routine = CalibrationRoutine(GripRelease)
    said = routine.start(clock.t)
    assert "open your hand" in said[-1].text.lower()
    # partial opening only: this is her maximum
    while routine.step.name == "open":
        t = clock.tick()
        routine.update(feat(t, flex=(30, 40, 20)), t)
    while not routine.done:
        t = clock.tick()
        routine.update(feat(t, flex=(60, 80, 50)), t)
    res = routine.result
    assert res["open"]["mean"] > res["closed"]["mean"]
    assert res["open"]["mean"] < 0.7           # not a healthy hand's opening


def test_calibration_pauses_on_bad_frames():
    clock = Clock()
    routine = CalibrationRoutine(GripRelease, settle_s=0.5, hold_s=1.0)
    routine.start(clock.t)
    for _ in range(90):          # 3 s of bad frames: no progress
        t = clock.tick()
        routine.update(feat(t), t, quality_ok=False)
    assert routine.step.name == "open"
    for _ in range(40):
        t = clock.tick()
        routine.update(feat(t), t)
    assert routine.step.name == "closed"


def test_is_stale():
    from datetime import date
    assert is_stale(None)
    assert is_stale({"date": "2020-01-01", "steps": {"a": {}}}, today=date(2020, 2, 1))
    assert not is_stale({"date": "2020-01-01", "steps": {"a": {}}}, today=date(2020, 1, 5))
