"""The benchmark engine (rehab/benchmarks.py), the body measures and the benchmark data."""

import json

import numpy as np
import pytest

from rehab import benchmarks, body, config, features
from rehab.exercises import ARM_EXERCISES
from rehab.filters import LowPass, LowPassBank
from synthetic_body import IMAGE_SIZE, feat, pose
from synthetic_hand import hand

B = benchmarks.load()


# --- data -------------------------------------------------------------------------------

def test_pose_landmarks_match_the_benchmark_file():
    for name, index in B.pose.items():
        if name in body.POSE:
            assert body.POSE[name] == index


def test_every_arm_exercise_has_a_profile():
    for name in list(ARM_EXERCISES) + ["grip_release", "thumb_opposition"]:
        assert B.exercise(name), name


def test_tolerance_is_at_least_the_goniometer_floor_and_the_worst_loa():
    """Plan 4: max(5 degrees, worst |95% LoA| of lab and home), rounded up to 0.5."""
    for key, entry in B.tol.items():
        loa = entry.get("loa") or {}
        worst = max((abs(v) for pair in loa.values() for v in pair), default=0.0)
        assert entry["tolerance_deg"] >= 5.0
        assert entry["tolerance_deg"] >= worst, key


def test_mdc_lab_home_and_provisional():
    assert B.mdc("shoulder_flexion_raise") == 13.19
    assert B.mdc("shoulder_flexion_raise", "home") == 6.35
    assert B.mdc("hand_to_head") == 32.65
    assert B.mdc("wrist_extension") == 10.0          # no MDC published: provisional


def test_every_proposed_default_is_marked():
    """Values in the plan's section 14 carry "proposed" in the json."""
    for key in ("visibility_min", "invalid_frames_pause_s", "state_debounce_frames"):
        assert B.raw["signal_processing"][key]["proposed"] is True
    assert B.compensation("shoulder_girdle_elevation")["proposed"] is True
    assert B.compensation("arm_leaves_movement_plane")["proposed"] is True


# --- Sense: angles in pixels ------------------------------------------------------------

@pytest.mark.parametrize("shoulder,elbow", [(0, 0), (45, 10), (90, 0), (120, 30)])
def test_shoulder_and_elbow_angles(shoulder, elbow):
    f = feat(0.0, shoulder=shoulder, elbow=elbow)
    assert abs(f.arm["left"]["shoulder_elevation"] - shoulder) < 0.5
    assert abs(f.arm["left"]["elbow_flexion"] - elbow) < 0.5


def test_pixel_scaling_matters():
    """A 45 degree arm measured in normalised 16:9 coordinates is more than 5 degrees off."""
    p = pose(shoulder=45)
    right = body.to_px(p.image, *IMAGE_SIZE)
    wrong = p.image[:, :2]
    assert abs(body.shoulder_elevation(right, "left") - 45) < 0.5
    assert abs(body.shoulder_elevation(wrong, "left") - 45) > 5


def test_view_from_shoulder_width():
    assert feat(0.0, view="sagittal").view == "sagittal"
    assert feat(0.0, view="frontal").view == "frontal"
    assert feat(0.0, view="oblique").view == "oblique"


def test_wrist_extension_sign_is_hand_up():
    assert abs(feat(0.0, elbow=90, hand_ext=30).arm["left"]["wrist_extension"] - 30) < 0.5
    assert abs(feat(0.0, elbow=90, hand_ext=-20).arm["left"]["wrist_extension"] + 20) < 0.5


def test_trunk_lean_and_shoulder_hike():
    assert abs(feat(0.0, trunk=12).trunk_angle - 12) < 0.5
    a, b = feat(0.0, view="frontal"), feat(0.0, view="frontal", hike=40)
    assert (a.arm["left"]["ear_gap"] - b.arm["left"]["ear_gap"]) / b.shoulder_width > 0.15


def test_setup_problem():
    need = ["left_hip", "left_shoulder", "left_elbow", "left_wrist"]
    assert body.setup_problem(feat(0.0), need, "sagittal", "left") is None
    assert body.setup_problem(feat(0.0, view="frontal"), need, "sagittal", "left") == "turn_side"
    assert body.setup_problem(feat(0.0, hidden=("left_wrist",)), need, "sagittal", "left") == "arm_hidden"
    assert body.setup_problem(feat(0.0), need, "frontal", "left") == "face_camera"
    empty = body.extract(None, 0.0, IMAGE_SIZE)
    assert body.setup_problem(empty, need, "sagittal", "left") == "no_body"


def test_hand_benchmark_measures():
    f = features.extract(hand(flex=(0, 0, 0)), 0.0, (1280, 720))
    fist = features.extract(hand(flex=(85, 100, 75), thumb_out=1.0), 0.0, (1280, 720))
    assert f.aperture > fist.aperture
    assert fist.tip_to_palm["middle"] < f.tip_to_palm["middle"]
    assert f.pinch_gap == f.thumb_tip_dist["index"]


def test_finger_scores_open_and_closed_hand():
    tol = B.tol["finger_joints"]["tolerance_deg"]
    flat = features.extract(hand(flex=(0, 0, 0)), 0.0, (1280, 720))
    res = benchmarks.hand_open_score(benchmarks.bain_joints(flat.joint_flexion), tol)
    assert res["fma25_style"] == 2 and res["bain_functional_open"]
    bent = features.extract(hand(flex=(40, 50, 30)), 0.0, (1280, 720))
    res = benchmarks.hand_open_score(benchmarks.bain_joints(bent.joint_flexion), tol)
    assert res["fma25_style"] == 1 and not res["bain_functional_open"]
    fist = features.extract(hand(flex=(85, 100, 75)), 0.0, (1280, 720))
    res = benchmarks.hand_close_score(benchmarks.bain_joints(fist.joint_flexion),
                                      fist.tip_to_palm, thumb_outside=True)
    assert res["bain_functional_grasp"]


def test_pinch_score_needs_the_other_fingers_away():
    assert benchmarks.pinch_score(0.08, [0.6, 0.8, 0.9]) == 1
    assert benchmarks.pinch_score(0.08, [0.1, 0.8, 0.9]) == 0      # the middle finger helps
    assert benchmarks.pinch_score(0.3, [0.6, 0.8, 0.9]) == 0


# --- filters ------------------------------------------------------------------------------

def test_lowpass_settles_and_smooths():
    lp = LowPass(fs=30, fc=6)
    assert abs([lp(90.0) for _ in range(30)][-1] - 90) < 1e-6
    rng = np.random.default_rng(0)
    noisy = 90 + rng.normal(0, 3, 300)
    lp = LowPass(fs=30, fc=6)
    out = np.array([lp(v) for v in noisy])
    assert np.std(out[30:]) < np.std(noisy[30:])
    assert lp(float("nan")) == out[-1]


def test_lowpass_bank_restarts_after_a_gap():
    bank = LowPassBank(fs=30, fc=6, reset_after_s=0.5)
    for i in range(30):
        bank("a", 10.0, i / 30)
    assert bank("a", 50.0, 3.0) == 50.0          # a new start, no glide from 10


# --- layer 1: one rep ----------------------------------------------------------------------

def up_down(peak, start=5.0, n=40):
    return list(np.linspace(start, peak, n)) + list(np.linspace(peak, start, n))


def test_score_rep_levels():
    tol = B.tolerance("shoulder_flexion_raise")
    r = benchmarks.score_rep(up_down(100), "increase", tol, personal_target=80, fma_threshold=90)
    assert r.fma_style_score == 2 and r.coaching_success and r.feedback_key == "praise_specific"
    # inside the uncertainty band around 90: the lower score (FMA: when uncertain)
    r = benchmarks.score_rep(up_down(93), "increase", tol, 80, 90)
    assert r.fma_style_score == 1 and "within_uncertainty_band" in r.reasons and r.coaching_success
    # elbow bends before 90: 1, and not a coaching success
    r = benchmarks.score_rep(list(np.linspace(5, 110, 60)), "increase", tol, 80, 90,
                             form_frames={"elbow_straight": list(range(10, 30))})
    assert r.fma_style_score == 1 and not r.coaching_success and r.cue == "elbow_straight"
    # trunk compensation: 1, but still a success, with one cue
    r = benchmarks.score_rep(list(np.linspace(5, 110, 60)), "increase", tol, 80, 90,
                             compensation_frames={"trunk": list(range(40, 60))})
    assert r.fma_style_score == 1 and r.coaching_success and r.with_compensation
    assert r.feedback_key == "praise_plus_one_cue" and r.cue == "trunk"
    # moved less than the tolerance: 0
    assert benchmarks.score_rep([5, 6, 8, 7, 6], "increase", tol, 80, 90).fma_style_score == 0
    # the start position could not be obtained: 0
    assert benchmarks.score_rep(up_down(100), "increase", tol, 80, 90, start_ok=False).fma_style_score == 0
    # target reached within the tolerance counts (benefit of the doubt)
    assert benchmarks.score_rep(up_down(76), "increase", tol, 80).coaching_success
    assert benchmarks.score_rep(up_down(70), "increase", tol, 80).cue == "more_range"


def test_rules_need_consecutive_frames():
    tol = 5.5
    scattered = benchmarks.score_rep(up_down(100), "increase", tol, 80, 90,
                                     form_frames={"elbow_straight": [3, 7, 11, 15, 19, 23]})
    assert scattered.coaching_success and not scattered.broken


def test_webcam_cap_and_scale_end():
    r = benchmarks.score_rep(up_down(40, start=0), "increase", 5.0, 20, 15, max_score=1)
    assert r.fma_style_score == 1                     # FMA 19: resistance cannot be measured
    # elbow extension to 0: a 2D angle is never below 0, so within the tolerance counts
    ext = list(np.linspace(90, 3, 40)) + list(np.linspace(3, 90, 40))
    r = benchmarks.score_rep(ext, "decrease", 5.0, 10, 0, at_scale_end=True)
    assert r.fma_style_score == 2
    r = benchmarks.score_rep(list(np.linspace(90, 20, 40)), "decrease", 5.0, 10, 0, at_scale_end=True)
    assert r.fma_style_score == 1


# --- layer 2 --------------------------------------------------------------------------------

def test_plausibility_uses_soucie_max():
    top = B.plausible_max("shoulder_flexion_raise", "female")
    assert top == B.normative("shoulder_flexion", "female")["max_deg"] + 5.5
    assert benchmarks.is_plausible(170, top) and not benchmarks.is_plausible(200, top)


def test_rps_trunk():
    assert benchmarks.rps_trunk_score(0.05, "close") == 3
    assert benchmarks.rps_trunk_score(0.3, "close") == 2
    assert benchmarks.rps_trunk_score(0.6, "close") == 1
    assert benchmarks.rps_trunk_score(0.25, "far") == 3
    assert benchmarks.rps_trunk_score(0.25, "far", elbow_almost_full=False) == 2
    assert benchmarks.rps_trunk_score(0.8, "far", hand_arrived=False) == 0


def test_fma33_and_fma32():
    assert benchmarks.fma33_time_score(9.5, 8.0) == 2
    assert benchmarks.fma33_time_score(12.0, 8.0) == 1
    assert benchmarks.fma33_time_score(14.0, 8.0) == 0
    assert benchmarks.fma33_time_score(9.0, 8.0, completed_reps=4) == 0
    assert benchmarks.fma32_dysmetria_score([0.1, 0.2, 0.3]) == 2
    assert benchmarks.fma32_dysmetria_score([0.1, 0.6]) == 1
    assert benchmarks.fma32_dysmetria_score([1.5]) == 0


# --- layer 3 --------------------------------------------------------------------------------

def test_ladder_and_milestones():
    ms = [dict(m) for m in B.milestones("shoulder_elevation")] + [{"deg": 90.0, "task": "FMA 13"}]
    lad = benchmarks.build_ladder(55, unaffected_best=150, tolerance=5.5, milestones=ms,
                                  normative_ceiling=160)
    assert lad.ceiling == 150 and lad.target(0) == 55 and lad.target(3) == 55 + 16.5
    assert lad.target(99) == 150 and lad.max_level == 18
    assert [m["deg"] for m in lad.milestones_crossed(60, 90)] == [71, 86, 90]
    # the plan's worked example: right 150, left 58, tolerance 5.5
    lad = benchmarks.build_ladder(58, 150, 5.5, ms, 160)
    assert (lad.target(1), lad.target(3)) == (63.5, 74.5)
    # therapist limit caps it; never below the baseline
    assert benchmarks.build_ladder(55, 150, 5.5, therapist_limit=100).ceiling == 100
    assert benchmarks.build_ladder(55, 40, 5.5).ceiling == 55


def test_ladder_going_down():
    lad = benchmarks.build_ladder(40, unaffected_best=2, tolerance=5, normative_ceiling=0,
                                  milestones=[{"deg": 0.0}], direction="decrease")
    assert lad.ceiling == 2 and lad.target(1) == 35 and lad.target(20) == 2
    assert lad.milestones == []                       # 0 is beyond her ceiling of 2


def test_calibration_summary_and_symmetry():
    right = benchmarks.side_summary([145, 150, 148])
    left = benchmarks.side_summary([55, 58, 50])
    assert right["best"] == 150 and left["best"] == 58 and left["mean"] == pytest.approx(54.33)
    assert benchmarks.symmetry_index(left, right) == pytest.approx(58 / 150, abs=1e-3)
    down_l = benchmarks.side_summary([40, 35, 38], "decrease", starts=[90, 90, 90])
    down_r = benchmarks.side_summary([2, 3, 4], "decrease", starts=[90, 90, 90])
    assert benchmarks.symmetry_index(down_l, down_r, "decrease") == pytest.approx(55 / 88, abs=1e-3)


def test_progress_claim_uses_mdc():
    mdc = B.mdc("shoulder_flexion_raise")          # 13.19
    assert not benchmarks.claim_improvement(70, 60, mdc)
    assert benchmarks.claim_improvement(75, 60, mdc)
    assert benchmarks.claim_improvement(19, 40, 20.55, "decrease")
    assert not benchmarks.claim_improvement(20, 40, 20.55, "decrease")
    assert not benchmarks.claim_improvement(75, 60, None)


def test_progression_rules():
    good = {"success_rate": 1.0, "clean_success_rate": 0.9, "compensation_rate": 0.0}
    ok = {"success_rate": 0.7, "clean_success_rate": 0.6, "compensation_rate": 0.0}
    bad = {"success_rate": 0.4, "clean_success_rate": 0.3, "compensation_rate": 0.0}
    comp = {"success_rate": 0.9, "clean_success_rate": 0.3, "compensation_rate": 0.7}
    assert benchmarks.next_level(2, [good, good]) == (3, "raised")
    assert benchmarks.next_level(2, [good]) == (2, None)             # needs two sessions in a row
    assert benchmarks.next_level(2, [ok, good]) == (2, None)
    assert benchmarks.next_level(2, [good, ok]) == (2, None)
    assert benchmarks.next_level(2, [good, bad]) == (1, "lowered")
    assert benchmarks.next_level(2, [good, comp]) == (1, "lowered")
    rising = dict(good, compensation_rate=0.2)
    assert benchmarks.next_level(2, [good, rising]) == (2, None)     # compensation rising
    assert benchmarks.next_level(0, [bad]) == (0, None)
    assert benchmarks.next_level(2, [good, good], pain_delta=2) == (2, "pause")
    assert benchmarks.start_level(3, 1) == 3
    assert benchmarks.start_level(3, config.MISSED_SESSION_DAYS) == 2


# --- therapist profile --------------------------------------------------------------------

def test_therapist_profile_defaults(tmp_path):
    t = benchmarks.load_therapist_profile(tmp_path / "missing.json")
    assert t["affected_side"] == "left" and t["mdc_setting"] == "laboratory"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"mdc_setting": "garage", "exercises": "nope", "sex_for_norms": "male"}))
    t = benchmarks.load_therapist_profile(bad)
    assert t["mdc_setting"] == "laboratory" and t["exercises"] == {} and t["sex_for_norms"] == "male"
    s = benchmarks.exercise_settings(t, "shoulder_flexion_raise")
    assert s["enabled"] is False and s["sets"] == config.ARM_DEFAULT_SETS


def test_shipped_therapist_profile():
    t = benchmarks.load_therapist_profile()
    for name in ARM_EXERCISES:
        assert name in t["exercises"], name
    assert benchmarks.elbow_deficit(t) == 0.0
    assert benchmarks.contracture_cap(t, "shoulder_flexion_raise") == 2
    assert benchmarks.contracture_cap({"contracture_over_quarter_range": {"shoulder_flexion_raise": True}},
                                      "shoulder_flexion_raise") == 1
