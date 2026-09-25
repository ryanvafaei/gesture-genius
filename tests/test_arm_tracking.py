"""
Arm tracking in real conditions: hips hidden under the table, left and
right swapped by the model, hands with wrong labels, a slow frame rate,
imperfect returns, jitter and a coach who is still talking.
"""

import numpy as np
import pytest

from rehab import body, config
from rehab.body import POSE, PoseObservation
from rehab.exercises import create
from rehab.filters import LowPassBank
from helpers import texts
from synthetic_body import IMAGE_SIZE, feat, pose
from synthetic_hand import hand
from test_arm_exercises import Driver, raise_ex, therapist

HIPS = ("left_hip", "right_hip")


def swapped(p):
    """The same pose with the model's left and right labels swapped."""
    return PoseObservation(image=p.image[body._SWAP], visibility=p.visibility[body._SWAP])


def facing(p, dx):
    """Nose (and eyes) moved dx pixels forward: side-on she faces that way."""
    img = p.image.copy()
    for name in ("nose", "left_eye", "right_eye"):
        img[POSE[name], 0] += dx / IMAGE_SIZE[0]
    return PoseObservation(image=img, visibility=p.visibility)


# --- the hips are not needed -------------------------------------------------------------------

def test_hidden_hips_do_not_hide_the_arm():
    ex = raise_ex()
    f = feat(0.0, shoulder=40, hidden=HIPS)
    assert ex.quality_problem(f) is None
    assert body.setup_problem(f, ex.required(), "sagittal", "left") is None
    assert not f.hips_visible and np.isnan(f.trunk_angle)
    # measured against the vertical: the same angle for an upright trunk
    assert f.arm["left"]["shoulder_elevation"] == pytest.approx(40, abs=0.5)


def test_hips_below_the_picture_are_not_at_the_edge():
    ex = raise_ex()
    p = pose(shoulder=5)
    p.image[POSE["left_hip"], 1] = p.image[POSE["right_hip"], 1] = 1.1     # guessed, off screen
    f = body.extract(p, 0.0, IMAGE_SIZE)
    assert not f.hips_visible
    assert body.setup_problem(f, ex.required(), "sagittal", "left") is None


def test_rep_counts_with_hidden_hips():
    d = Driver(raise_ex())
    rec = d.rep("shoulder", 2, 100, hidden=HIPS)
    assert rec is not None and rec.success and rec.raw_high == pytest.approx(100, abs=0.5)


def test_trunk_lean_is_seen_without_the_hips():
    """Hips hidden: a lean shows as the shoulders moving (the hips stay on the chair)."""
    d = Driver(raise_ex())
    d.hold(0.5, shoulder=2, hidden=HIPS)
    d.ramp("shoulder", 2, 100, 1.2, hidden=HIPS)
    d.hold(1.3, shoulder=100, trunk=20, hidden=HIPS)
    d.ramp("shoulder", 100, 2, 1.2, hidden=HIPS)
    d.hold(0.5, shoulder=2, hidden=HIPS)
    rec = d.ex.reps[-1]
    assert "trunk" in rec.compensation and rec.extra["trunk_max_change_deg"] > 15


def test_hip_gate_does_not_flicker():
    gate = body.HipGate(on=0.6, off=0.4)
    assert [gate(v) for v in (0.3, 0.55, 0.65, 0.55, 0.45, 0.35, 0.55)] == \
        [False, False, True, True, True, False, False]


def test_unseen_landmarks_give_no_measure():
    f = feat(0.0, shoulder=40, hidden=("left_wrist",))
    assert np.isnan(f.arm["left"]["elbow_flexion"])
    assert f.arm["left"]["shoulder_elevation"] == pytest.approx(40, abs=0.5)


# --- which arm is which -----------------------------------------------------------------------

def test_views_of_real_shoulders():
    """A real torso facing the camera measures about 0.55 (MediaPipe photos): that is facing."""
    assert body.estimate_view(0.55) == "frontal"
    assert body.view_accepted(0.55, "frontal") and body.view_accepted(0.42, "frontal")
    assert body.view_accepted(0.4, "oblique") and not body.view_accepted(0.1, "oblique")
    assert body.view_accepted(0.15, "sagittal") and not body.view_accepted(0.5, "sagittal")


def test_swapped_labels_facing_the_camera_are_put_back():
    p = pose(shoulder=60, view="frontal", other_shoulder=5)
    f = body.extract(swapped(p), 0.0, IMAGE_SIZE)
    assert f.sides_swapped
    assert f.arm["left"]["shoulder_elevation"] == pytest.approx(60, abs=0.5)
    assert not body.extract(p, 0.0, IMAGE_SIZE).sides_swapped


def test_swapped_labels_side_on_are_put_back():
    """Facing the right of the mirrored picture, her left side is nearest the camera."""
    p = facing(pose(shoulder=60, other_shoulder=5), 40.0)
    f = body.extract(p, 0.0, IMAGE_SIZE)
    assert not f.sides_swapped and body.near_side(f) == "left"
    f = body.extract(swapped(p), 0.0, IMAGE_SIZE)
    assert f.sides_swapped and body.near_side(f) == "left"
    assert f.arm["left"]["shoulder_elevation"] == pytest.approx(60, abs=0.5)


def test_setup_asks_to_turn_when_the_other_arm_is_nearest():
    ex = raise_ex()
    # facing the left of the mirrored picture: her right side is nearest
    p = facing(pose(side="right"), -40.0)
    f = body.extract(p, 0.0, IMAGE_SIZE)
    assert body.near_side(f) == "right"
    assert body.setup_problem(f, ex.required(), "sagittal", "left") == "turn_side"
    f = body.extract(facing(pose(side="left"), 40.0), 0.0, IMAGE_SIZE)
    assert body.setup_problem(f, ex.required(), "sagittal", "left") is None


def test_setup_accepts_a_real_frontal_view():
    """Shoulders 0.5 torso lengths apart (a real person facing the camera) pass."""
    ex = create("finger_to_nose_timed", therapist=therapist())
    p = pose(view="frontal")
    f = body.extract(p, 0.0, IMAGE_SIZE)
    # narrow the shoulders to 125 px on a 250 px torso
    for s, sign in (("left", -1), ("right", 1)):
        f.points[POSE[f"{s}_shoulder"]][0] = f.shoulder_mid[0] + sign * 62.5
    f.shoulder_width = 125.0
    f.view_ratio = 0.5
    assert body.setup_problem(f, ex.required(), "frontal", "right") is None


def test_hands_go_to_the_arm_they_are_at():
    """A hand labelled "Right" at her left wrist is her left hand."""
    f = feat(0.0, elbow=90)
    wrist = f.point("left_wrist") / np.array(IMAGE_SIZE)
    obs = hand()
    img = obs.image.copy()
    img[:, :2] += wrist - img[0, :2]
    wrong = obs.__class__(**{**vars(obs), "image": img, "handedness": "Right"})
    got = body.assign_hands([wrong], f)
    assert list(got) == ["left"] and got["left"].handedness == "Left"
    # a hand far from both wrists belongs to neither arm
    far = obs.__class__(**{**vars(obs), "image": img + [0.0, -0.9, 0.0]})
    assert body.assign_hands([far], f) == {}


def test_wrist_extension_with_a_mislabelled_hand():
    p = pose(elbow=90)
    f = body.extract(p, 0.0, IMAGE_SIZE)
    from synthetic_body import hand_for_wrist
    h = hand_for_wrist(p, "left", 30)
    h.handedness = "Right"
    body.attach_hands(f, {"left": h})
    assert f.arm["left"]["wrist_extension"] == pytest.approx(30, abs=0.5)


# --- the rep machine ------------------------------------------------------------------------------

def test_rep_ends_without_the_exact_start_angle():
    """Back most of the way (not within the tolerance of the start): the rep is over."""
    cal = {"right": {"best": 140.0, "start": 60.0}, "left": {"best": 110.0, "start": 60.0}}
    ex = create("hand_to_mouth", {"hand_to_mouth": cal}, level=0, therapist=therapist())
    ex.start_set(1, 0.0)
    d = Driver(ex)
    d.hold(0.5, elbow=60)
    d.ramp("elbow", 60, 115, 1.2)
    d.hold(1.3, elbow=115)
    d.ramp("elbow", 115, 72, 1.0)          # 12 degrees short of where she started
    d.hold(0.5, elbow=72)
    assert len(ex.reps) == 1 and ex.reps[0].success


def test_jitter_is_not_a_rep():
    ex = raise_ex()
    d = Driver(ex)
    d.hold(0.5, shoulder=5)
    for _ in range(6):
        d.ramp("shoulder", 5, 14, 0.3)
        d.ramp("shoulder", 14, 5, 0.3)
    d.hold(0.5, shoulder=5)
    assert ex.reps == []
    assert "That's one." not in texts(d.said)


def test_a_pause_on_the_way_up_is_one_rep():
    """She stops and dips a little, then goes on up to the target: one rep, with its hold."""
    ex = raise_ex()
    d = Driver(ex)
    d.hold(0.5, shoulder=2)
    d.ramp("shoulder", 2, 60, 1.0)
    d.ramp("shoulder", 60, 48, 0.4)
    d.ramp("shoulder", 48, 100, 1.0)
    d.hold(1.3, shoulder=100)
    d.ramp("shoulder", 100, 2, 1.2)
    d.hold(0.5, shoulder=2)
    assert len(ex.reps) == 1 and ex.reps[0].success
    assert "And hold." in texts(d.said)


def test_hold_survives_small_dips():
    ex = raise_ex()
    d = Driver(ex)
    d.hold(0.5, shoulder=2)
    d.ramp("shoulder", 2, 80, 1.2)
    target = ex.target_deg
    for _ in range(3):                      # a wobble of 1.5 tolerances below the target
        d.hold(0.3, shoulder=target)
        d.hold(0.1, shoulder=target - 1.5 * ex.tolerance)
    assert ex.state in ("hold", "returning")
    assert texts(d.said).count("And hold.") == 1


def test_next_prompt_waits_for_the_coach():
    """The rep count and cue are heard in full; "And again." comes when the coach is quiet."""
    ex = raise_ex()
    d = Driver(ex)
    d.hold(0.5, shoulder=2)
    d.ramp("shoulder", 2, 100, 1.2)
    d.hold(1.3, shoulder=100)
    n = len(d.said)
    ex.speaking = True                      # "And slowly down.", then the count, still being said
    d.ramp("shoulder", 100, 2, 1.2)
    d.hold(1.0, shoulder=2)
    assert "That's one." in texts(d.said[n:])
    assert "And again." not in texts(d.said[n:]) and ex.state == "ready"
    ex.speaking = False
    d.hold(0.2, shoulder=2)
    assert texts(d.said[n:]).count("And again.") == 1
    d.hold(1.0, shoulder=2)
    assert texts(d.said[n:]).count("And again.") == 1       # said once per rep


def test_no_prompt_when_she_already_started():
    ex = raise_ex()
    d = Driver(ex)
    d.hold(0.5, shoulder=2)
    d.ramp("shoulder", 2, 100, 1.2)
    d.hold(1.3, shoulder=100)
    n = len(d.said)
    ex.speaking = True
    d.ramp("shoulder", 100, 2, 1.2)
    d.hold(0.4, shoulder=2)
    d.ramp("shoulder", 2, 60, 0.8)
    ex.speaking = False
    d.ramp("shoulder", 60, 100, 0.8)
    assert "That's one." in texts(d.said[n:]) and "And again." not in texts(d.said[n:])
    assert ex.state in ("moving", "hold")


def test_and_hold_is_an_instruction():
    """ "And hold." is not a count: a busy speaker must not drop it."""
    d = Driver(raise_ex())
    d.rep("shoulder", 2, 100)
    hold = next(m for m in d.said if m.text == "And hold.")
    assert hold.kind == "instruction" and not hold.ephemeral


def test_finger_to_nose_lap_needs_no_hips():
    ex = create("finger_to_nose_timed", therapist=therapist())
    ex.start_set(1, 0.0)
    f = feat(0.0, side="right", view="frontal", shoulder=5, elbow=60, hidden=HIPS)
    assert ex.quality_problem(f) is None and ex._in_lap(f)
    f = feat(0.0, side="right", view="frontal", shoulder=60, elbow=120, hidden=HIPS)
    assert not ex._in_lap(f)


# --- filtering ----------------------------------------------------------------------------------

@pytest.mark.parametrize("fps", [10.0, 30.0])
def test_low_pass_keeps_its_cutoff_at_any_frame_rate(fps):
    """A step is followed within about 0.1 s at 10 fps as at 30 fps (the filter used to assume 30)."""
    bank = LowPassBank(fs=30.0, fc=config.LOW_PASS_HZ)
    t, out = 0.0, []
    for i in range(int(fps)):
        t += 1 / fps
        out.append((t, bank("x", 0.0 if t < 0.3 else 100.0, t)))
    after = [v for tt, v in out if tt >= 0.3 + 0.12]
    assert after[0] > 75
