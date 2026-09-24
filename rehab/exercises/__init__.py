"""The six hand exercises and the eight arm exercises, by name."""

from rehab.exercises.arm_raise import ShoulderAbductionRaise, ShoulderFlexionRaise
from rehab.exercises.elbow_bend import ElbowExtension, HandToHead, HandToMouth
from rehab.exercises.finger_abduction import FingerAbduction
from rehab.exercises.finger_tapping import FingerTapping
from rehab.exercises.finger_to_nose import FingerToNose
from rehab.exercises.grip_release import GripRelease
from rehab.exercises.grip_squeeze import GripSqueeze
from rehab.exercises.tabletop_reach import TabletopReach
from rehab.exercises.thumb_flexion import ThumbFlexion
from rehab.exercises.thumb_opposition import ThumbOpposition
from rehab.exercises.wrist_extension import WristExtension

HAND_EXERCISES = {cls.name: cls for cls in (
    GripRelease, FingerAbduction, ThumbFlexion,
    ThumbOpposition, FingerTapping, GripSqueeze,
)}
# body tracking (MediaPipe Pose); see rehab/exercises/arm.py
ARM_EXERCISES = {cls.name: cls for cls in (
    ShoulderFlexionRaise, ShoulderAbductionRaise, HandToMouth, HandToHead,
    ElbowExtension, WristExtension, TabletopReach, FingerToNose,
)}
EXERCISES = {**HAND_EXERCISES, **ARM_EXERCISES}


def is_arm(name):
    return name in ARM_EXERCISES


def create(name, calibrations=None, thresholds=None, params=None, **kwargs):
    """
    Build an exercise with its calibration.

    calibrations  {exercise name: {step: {key: value}}} for all exercises;
                  some exercises also use another one's calibration.
    kwargs        arm exercises: level, therapist; thumb opposition: level
    """
    calibrations = calibrations or {}
    cls = EXERCISES[name]
    if cls is FingerAbduction:
        kwargs.setdefault("grip_calibration", calibrations.get("grip_release"))
    return cls(params=params, calibration=calibrations.get(name),
               thresholds=(thresholds or {}).get(name), **kwargs)
