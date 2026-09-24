"""The exercises, by name: six hand exercises, bubble pinch, two-hand match and memory pairs."""

from rehab.exercises.bubble_pinch import BubblePinch
from rehab.exercises.finger_abduction import FingerAbduction
from rehab.exercises.finger_tapping import FingerTapping
from rehab.exercises.grip_release import GripRelease
from rehab.exercises.grip_squeeze import GripSqueeze
from rehab.exercises.memory_pairs import MemoryPairs
from rehab.exercises.thumb_flexion import ThumbFlexion
from rehab.exercises.thumb_opposition import ThumbOpposition
from rehab.exercises.two_hand_match import TwoHandMatch

EXERCISES = {cls.name: cls for cls in (
    GripRelease, FingerAbduction, ThumbFlexion,
    ThumbOpposition, FingerTapping, GripSqueeze,
    BubblePinch, TwoHandMatch, MemoryPairs,
)}


def create(name, calibrations=None, thresholds=None, params=None, **kwargs):
    """
    Build an exercise with its calibration.

    calibrations  {exercise name: {step: {key: value}}} for all exercises;
                  some exercises also use another one's calibration.
    """
    calibrations = calibrations or {}
    cls = EXERCISES[name]
    if cls is FingerAbduction:
        kwargs.setdefault("grip_calibration", calibrations.get("grip_release"))
    return cls(params=params, calibration=calibrations.get(name),
               thresholds=(thresholds or {}).get(name), **kwargs)
