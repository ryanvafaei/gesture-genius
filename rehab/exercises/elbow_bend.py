"""
Elbow exercises, seen side-on: hand to mouth, hand to head (FMA item 7)
and straightening the elbow (FMA item 10).

Measure: elbow flexion, 180 - angle(shoulder, elbow, wrist), 0 = straight
(Lazem 2026, Jayavel 2025). The trunk stays against the chair.

Hand to mouth: Gates 2016 found drinking from a cup needs about 121 degrees
of elbow flexion (and 71 of shoulder elevation, logged here). An empty cup
only, until swallowing advice allows food or drink.
Hand to head: done when the wrist reaches the ear (FMA 7: "hand touches the
ear"), within HAND_TO_EAR_TORSO torso lengths (proposed).
Elbow extension: FMA 10 asks for 0 degrees. A 2D angle is never below 0, so
reaching 0 within the tolerance counts (at_scale_end). A contracture of less
than 30 degrees entered by the therapist is her full extension; 30 or more
means the item is not testable (FMA 2026).

The elbow's minimum detectable change is large (15-33 degrees, Lazem 2026),
so improvement is only claimed from the weekly assessment (plan rule T5).
"""

from dataclasses import replace

from rehab import benchmarks, config
from rehab.benchmarks import finite
from rehab.exercises.arm import ArmExercise

CONTRACTURE_NOT_TESTABLE_DEG = 30


class _ElbowExercise(ArmExercise):
    metric = "elbow_flexion"
    metric_label = "Elbow bend"
    view = "sagittal"
    claims_from_assessment_only = True
    compensation_rules = ("trunk",)

    def display_range(self):
        return 0.0, 160.0


class HandToMouth(_ElbowExercise):
    name = "hand_to_mouth"
    title = "Hand to your mouth"
    instructions = (
        "Let's bring your hand to your mouth, as if you drink a cup of tea.",
        "Use an empty cup, or no cup at all.",
        "Sit side-on to the screen, with your {side} arm nearest to it.",
    )
    direction = "increase"
    milestone_metric = "elbow_flexion"
    secondary_metrics = ("shoulder_elevation",)
    start_prompt = "Rest your hand in your lap."
    move_prompt = "Now bring your hand up to your mouth."
    return_prompt = "And back down to your lap."
    calibration_screen = "Hand to mouth"
    progress_phrase = "Your elbow bent {deg} degrees further than {when}."

    def in_start(self, f, arm, value):
        return arm.get("wrist_drop", 0.0) >= config.ARM_HAND_DOWN_TORSO

    def normative_ceiling(self):
        return self.bench.normative("elbow_flexion", self.sex)["mean_deg"]


class HandToHead(_ElbowExercise):
    name = "hand_to_head"
    title = "Hand to your head"
    instructions = (
        "Let's bring your hand up to your ear, as if you brush your hair.",
        "Sit side-on to the screen, with your {side} arm nearest to it.",
    )
    direction = "increase"
    joints = ("shoulder", "elbow", "wrist", "ear")
    track_joints = ("shoulder", "elbow", "wrist")      # the ear only scores the rep
    fma_item = "07"
    secondary_metrics = ("shoulder_elevation",)
    start_prompt = "Rest your hand in your lap."
    move_prompt = "Now bring your hand up to your ear."
    return_prompt = "And back down to your lap."
    calibration_screen = "Hand to ear"
    progress_phrase = "Your elbow bent {deg} degrees further than {when}."

    def in_start(self, f, arm, value):
        return arm.get("wrist_drop", 0.0) >= config.ARM_HAND_DOWN_TORSO

    def adjust_result(self, result, rep):
        """FMA 7: 2 when the hand reached the ear with no form break or compensation."""
        at_ear = finite(rep["wrist_to_ear_min"]) and rep["wrist_to_ear_min"] <= config.HAND_TO_EAR_TORSO
        if result.fma_style_score >= 1 and at_ear and not result.broken and not result.compensations:
            return replace(result, fma_style_score=min(2, self.max_score))
        return replace(result, fma_style_score=min(result.fma_style_score, 1),
                       reasons=result.reasons + ([] if at_ear else ["hand_not_at_ear"]))

    def rep_extra(self, result, rep):
        return {"wrist_to_ear_min": rep["wrist_to_ear_min"]}


class ElbowExtension(_ElbowExercise):
    name = "elbow_extension"
    title = "Straighten your elbow"
    instructions = (
        "Let's straighten your elbow.",
        "Sit side-on to the screen, with your {side} arm nearest to it.",
    )
    direction = "decrease"
    fma_item = "10"
    at_scale_end = True
    start_prompt = "Rest your hand on your lap, with your elbow bent."
    move_prompt = "Now straighten your arm, reaching down past your knee."
    return_prompt = "And bend it back to your lap."
    calibration_screen = "Straighten the elbow"
    progress_phrase = "Your elbow straightened {deg} degrees more than {when}."

    @property
    def fma_threshold(self):
        """0 degrees, or her contracture deficit when it is under 30 degrees (FMA 2026)."""
        deficit = benchmarks.elbow_deficit(self.therapist, self.side)
        return float(deficit) if 0 < deficit < CONTRACTURE_NOT_TESTABLE_DEG else 0.0

    @classmethod
    def not_testable(cls, therapist):
        side = therapist.get("affected_side", "left")
        return benchmarks.elbow_deficit(therapist, side) >= CONTRACTURE_NOT_TESTABLE_DEG

    def in_start(self, f, arm, value):
        return value >= config.ARM_BENT_ELBOW_MIN

    def normative_ceiling(self):
        return self.fma_threshold

    def therapist_limit(self):
        """An extension deficit, not the flexion limit, caps how straight she can get."""
        deficit = benchmarks.elbow_deficit(self.therapist, self.side)
        return deficit if deficit > 0 else None

    def milestones(self):
        return [{"deg": self.fma_threshold, "task": "FMA 10", "source": "FMA2026", "kind": "fma"}]
