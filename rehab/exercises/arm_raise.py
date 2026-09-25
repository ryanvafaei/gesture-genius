"""
Arm raises: to the front (FMA items 13 and 16) and to the side (FMA 15).

Measure: shoulder elevation, angle(hip, shoulder, elbow) in pixels (Lazem
2026). Seen side-on for the front raise, facing the camera for the side
raise. The elbow stays straight, the arm stays in its plane, the trunk and
(side raise) the shoulder stay down.

Targets climb from her own baseline towards the height her right arm
reaches, capped at 160 degrees of active flexion (Gill 2020, via the FMA
manual). On the way she passes the heights daily tasks need (Gates 2016):
71 degrees to drink from a cup, 86 for a can on a 1.48 m shelf, 90 for FMA
item 13, 105 and 108 for a shelf at head height.
"""

from rehab import benchmarks
from rehab.benchmarks import finite
from rehab.exercises.arm import ArmExercise

CONTRACTURE_NOT_TESTABLE_DEG = 30     # FMA 2026: elbow deficit of 30 degrees or more


class _StraightArmRaise(ArmExercise):
    metric = "shoulder_elevation"
    metric_label = "Arm height"
    direction = "increase"
    form_rules = ("elbow_straight", "in_plane")
    # lifted high, the hand may leave the top of the picture: the shoulder
    # angle needs only the shoulder and the elbow (the elbow rule is judged
    # while the wrist is seen)
    track_joints = ("shoulder", "elbow")
    start_prompt = "Let your arm hang down by your side, with your elbow straight."
    return_prompt = "And slowly down."

    def elbow_allowance(self):
        """
        Start elbow flexion allowed: the tolerance, or a therapist-entered
        contracture deficit under 30 degrees (FMA: then that is her full
        extension).
        """
        deficit = benchmarks.elbow_deficit(self.therapist, self.side)
        if deficit >= CONTRACTURE_NOT_TESTABLE_DEG:
            deficit = 0.0
        return max(self.elbow_tol, deficit)

    def start_ok(self, refs):
        # an elbow not seen at the start (wrist out of the picture) gets the benefit of the doubt
        return not finite(refs["elbow"]) or refs["elbow"] <= self.elbow_allowance()

    @classmethod
    def not_testable(cls, therapist):
        side = therapist.get("affected_side", "left")
        return benchmarks.elbow_deficit(therapist, side) >= CONTRACTURE_NOT_TESTABLE_DEG


class ShoulderFlexionRaise(_StraightArmRaise):
    name = "shoulder_flexion_raise"
    title = "Lift your arm to the front"
    instructions = (
        "Let's lift your arm to the front.",
        "Sit side-on to the screen, with your {side} arm nearest to it.",
        "Keep your elbow straight and your back against the chair.",
    )
    view = "sagittal"
    fma_item = "13"
    fma_threshold = 90.0
    milestone_metric = "shoulder_elevation"
    move_prompt = "Now lift your arm to the front, as high as is comfortable."
    calibration_screen = "Lift to the front"
    progress_phrase = "Your arm lifted {deg} degrees higher than {when}."

    def normative_ceiling(self):
        return self.bench.active_shoulder_flexion

    def rep_extra(self, result, rep):
        """FMA item 16 (90-180 degrees): full = within the tolerance of her right arm's best."""
        above = result.peak - self.fma_threshold
        ceiling = self.ladder.ceiling if self.ladder else None
        if result.fma_style_score == 0 or above <= self.tolerance:
            fma16 = 0
        elif (result.fma_style_score == 2 and ceiling is not None
              and result.peak >= ceiling - self.tolerance):
            fma16 = 2
        else:
            fma16 = 1
        return {"fma16_style": min(fma16, self.max_score)}


class ShoulderAbductionRaise(_StraightArmRaise):
    name = "shoulder_abduction_raise"
    title = "Lift your arm to the side"
    instructions = (
        "Let's lift your arm out to the side.",
        "Sit facing the screen, so I can see both shoulders.",
        "Keep your elbow straight and your shoulder relaxed.",
    )
    view = "frontal"
    joints = ("shoulder", "elbow", "wrist", "ear")
    both_sides = ("shoulder",)
    fma_item = "15"
    fma_threshold = 90.0
    form_rules = ("elbow_straight", "in_plane", "no_shoulder_hike")
    move_prompt = "Now lift your arm out to the side, as high as is comfortable."
    calibration_screen = "Lift to the side"
    progress_phrase = "Your arm lifted {deg} degrees higher than {when}."
