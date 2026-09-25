"""
Wrist up (FMA items 19/21 and 20/22).

Measure: the hand (hand landmarks wrist -> middle MCP) against the forearm
(pose elbow -> wrist), seen from the side, positive = hand lifted
(Jayavel 2025 combined BlazePose and BlazeHand the same way). The forearm
rests on the table, palm down, hand over the edge: the forearm is then
level, so "up" in the picture is dorsal extension.

FMA 19/21 ask for 15 degrees of extension held against resistance; a webcam
cannot feel resistance, so the FMA-style score is at most 1. Each rep up
and down is one wrist cycle (FMA 20/22 ask for 2 full cycles). Milestones:
15 degrees (FMA), 33 to drink from a cup and 40 for all eight daily tasks
(Gates 2016). No minimum detectable change was published for the wrist:
10 degrees is the provisional change threshold.
"""

from rehab.exercises.arm import ArmExercise
from rehab import config


class WristExtension(ArmExercise):
    name = "wrist_extension"
    title = "Lift your hand at the wrist"
    instructions = (
        "Let's lift your hand at the wrist.",
        "Rest your {side} forearm on the table, palm down, with your hand over the edge.",
        "Sit side-on to the screen, with that arm nearest to it.",
    )
    metric = "wrist_extension"
    metric_label = "Wrist lift"
    direction = "increase"
    view = "sagittal"
    joints = ("shoulder", "elbow", "wrist")
    track_joints = ("elbow", "wrist")
    needs_hand = True
    fma_item = "19"
    fma_threshold = 15.0
    fma_max_score = 1                 # resistance cannot be measured
    form_rules = ("elbow_steady",)
    compensation_rules = ()
    start_prompt = "Let your hand rest down over the edge of the table."
    move_prompt = "Now lift your hand up at the wrist."
    return_prompt = "And let it down again."
    calibration_screen = "Lift the hand"
    progress_phrase = "Your wrist lifted {deg} degrees higher than {when}."

    def in_start(self, f, arm, value):
        return value <= config.ARM_WRIST_START_MAX

    def milestones(self):
        out = []
        for m in self.bench.exercise(self.name).get("milestones_deg", []):
            task = m.get("task") or f"FMA {self.fma_item}"
            out.append({"deg": float(m["deg"]), "task": task, "source": m.get("source", ""),
                        "kind": "task" if m.get("task") else "fma"})
        return out

    def display_range(self):
        return -30.0, 80.0

    def rep_extra(self, result, rep):
        return {"wrist_cycle": 1 if result.moved >= self.tolerance else 0}
