# Eleanor Rehab Coach: Benchmark Implementation Plan

This plan covers how the Think module decides whether a movement was good enough, how that decision is backed by published clinical and normative data, and how to build it on top of the existing Sense / Think / Act code.

Every number in this plan and in `benchmarks.json` is either tagged with a source ID (see §16) or marked **proposed**, meaning it's a project default that isn't taken from any of the seven sources. The proposed values are collected in §14 so they can be tuned during testing.

---

## 0. Package contents

| File | What it is | Used by |
|---|---|---|
| `benchmarks.json` | Single machine-readable config: sources, landmarks, angle definitions, tolerances, all 3 layers, scoring rules, 10 exercise profiles | Think (load once at start) |
| `data/fma_ue_2026_items.csv` | All 33 FMA-UE items: 0/1/2 criteria, webcam feasibility, signals, max webcam score | Layer 1 |
| `data/fma_ue_2026_general_rules.csv` | FMA general rules translated into engine rules | Layer 1, scoring |
| `data/soucie2011_normative_rom.csv` | Normal passive ROM, 11 motions × 2 sexes × 4 age groups, mean/CI/SD/percentiles | Layer 2a |
| `data/gates2016_adl_peak_angles.csv` | Peak shoulder/elbow/forearm/wrist angles for 8 daily tasks | Layer 2b |
| `data/gates2016_adl_minimum_rom_summary.csv` | Minimum ROM to complete all 8 tasks; trunk ROM | Layer 2b, compensation |
| `data/bain2015_finger_rom.csv` | Active and functional finger-joint ROM (MCP/PIP/DIP), pre-grasp/grasp, per-finger arcs | Layer 2b |
| `data/rps_levin2004.csv` + `rps_levin2004_numeric_anchors.csv` | Reaching Performance Scale components and numeric anchors | Layer 2c |
| `data/measurement_error.csv` | MediaPipe limits of agreement, ICC, SEM, MDC (Lazem 2026, Jayavel 2025), goniometer inter-rater error (Soucie 2011) | Tolerance |
| `data/sources.csv` | Full citations, DOIs, what each source is used for, its limits | Report |
| `engine/benchmark_engine.py` | Reference implementation: angles, filter, calibration, target ladder, rep scoring, change detection, FMA-33 timing, RPS trunk score | Sense + Think |
| `engine/test_benchmark_engine.py` | 10 tests on synthetic data (all passing) | QA |
| `tools/build_benchmarks_json.py` | Rebuilds `benchmarks.json` from the CSVs after edits | Maintenance |
| `templates/*.csv`, `templates/therapist_profile.json` | Rep log, session log, therapist settings | Logging, config |

---

## 1. The model

```
                    ┌───────────────── Measurement tolerance (±5–9° per joint/task; MDC 11–33° between sessions) ─────────────────┐
Sense (MediaPipe) → angles/metrics → │ Layer 1  FMA-UE form rules      → "was it done the right way?"       (0/1/2 per rep)       │ → Think decision → Act feedback
                                     │ Layer 2  Normative + functional → "how far is far enough?"           (ceilings, milestones)│
                                     │ Layer 3  Personal baseline      → "what is Eleanor's target today?"  (calibrated ladder)   │
                                     └──────────────────────────────────────────────────────────────────────────────────────────┘
```

- **Layer 1: form.** FMA-UE defines what counts as a correct movement for each item: start position, end position, what must stay still, and which compensations are banned. It supplies the 0/1/2 scoring logic and the fixed clinical thresholds: 90° shoulder flexion and abduction, 0° elbow extension, 15° wrist extension, 2 full wrist cycles, and the 2 s / 6 s time rule.
- **Layer 2: range.** Three kinds of reference:
  - normal ROM (Soucie), used as an upper ceiling and to catch tracking errors
  - ROM needed for daily tasks (Gates for the arm, Bain for the fingers), used as named milestones tied to Eleanor's goals
  - compensation limits (RPS, Gates trunk data)
- **Layer 3: personal.** Calibrate the unaffected right side first, then the affected left side. Targets climb from her own baseline toward min(right side, norms, therapist limit), in steps no smaller than the measurement tolerance.
- **Tolerance** ties the layers together:
  - Within a session, differences smaller than the 95% limits of agreement are treated as noise.
  - Between sessions, the system only says "you improved" when the change is at least the minimum detectable change (MDC).

---

## 2. What each source can and cannot support

| ID | Gives you | Population / method | Do not use it for |
|---|---|---|---|
| FMA2026 | Item criteria, start/end positions, banned compensations, scoring rules, time rule | Standard stroke motor assessment (0–66) | A clinical FMA score from a webcam. Items needing resistance, reflexes or objects can't be fully scored |
| SOUCIE2011 | Normal ROM by sex and age, including percentiles | 674 healthy people aged 2–69; **passive** ROM measured by goniometer (shoulder and elbow measured supine) | Targets. Stroke active ROM is lower, and Eleanor is 72 (the oldest group is 45–69) |
| GATES2016 | Peak angles used in 8 daily tasks, trunk motion in healthy people | 15 healthy adults aged ~26, right arm, 3D motion capture | Exact comparison with 2D webcam angles. Treat values as milestones |
| BAIN2015 | Finger ROM needed for 90% of daily hand tasks | 10 healthy adults aged 18–53, electrogoniometer | Claims about MediaPipe finger accuracy (not validated) |
| LEVIN2004 | Reach compensation rules: trunk share of the reach, shoulder/elbow contribution; reach setup | 28 people with hemiparesis; reliability only preliminary | Precise cut-offs. Some score levels are verbal only |
| LAZEM2026 | Tolerance (limits of agreement) and MDC for MediaPipe shoulder/elbow angles **in stroke**; camera setup | 27 chronic stroke survivors; compared against Kinovea | Treating Kinovea as ground truth. Home recording was hard (13/27 usable) |
| JAYAVEL2025 | MediaPipe vs goniometer agreement for elbow and wrist; landmark choice for the wrist | 10 hemiplegic patients | Strong claims (small sample; the paper's text and Table 1 disagree, and Table 1 is used here) |

---

## 3. Sense: measurement setup and angle definitions

### 3.1 Camera and body setup

These are the conditions the tolerance numbers were validated under:

| Item | Setting | Source |
|---|---|---|
| Camera distance / height | 1.5 m from the chair, lens ~90 cm high, perpendicular to the movement plane | LAZEM2026 |
| View per task | **Frontal** for abduction, finger-to-nose, shoulder hike, lateral lean. **Sagittal** (side-on, affected arm closest) for forward raise, hand-to-mouth, hand-to-head, elbow extension, trunk flexion. **45°** between frontal and sagittal on the affected side for the tabletop reach | LAZEM2026, LEVIN2004 |
| Resolution / light | ≥ 1280×720, even white light | JAYAVEL2025 |
| Clothing | Light, sleeveless or short-sleeved (full sleeves reduced accuracy) | LAZEM2026, JAYAVEL2025 |
| Seat | FMA standard is a chair without armrests. An armchair is allowed if it's logged, which you'll want for Eleanor's fall risk. Log `seat_type` every session | FMA2026 |
| Reach setup | Seat 42 cm with backrest, no armrests; table 72 cm; chair at arm's length so the wrist crease sits at a mark 4 cm from the table edge; targets 1 cm (close) and 30 cm (far) from the edge | LEVIN2004 |
| Framing check | Show a live guide before each session ("move back a little"). Lazem found that without guidance, most self-recorded home videos were unusable | LAZEM2026 (recommendation) |

A laptop webcam is naturally frontal. For sagittal tasks, turn the chair side-on and check that hip, shoulder, elbow and wrist are all visible before starting.

### 3.2 Landmarks

**Pose landmarks:**
- nose 0, eyes 2/5, ears 7/8
- shoulders 11 (L), 12 (R); elbows 13/14; wrists 15/16
- hips 23/24; knees 25/26

Pose left/right means the person's own side. **Eleanor's affected side is her left, so use 11/13/15/23.**

**Hand landmarks:**
- wrist 0
- thumb 1–4
- index 5–8, middle 9–12, ring 13–16, little 17–20

Check the hand "handedness" label during calibration. It depends on whether the frame is mirrored.

### 3.3 Angle definitions

These are in `benchmarks.json → angle_definitions`.

| Metric | Formula | Space | Aligns with |
|---|---|---|---|
| `shoulder_elevation` | angle(hip, shoulder, elbow) | 2D **pixels** | LAZEM2026 exactly; approximates GATES2016 humeral elevation |
| `elbow_flexion` | 180 − angle(shoulder, elbow, wrist); 0 = straight | 2D pixels | LAZEM2026, JAYAVEL2025, clinical convention |
| `wrist_extension` | signed angle(pose elbow→wrist, hand 0→9); sign fixed at calibration so "hand up" is positive | 2D pixels, side view | JAYAVEL2025 (BlazePose + BlazeHand) |
| `finger_flexion` | 180 − angle(prev, joint, next): MCP (0,5,6)…, PIP (5,6,7)…, DIP (6,7,8)…; thumb MCP (1,2,3), IP (2,3,4) | 3D hand world landmarks | BAIN2015 joints (0 = straight; hyperextension can't be seen) |
| `hand_aperture` | mean tip-to-wrist distance / palm size (0→9) | 3D | proposed normalised metric |
| `fingertip_to_palm` | tip-to-palm-centre distance / palm size | 3D | FMA item 24 "fingertips touch palm" |
| `pinch_gap` | distance(4, 8) / palm size | 3D | FMA item 28 opposition |
| `trunk_flexion` / `trunk_lateral_lean` | hip-mid → shoulder-mid vs vertical, as change from start | 2D | FMA "no trunk compensation" |
| `shoulder_girdle_elevation` | change in ear-to-shoulder vertical gap / shoulder width | 2D frontal | FMA item 15; RPS "scapular elevation" |
| `upper_arm_in_plane_ratio` | 2D upper-arm length / length at start (arm at side) | 2D | proposed check for "without abduction" (0.87 = ~30° out of the camera plane) |
| `trunk_share_of_reach` | shoulder-mid forward displacement / wrist forward displacement | 2D | RPS trunk component |
| `nose_touch_error` | index tip to nose distance / distance between the eyes | 2D | FMA item 32 dysmetria |

### 3.4 Signal processing rules

1. **Convert to pixels first.** Multiply normalised x by the frame width and y by the frame height before computing any 2D angle. A 45° arm measured in normalised 16:9 coordinates comes out more than 5° wrong (`test_pixel_scaling_matters`).
2. **Low-pass filter at 6 Hz.** Gates filtered marker data with a 6 Hz Butterworth. `LowPass(fs=30, fc=6)` is a causal real-time version.
3. **Visibility gate (proposed).** Ignore a frame if any required landmark has visibility < 0.5. If there are no valid frames for 1 s, pause and say "I can't see your arm. Please move into the box." This puts the blame on the system, not the user.
4. **Debounce (proposed).** A form or compensation flag counts only if it holds for ≥ 5 consecutive frames.
5. **Plausibility check.** Reject any angle above the Soucie group maximum + tolerance (e.g., shoulder flexion > 188° for women 45–69) as a tracking error, never as a success.

---

## 4. Measurement tolerance and detectable change

Stored in `benchmarks.json → measurement_tolerance`, built from `data/measurement_error.csv`. The table covers the affected side.

| Task · joint | 95% LoA lab (°) | 95% LoA home (°) | **Tolerance used (°)** | MDC lab (°) | MDC home (°) |
|---|---|---|---|---|---|
| Abduction · shoulder | −4.06 to 6.41 | −6.21 to 3.62 | **6.5** | 11.41 | 8.95 |
| Forward elevation · shoulder | −5.07 to 5.05 | −2.66 to 3.45 | **5.5** | 13.19 | 6.35 |
| Forward elevation · elbow | −8.78 to 6.29 | −3.99 to 1.47 | **9.0** | 14.77 | 36.24 |
| Hand to head · shoulder | −3.18 to 3.22 | −3.54 to 3.09 | **5.0** | 18.19 | 12.54 |
| Hand to head · elbow | −2.49 to 2.33 | −3.79 to 2.53 | **5.0** | 32.65 | 16.48 |
| Hand to mouth · shoulder | −3.31 to 2.84 | −2.95 to 2.41 | **5.0** | 11.65 | 11.92 |
| Hand to mouth · elbow | −5.35 to 3.81 | −4.06 to 2.05 | **5.5** | 20.55 | 25.28 |
| Wrist flex/ext (goniometer) | −1.98 to 0.92 (JAYAVEL) | — | **5.0** | none reported | provisional 10 |
| Finger joints | not validated | — | **10 (proposed)** | — | provisional 20 (proposed) |
| Trunk | — | — | 5.0; compensation cue at > 10° change | — | — |

**How the tolerance was chosen:** the larger of (a) 5°, because trained therapists with goniometers differed by up to 5° (SOUCIE2011), and (b) the worst absolute 95% LoA on the affected side, lab or home, rounded up to 0.5° (LAZEM2026).

**Rules:**

| Rule | What it means |
|---|---|
| T1: within-rep decisions | "Reached target", "elbow stayed straight" and similar checks are all judged with ± tolerance. |
| T2: uncertainty band | If a peak is within ± tolerance of an FMA threshold, the logged FMA-style score takes the **lower** level (FMA2026 rule: "a lower score should be selected if uncertainty exists"). |
| T3: level steps | A target step is never smaller than the tolerance, because a smaller step can't be detected. |
| T4: progress claims | Only say "your arm lifts higher than last week" when the change in session best is at least the MDC (LAZEM2026). Otherwise praise effort and consistency ("You did all ten, well done"). Default is the **lab MDC** when the guided setup is followed; `therapist_profile.json → mdc_setting` switches to home. |
| T5: elbow task MDC | Elbow MDCs (15–33°) are large. For elbow tasks, weekly assessment-mode values are a better basis for claims than day-to-day values. |

---

## 5. Layer 1: FMA-UE form rules

### 5.1 General FMA rules and how the engine applies them

From `data/fma_ue_2026_general_rules.csv`:

| FMA rule | Engine behaviour |
|---|---|
| Standard seat: chair without armrests; alternatives allowed if documented | Log `seat_type`; flag non-standard sessions |
| Demonstrate first; physical guidance for understanding allowed | Demo clip + one spoken line before the first rep |
| 1–3 repetitions, best performance counts | **Assessment mode**: 3 attempts, item score = best |
| Non-affected arm first, as comparison | Calibrate the right side first and store it as the personal reference |
| No assistance for the tested limb | Supervisor check-in can't help the tested arm; log `assistance_given` |
| All items 0–2 | `fma_style_score ∈ {0, 1, 2}` |
| No compensation with other body parts | Any compensation flag caps the rep at 1 |
| Choose the lower score if uncertain | Uncertainty band = ± tolerance around the threshold (T2) |
| Contracture > ¼ of normal range: max score 1 or 0 | Therapist flag caps the item at 1 |
| Elbow deficit < 30° from contracture = max available; ≥ 30° = not testable | `elbow_start_allowance`; disable the item if ≥ 30° |

### 5.2 Items the webcam can measure

The full list of 33 items is in the CSV.

| Item | Criterion for 2 | Measured by | Max webcam score |
|---|---|---|---|
| 05 Abduction in flexor synergy | ≥ 90° | shoulder_elevation, frontal | 2 |
| 07 Elbow flexion in synergy | Full flexion, hand touches ear | elbow_flexion + wrist-to-ear distance | 2 |
| 10 Elbow extension in synergy | Full extension to 0° | elbow_flexion → 0 | 2 |
| 13 Shoulder flexion 0–90°, elbow 0° | 90° without abduction, elbow kept straight; score 0 if the elbow bends at the start | shoulder_elevation (sagittal) + elbow_flexion + in-plane ratio | 2 |
| 15 Shoulder abduction 0–90°, elbow 0° | 90°, elbow straight, forearm held; no lateral trunk lean, no shoulder hike | shoulder_elevation (frontal) + elbow + lean + hike | 2 |
| 16 Shoulder flexion 90–180° | Full available range (manual cites ~160° active in non-disabled adults) | as 13 | 2 |
| 19 / 21 Wrist 15° dorsal extension | 15° held against resistance | wrist_extension ≥ 15° | **1** (resistance can't be measured) |
| 20 / 22 Repeated wrist ext/flex | ≥ 2 full cycles; speed not scored | wrist angle time series | 2 |
| 23 Circumduction | Full and smooth | hand trajectory circularity + jerk | 2 (medium reliability) |
| 24 Mass flexion | All fingertips touch the palm, thumb outside the fist | fingertip_to_palm ≤ 0.35 (proposed) + thumb position | 2 |
| 25 Mass extension | All 5 fingers fully extended | finger_flexion ≤ tolerance on all joints | 2 |
| 26 Hook grasp | MCP 0°, PIP/DIP flexed, held against resistance | finger angles | **1** |
| 28 Pincer | Pad-to-pad pencil hold against a tug | pinch_gap ≤ 0.12 (proposed), digits 3–5 not helping | **1** |
| 31–33 Finger-to-nose ×5 | No tremor; lands on nose; affected arm < 2 s slower | index-tip path, nose_touch_error, timer on both arms | 2 (adapted: eyes open) |

**Not measurable:**
- reflexes (01, 02, 18)
- hand to lumbar spine (12): the hand is hidden behind the body
- low reliability: shoulder girdle retraction (03), external rotation (06), cylinder and spherical grasp (29, 30), because they need depth or objects
- medium reliability: forearm rotation items (08, 11, 14, 17), via the palm-normal direction

### 5.3 Assessment mode vs training mode

- **Assessment mode** (weekly, proposed): FMA-like. Demo, 2 practice trials, then 3 attempts. The item score is the best attempt. This produces an "FMA-style profile" for the weekly report and for progress claims.
- **Training mode** (daily): sets of reps with coaching. Each rep gets an FMA-style score for logging and a **coaching success** flag for feedback (§10).

---

## 6. Layer 2a: normal range of motion (ceiling and plausibility)

Values are from SOUCIE2011, age 45–69. The full table is in `data/soucie2011_normative_rom.csv`.

| Motion (passive) | Female mean (95% CI) | Female p25 / max | Male mean (95% CI) | Male max |
|---|---|---|---|---|
| Shoulder flexion | 168.1 (166.7–169.5) | 163.5 / 188 | 164.0 (162.3–165.7) | 185.5 |
| Elbow flexion | 148.3 (147.3–149.3) | 145 / 164 | 143.5 (142.3–144.7) | 157.5 |
| Elbow extension (past 0) | 3.6 (2.6–4.6) | 0 / 20 | −0.7 (−1.5–0.1) | 14 |
| Pronation | 80.8 (79.7–81.9) | 78 / 99 | 77.7 (76.5–78.9) | 88 |
| Supination | 87.2 (86.0–88.4) | 82.5 / 109 | 82.4 (80.9–83.9) | 101.5 |
| Knee flexion | 137.8 (136.5–139.1) | 134 / 160 | 132.9 (131.6–134.2) | 150 |
| Hip flexion | 130.8 (129.2–132.4) | 124.5 / 154.5 | 127.2 (125.7–128.7) | 143 |

How to use these numbers:

1. **Ceiling only.** Use them as the upper cap, combined with the unaffected side. For active shoulder flexion, use **160°** (FMA2026, citing Gill 2020) rather than the passive 168°.
2. **Plausibility.** Any reading above the group max + tolerance is a tracking error.
3. **Sex setting.** Use `sex_for_norms` in `therapist_profile.json`. Marketplace testers aren't all female.
4. **Age.** ROM falls with age, and the norms stop at 69, so Eleanor's real ceiling is probably lower. That's another reason the unaffected right side is the main ceiling. Soucie also found left-right differences under 1°, which supports using the healthy limb as the comparison.

---

## 7. Layer 2b: range needed for daily tasks (milestones)

### 7.1 Arm and wrist

Values are from GATES2016: mean peak angle (95% CI) in healthy adults. The full table is in `data/gates2016_adl_peak_angles.csv`.

| Milestone | Shoulder elevation | Elbow flexion | Wrist extension | Supination | Eleanor goal link |
|---|---|---|---|---|---|
| Drinking from a cup | 71 (67–75) | **121 (115–126)** | 33 (26–40) | 22 (13–31) | Tea |
| Can off shelf, 1.48 m | 86 (78–95) | 100 (92–107) | 31 (26–36) | 27 | Kitchen cupboard |
| Can off shelf, head height | 105 (102–108) | 105 (98–111) | 32 | 32 | Top shelf |
| Box off shelf, head height | **108 (104–111)** | 120 (115–125) | 31 | 38 | Top shelf, two hands |
| Donning and zipping pants | 51 | 98 | 40 | 24 | Dressing |
| **All 8 tasks** | 0–108 | 0–121 (every task ≥ 81) | −40 to +38 (flex) | −53 to +13 | "Enough for daily life" |

Healthy participants moved the trunk less than 10° in most tasks (GATES2016). That's the basis for the trunk compensation rule in §8.

Gates measured "humeral elevation" in any plane with 3D motion capture. Our 2D `shoulder_elevation` approximates it, so treat these as milestones to announce ("That's about the height your arm needs for the cupboard"), not as pass/fail lines.

### 7.2 Fingers

Values are from BAIN2015: degrees of flexion, with 0 = straight and negative = hyperextension.

| Joint | Healthy active range | **Open enough for 90% of tasks** (flexion ≤) | **Closed enough for 90% of tasks** (flexion ≥) | Pre-grasp mean min | Grasp mean max |
|---|---|---|---|---|---|
| MCP | −19 to 90 | **19** | **71** | 11 | 77 |
| PIP | −7 to 101 | **23** | **87** | 20 | 93 |
| DIP | −6 to 84 | **10** | **64** | 9 | 72 |

The functional arcs per finger (°), read from the bar labels in Bain's Figure 4:
- MCP: index 40, middle 50, ring 53, little 65
- PIP: 64, 59, 63, 71
- DIP: 57, 54, 45, 59

**How the Bloom exercise uses these:**
- **Open:** the "functional open" milestone is MCP ≤ 19°, PIP ≤ 23°, DIP ≤ 10° on digits 2–5, each + tolerance. FMA-25 score 2 needs all 5 fingers straight (≤ tolerance).
- **Close:** the "functional grasp" milestone is MCP ≥ 71°, PIP ≥ 87°, DIP ≥ 64°. FMA-24 score 2 needs fingertips on the palm with the thumb outside.

MediaPipe finger angles aren't validated in these sources. Also show progress as `hand_aperture` in % of her right-hand range, and only claim improvement at ≥ 20° (proposed) or with a consistent weekly trend.

---

## 8. Layer 2c: compensation rules

In `benchmarks.json → layer2_compensation`.

| Compensation | Detect | Threshold | Source | Cue |
|---|---|---|---|---|
| Trunk flexion or lean | change in trunk angle from start | > 10° | GATES2016 (healthy ADLs mostly < 10°); FMA bans trunk compensation | "Keep your back against the chair." |
| Shoulder hike | shoulder_girdle_elevation | > 0.15 of shoulder width (**proposed**) | FMA item 15; RPS | "Let your shoulder relax down." |
| Elbow bends in straight-arm items | elbow_flexion > start + tolerance | tolerance (9° forward, 6.5° abduction) | FMA items 13, 15, 16 | "Keep your elbow straight." |
| Arm leaves the plane | upper_arm_in_plane_ratio | < 0.87 (**proposed**) | FMA 13/16 "without abduction" | "Lift straight in front of you." / "…out to the side." |
| Reach done by the trunk (close target) | trunk_share_of_reach | 3: ≤ 0.10 (proposed reading of "no or almost no"); 1: > 0.50; 0: = 1.0 | LEVIN2004 | "Let your arm do the reaching." |
| Reach done by the trunk (far target) | trunk_share_of_reach | 3: ~0.25 ± 0.10 with elbow almost straight; 1: ~0.50; 0: > 0.75 and the hand doesn't arrive | LEVIN2004 | as above |

RPS joint anchors for the reach (LEVIN2004):

| Motion | Close target | Far target |
|---|---|---|
| Shoulder flexion | ~20° | ~40° |
| Horizontal adduction | ~40° | ~40° |
| Elbow extension excursion | ~80° | ~100° |

People with stroke used up to 4.5× the trunk movement of unimpaired people, with ~25% less elbow extension and > 50% less shoulder flexion.

**Compensation policy:**
- A compensation lowers the FMA-style score to 1.
- It **doesn't** cancel coaching success. The rep counts, gets one cue, and is logged.
- If most reps in a set show compensation, the level goes down (§9.4).

---

## 9. Layer 3: personal baseline

### 9.1 Calibration protocol (session 0 and weekly assessment)

1. **Setup check** (§3.1). Camera distance and height, view for the task, light, clothing. Block the session until framing passes.
2. **Seat.** Log the seat type (FMA2026).
3. **Demo + 2 practice trials** (FMA2026, LAZEM2026, LEVIN2004).
4. **Right side first, 3 recorded reps.** Then the left side, 3 reps (FMA2026).
5. **Rest.** About 10 s between reps and 1 min between tasks (LAZEM2026).
6. **Store per metric and side:**
   - **best** of 3 (FMA "best performance")
   - **mean** of 3
   - **symmetry index** = left best / right best
7. **Therapist limits (optional).**
   - passive ROM caps
   - contracture > ¼ of normal range → item max 1
   - elbow deficit < 30° counts as available range; ≥ 30° means the straight-arm items aren't testable

In code: `Calibration.add(metric, side, reps, direction)`, then `best()` and `symmetry_index()`.

### 9.2 Target ladder

```
ceiling  = min(unaffected_best, therapist_passive_limit, normative_ceiling)
step     = tolerance_deg(task, joint)            # T3: never smaller than tolerance
target_k = min(ceiling, affected_baseline + k * step)
milestones = FMA thresholds + Gates/Bain task values between baseline and ceiling
```

In code: `build_ladder(baseline, unaffected_max, tolerance, milestones, normative_ceiling, therapist_limit)`.

**Worked example (hypothetical numbers):** forward arm raise.
- Right side best = 150°, left best = 58°, tolerance = 5.5°.
- Ceiling = min(150, 160) = 150°.
- Level 0 = 58°, level 1 = 63.5°, level 3 = 74.5° …
- Milestones on the way:
  - 71° drinking (Gates)
  - 86° shelf at 1.48 m (Gates)
  - 90° FMA-13
  - 105° / 108° head-height shelf (Gates)
- Each milestone crossed triggers a goal-linked message once.

### 9.3 Which reference applies to which question

| Question Think asks | Reference used |
|---|---|
| Did this rep reach today's goal? | Layer 3 target − tolerance |
| Was the rep done correctly? | Layer 1 form rules + Layer 2c compensation |
| What FMA-style score does it get? | FMA threshold with the conservative uncertainty band |
| How far could she eventually go? | Layer 3 ceiling (right side), capped by Layer 2a |
| What does this range mean in daily life? | Layer 2b milestones |
| Did she really improve? | Change ≥ MDC (T4) |
| Is this reading real? | Layer 2a plausibility |

### 9.4 Progression rules

These are **proposed** project defaults from the Eleanor Extended Program doc. They aren't from the seven papers.

| Condition | Action |
|---|---|
| Clean success ≥ 80% in 2 sessions in a row, pain not higher, compensation not rising | Level +1 |
| Success 50–79% | Stay |
| Success < 50%, or compensation in most reps | Level −1 |
| Pain up ≥ 2 points, or any new shoulder pain | Pause that exercise; flag the therapist |
| 2 missed sessions | Restart one level lower; no guilt message |

---

## 10. Scoring

### 10.1 Rep state machine (`transitions` library)

```
IDLE → CHECK_START (start posture held ≥ debounce frames; e.g., arm at side, elbow ≤ allowance)
     → MOVING (primary metric leaves start by > tolerance)
     → PEAK/HOLD (track peak; hold timer if the exercise has hold_s)
     → RETURN (back within start + tolerance)  → REP_DONE → score_rep() → Act
Any state → PAUSED  (landmarks invalid ≥ 1 s)   → system-blame message, then resume
Any state → SAFETY  (Stop key/voice, "I feel unwell")
```

### 10.2 FMA-style score per rep

`score_rep()` in `benchmark_engine.py`.

| Score | Condition |
|---|---|
| **0** | Movement from start < tolerance, **or** the item's start position can't be obtained (e.g., FMA-13 elbow bent at the start) |
| **1** | Moved, but: peak < FMA threshold; **or** peak within ± tolerance of the threshold (uncertain → lower score); **or** a form rule broke before the threshold was reached; **or** any compensation |
| **2** | Peak > threshold + tolerance, every form rule held, no compensation |

Items capped by webcam limits (19, 21, 26, 27, 28, 29, 30) never exceed 1.

### 10.3 Coaching success

- **Success:** peak ≥ personal target − tolerance **and** all form rules held. Benefit of the doubt goes to the user, which fits the "nothing can go wrong" design principle.
- **Success with cue:** a success where a compensation was flagged. The rep counts, gets one cue, and is logged.

### 10.4 Aggregation

- **Set:** clean success rate, compensation rate, best peak.
- **Session:** best of 3 in assessment mode. In training mode, the progression inputs from §9.4.
- **Week:** FMA-style item profile, best values vs baseline (MDC rule), milestones crossed.

### 10.5 Feedback mapping (Act)

| Outcome | Feedback |
|---|---|
| Score 2 or clean success | Specific praise. If a milestone was crossed, name the life goal. Mention a gain only if Δ ≥ MDC |
| Success with compensation | Praise + **one** cue (the compensation) |
| Score 1 / not reached | One cue: the first broken form rule, else "a little further" |
| Score 0 | Encouragement + replay the demo |
| Tracking lost | "I can't see your arm. Please move into the box." |
| Limit | At most one technique cue per set (proposed) |

---

## 11. Exercise profiles

These are in `benchmarks.json → exercises`.

| Exercise | View | Primary metric | FMA items | Targets / milestones | Tol / MDC (°) | Form and compensation rules |
|---|---|---|---|---|---|---|
| `shoulder_flexion_raise` | sagittal | shoulder_elevation ↑ | 13, 16 | 90 (FMA); 71 / 86 / 105 / 108 (Gates); ceiling min(right, 160) | 5.5 / 13.19 | elbow ≤ start + 9; in-plane ≥ 0.87; trunk ≤ 10 |
| `shoulder_abduction_raise` | frontal | shoulder_elevation ↑ | 15, 05 | 90 (FMA); ceiling = right side | 6.5 / 11.41 | elbow straight; no hike > 0.15; no lateral lean > 10 |
| `hand_to_mouth` | sagittal | elbow_flexion ↑ | 07 (partial) | elbow 121 (115–126), shoulder 71 (Gates drinking) | 5.5 / 20.55 | trunk ≤ 10; **empty cup only** |
| `hand_to_head` | sagittal | elbow_flexion ↑ | 07 | hand reaches the ear zone | 5.0 / 32.65 | trunk ≤ 10 |
| `elbow_extension` | sagittal | elbow_flexion ↓ | 10 | 0° (FMA) or therapist contracture allowance | 5.0 / 20.55 | trunk ≤ 10 |
| `wrist_extension` | side, elbow ~90° or ~0° | wrist_extension ↑ | 19/21 (max 1), 20/22 | 15 (FMA), 33 cup, 40 all ADLs (Gates); ≥ 2 cycles | 5.0 / provisional 10 | elbow steady |
| `hand_open_close` (Bloom) | palm to camera | finger_flexion ↓ / ↑ | 25, 24 | Bain open (19/23/10), grasp (71/87/64); FMA all 5 fingers | 10 / provisional 20 | all fingers, not just some |
| `pinch` (Bubble pinch) | palm/side | pinch_gap ↓ | 28 (max 1) | gap ≤ 0.12 (proposed) | — | digits 3–5 not helping |
| `tabletop_reach` (Lamps) | 45° | trunk_share_of_reach | — (RPS) | close 1 cm, far 30 cm; trunk share 0 / ~0.25 | 5.0 (trunk) | RPS table §8 |
| `finger_to_nose_timed` | frontal | time, touch error, path | 31–33 | affected − unaffected: < 2 s → 2; 2–5.9 s → 1; ≥ 6 s → 0 | — | no head/trunk movement; **eyes open (adapted, not comparable)** |

---

## 12. Logging

Templates are in `/templates`.

**`rep_log_template.csv`**, one row per rep:
- session, mode, exercise, side, level
- target, threshold, tolerance
- start, peak, hold, duration
- FMA-style score, coaching success, compensation
- broken rules, flags, max trunk change, shoulder hike, in-plane ratio
- cue, feedback key
- tracking quality, plausibility rejects

**`session_log_template.csv`**, one row per session:
- seat type, camera check, supervisor, assistance
- pain and fatigue before and after
- success and compensation rates
- best and mean values
- improvement claims, milestones, level changes

These logs are also your **objective measures for the Testing section of the report**: repetitions, persistence, execution accuracy (clean success rate), compensation rate, and the FMA-style profile over time.

---

## 13. Build order and testing checklist

| Step | Build | Done when |
|---|---|---|
| 1 | Load `benchmarks.json` in Think; add pixel scaling + `LowPass` + visibility gate in Sense | Angles stable on a still teammate (jitter < tolerance) |
| 2 | Metric functions for the first exercises (`shoulder_elevation`, `elbow_flexion`, `finger_flexion`, `hand_aperture`) | `test_benchmark_engine.py` passes; a teammate's arm at 90° (checked with a phone goniometer app) reads within ± tolerance |
| 3 | Setup/framing check screen (§3.1) | Session can't start until hip–shoulder–elbow–wrist (or the hand) is fully visible |
| 4 | Calibration flow (§9.1) + therapist profile | Right-then-left calibration saves best/mean/symmetry |
| 5 | Rep state machine (`transitions`) + `score_rep()` | Every rep produces a row in `rep_log.csv` |
| 6 | Form + compensation rules (§8) | A teammate leaning forward triggers the trunk cue; bending the elbow triggers the elbow cue |
| 7 | Target ladder + milestones + MDC claim rule | Targets rise only after the §9.4 condition; no "you improved" below MDC |
| 8 | Remaining exercises: Bloom, pinch, wrist, reach, finger-to-nose timing | Each profile in §11 runs end-to-end |
| 9 | Marketplace test | 3+ testers mimicking the persona; log the numbers from §12 plus SUS / PACES / Borg questionnaires |

**Self-checks worth doing** (these give you reliability material for the report):
- **Agreement:** measure 5 static arm angles on 2 teammates with a phone goniometer and compare. Report the mean difference and range, and compare it with LAZEM2026.
- **Repeatability:** repeat the calibration on 2 different days. Your own day-to-day differences show whether the Lazem MDC is realistic for your setup.
- **Edge cases:** check left/right hand labels with a mirrored preview; two hands in view; a partial frame; low light.

---

## 14. Proposed defaults (not from the sources; tune in testing)

| Value | Default | Where |
|---|---|---|
| Visibility minimum | 0.5 | signal_processing |
| Invalid-frame pause | 1 s | signal_processing |
| Debounce | 5 frames | signal_processing |
| Start posture: arm at side | shoulder_elevation ≤ 20° | exercises |
| Arm-in-plane ratio | 0.87 (~30°) | compensation |
| Shoulder hike | 0.15 × shoulder width | compensation |
| RPS close-target "almost no trunk" | trunk share ≤ 0.10 | compensation |
| RPS far-target band | 0.25 ± 0.10 | compensation |
| Finger-joint tolerance / change threshold | 10° / 20° | tolerance |
| Wrist change threshold (no MDC published) | 10° | tolerance |
| Fingertip-to-palm "touch" | ≤ 0.35 palm lengths | hand_open_close |
| Pinch contact | gap ≤ 0.12 palm lengths | pinch |
| Hand-to-ear zone | wrist-to-ear ≤ 0.5 shoulder widths | hand_to_head |
| Nose-touch error | on nose ≤ 0.35 and slight ≤ 0.8 eye distances | finger_to_nose |
| Target step | = tolerance | layer3 |
| Assessment frequency | weekly | scoring |
| Progression rules | §9.4 | layer3 |
| Reps / rest / hold | therapist_profile.json | templates |
| Eyes-open finger-to-nose | adapted for MCI and fall risk | finger_to_nose |

---

## 15. Wording for the report and its limitations

For the team writing the report:

> Movement benchmarks were organised in three layers. Form criteria were adapted from the Fugl-Meyer Assessment for the Upper Extremity (2026 international manual), including its start and end positions, banned compensations and conservative scoring rule. Range targets used normative passive ROM (Soucie et al., 2011) as a ceiling, and task-specific ROM from healthy adults (Gates et al., 2016; Bain et al., 2015) as goal-linked milestones. Compensation rules for reaching followed the Reaching Performance Scale (Levin et al., 2004). Targets were personalised by calibrating the unaffected arm first, in line with FMA-UE practice. Measurement tolerance and minimum detectable change came from MediaPipe validation in stroke survivors (Lazem et al., 2026; Jayavel et al., 2025).

Limitations to state:

- The webcam scores are "FMA-style" and not a clinical FMA-UE.
- Items that need resistance are capped at 1.
- The normative data stops at age 69, and it's passive ROM.
- The functional ROM data comes from young healthy adults, measured with 3D motion capture.
- MediaPipe finger angles and wrist change thresholds are unvalidated.
- Lazem's reference method was Kinovea, not 3D motion capture.

---

## 16. References

- **FMA2026:** Hervé-Colas J, Newton SP, Engelter ST, et al., Alt Murphy M. Standardized international manual of the Fugl-Meyer Assessment of motor function after stroke. *Neurorehabil Neural Repair.* 2026. doi:10.1177/15459683251412300 (FMA-UE instruction manual, University of Gothenburg, May 2026)
- **SOUCIE2011:** Soucie JM, Wang C, Forsyth A, et al. Range of motion measurements: reference values and a database for comparison studies. *Haemophilia.* 2011;17:500–507. doi:10.1111/j.1365-2516.2010.02399.x
- **GATES2016:** Gates DH, Walters LS, Cowley J, Wilken JM, Resnik L. Range of motion requirements for upper-limb activities of daily living. *Am J Occup Ther.* 2016;70(1):7001350010. doi:10.5014/ajot.2016.015487
- **BAIN2015:** Bain GI, Polites N, Higgs BG, Heptinstall RJ, McGrath AM. The functional range of motion of the finger joints. *J Hand Surg Eur Vol.* 2015;40(4):406–411. doi:10.1177/1753193414533754
- **LEVIN2004:** Levin MF, Desrosiers J, Beauchemin D, Bergeron N, Rochette A. Development and validation of a scale for rating motor compensations used for reaching in patients with hemiparesis: the Reaching Performance Scale. *Phys Ther.* 2004;84(1):8–22. https://academic.oup.com/ptj/article/84/1/8/2805317
- **LAZEM2026:** Lazem H, Harris D, Hall A, et al. Validity and reliability of the Track-UL algorithm compared with Kinovea software for measuring upper-limb functional range of motion in people after stroke. *JMIR Rehabil Assist Technol.* 2026;13:e87128. doi:10.2196/87128
- **JAYAVEL2025:** Jayavel P, Srinivasan HK, Karthik V, Fouly A, Devaraj A. Human upper limb kinematics using a novel algorithm in post-stroke patients. *Proc Inst Mech Eng H.* 2025;239(1):48–55. doi:10.1177/09544119251315421
- **GILL2020 (via FMA2026):** Gill TK, et al. Shoulder range of movement in the general population. *BMC Musculoskelet Disord.* 2020;21:676. doi:10.1186/s12891-020-03665-9. Secondary citation only.
