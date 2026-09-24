"""Builds ../benchmarks.json from the source CSVs in ../data plus the rules below.

Every number carries either a `source` (a key in SOURCES) or `proposed: true`.
Re-run after editing: python tools/build_benchmarks_json.py
"""
import csv, json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")


def rows(name):
    with open(os.path.join(DATA, name), newline="") as fh:
        return list(csv.DictReader(fh))


def f(x):
    return None if x in ("", None) else float(x)


def ceil_half(x):
    return math.ceil(x * 2) / 2.0


SOURCES = {
    "FMA2026": {
        "citation": "Hervé-Colas J, Newton SP, Engelter ST, et al., Alt Murphy M. Standardized international manual of the Fugl-Meyer Assessment of motor function after stroke. Neurorehabil Neural Repair. 2026. FMA-UE instruction manual, University of Gothenburg, May 2026.",
        "doi": "10.1177/15459683251412300",
        "used_for": "Layer 1 item criteria (0/1/2), general scoring rules, start positions, compensation bans, time rule for coordination",
        "limits": "Assessment tool, not a training target; several items need an assessor's resistance, a reflex hammer or objects; webcam scores are 'FMA-style', not a clinical FMA score. Free for clinical/research use without charge; cite the publication.",
    },
    "SOUCIE2011": {
        "citation": "Soucie JM, Wang C, Forsyth A, et al. Range of motion measurements: reference values and a database for comparison studies. Haemophilia. 2011;17:500-507.",
        "doi": "10.1111/j.1365-2516.2010.02399.x",
        "used_for": "Layer 2a normative ROM ceilings and plausibility limits (female/male, 45-69 y)",
        "limits": "Passive ROM with goniometer, supine for shoulder/elbow; healthy US volunteers up to 69 y (Eleanor is 72); active ROM after stroke will be lower. Left-right differences < 1 deg, supporting the healthy limb as reference.",
    },
    "GATES2016": {
        "citation": "Gates DH, Walters LS, Cowley J, Wilken JM, Resnik L. Range of motion requirements for upper-limb activities of daily living. Am J Occup Ther. 2016;70(1):7001350010.",
        "doi": "10.5014/ajot.2016.015487",
        "used_for": "Layer 2b functional ROM milestones per daily task (drinking, shelf reach, dressing); healthy trunk-motion reference",
        "limits": "15 healthy young adults (mean 26 y), right arm, 3D motion capture with globe angles; webcam 2D angles only approximate these definitions.",
    },
    "BAIN2015": {
        "citation": "Bain GI, Polites N, Higgs BG, Heptinstall RJ, McGrath AM. The functional range of motion of the finger joints. J Hand Surg Eur Vol. 2015;40(4):406-411.",
        "doi": "10.1177/1753193414533754",
        "used_for": "Layer 2b finger ROM needed for 90% of Sollerman ADL tasks; active finger ROM ceiling",
        "limits": "10 healthy volunteers aged 18-53, dominant hand, electrogoniometer; MediaPipe finger angles are not validated against it.",
    },
    "LEVIN2004": {
        "citation": "Levin MF, Desrosiers J, Beauchemin D, Bergeron N, Rochette A. Development and validation of a scale for rating motor compensations used for reaching in patients with hemiparesis: the Reaching Performance Scale. Phys Ther. 2004;84(1):8-22.",
        "url": "https://academic.oup.com/ptj/article/84/1/8/2805317",
        "used_for": "Layer 2c compensation rules for reaching (trunk share of reach, shoulder/elbow contribution), reach task setup",
        "limits": "Observational 0-3 scale; preliminary reliability (kappa .14-.85); sample younger than typical stroke population.",
    },
    "LAZEM2026": {
        "citation": "Lazem H, Harris D, Hall A, et al. Validity and reliability of the Track-UL algorithm compared with Kinovea software for measuring upper-limb functional range of motion in people after stroke. JMIR Rehabil Assist Technol. 2026;13:e87128.",
        "doi": "10.2196/87128",
        "used_for": "Measurement tolerance (95% limits of agreement) and minimum detectable change for MediaPipe 2D shoulder/elbow angles in stroke; camera setup",
        "limits": "Reference was Kinovea (2D video tool), not 3D motion capture; 27 chronic stroke survivors; only 13/27 produced usable home videos.",
    },
    "JAYAVEL2025": {
        "citation": "Jayavel P, Srinivasan HK, Karthik V, Fouly A, Devaraj A. Human upper limb kinematics using a novel algorithm in post-stroke patients. Proc Inst Mech Eng H. 2025;239(1):48-55.",
        "doi": "10.1177/09544119251315421",
        "used_for": "Agreement of MediaPipe (BlazePose + BlazeHand) with a goniometer for elbow, wrist and shoulder rotation; wrist angle landmark choice",
        "limits": "10 patients; Results text swaps some bias values vs Table 1 (Table 1 used here).",
    },
    "GILL2020_via_FMA2026": {
        "citation": "Gill TK, Shanahan EM, Tucker GR, et al. Shoulder range of movement in the general population. BMC Musculoskelet Disord. 2020;21:676 - cited in FMA2026 item 16.",
        "doi": "10.1186/s12891-020-03665-9",
        "used_for": "Active shoulder flexion ~160 deg in community-living non-disabled adults (secondary citation; original not reviewed)",
        "limits": "Only the single statement quoted in the FMA manual is used.",
    },
}

POSE = {"nose": 0, "left_eye": 2, "right_eye": 5, "left_ear": 7, "right_ear": 8,
        "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14,
        "left_wrist": 15, "right_wrist": 16, "left_hip": 23, "right_hip": 24,
        "left_knee": 25, "right_knee": 26, "left_ankle": 27, "right_ankle": 28}
HAND = {"wrist": 0, "thumb_cmc": 1, "thumb_mcp": 2, "thumb_ip": 3, "thumb_tip": 4,
        "index_mcp": 5, "index_pip": 6, "index_dip": 7, "index_tip": 8,
        "middle_mcp": 9, "middle_pip": 10, "middle_dip": 11, "middle_tip": 12,
        "ring_mcp": 13, "ring_pip": 14, "ring_dip": 15, "ring_tip": 16,
        "pinky_mcp": 17, "pinky_pip": 18, "pinky_dip": 19, "pinky_tip": 20}

ANGLES = {
    "shoulder_elevation": {"formula": "angle(hip, shoulder, elbow)", "space": "2D pixel coords (x*W, y*H)", "view": "sagittal for flexion, frontal for abduction",
                           "zero": "arm at side = ~0", "matches": "LAZEM2026 shoulder angle (hip-shoulder-elbow, 2D dot product); approximates GATES2016 humeral elevation"},
    "elbow_flexion": {"formula": "180 - angle(shoulder, elbow, wrist)", "space": "2D pixel coords", "view": "sagittal (or view where the forearm moves in image plane)",
                      "zero": "full extension = 0", "matches": "LAZEM2026 elbow angle; JAYAVEL2025 elbow flexion; SOUCIE2011/GATES2016 clinical convention"},
    "wrist_extension": {"formula": "signed angle(forearm = pose elbow->pose wrist, hand = hand[0]->hand[9]); sign fixed at calibration so dorsal extension is positive", "space": "2D pixel coords, side view of forearm",
                        "zero": "hand in line with forearm = 0", "matches": "JAYAVEL2025 (BlazePose + BlazeHand) wrist flexion/extension"},
    "finger_flexion": {"formula": "180 - angle(prev, joint, next) on hand_world_landmarks; MCP=(0,5,6),(0,9,10),(0,13,14),(0,17,18); PIP=(5,6,7),(9,10,11),(13,14,15),(17,18,19); DIP=(6,7,8),(10,11,12),(14,15,16),(18,19,20); thumb MCP=(1,2,3), IP=(2,3,4)",
                       "space": "3D hand world landmarks (metres)", "view": "palm or side facing camera", "zero": "straight joint = 0 (unsigned: hyperextension cannot be distinguished)",
                       "matches": "BAIN2015 joint definitions (MCP/PIP/DIP flexion, 0 = straight)"},
    "hand_aperture": {"formula": "mean(dist(tip_k, wrist) for tips 4,8,12,16,20) / dist(0, 9)", "space": "3D hand world landmarks", "zero": "normalised, unitless", "matches": "proposed metric (no source)"},
    "fingertip_to_palm": {"formula": "dist(tip, palm_centre) / dist(0,9); palm_centre = mean(0,5,9,13,17)", "space": "3D hand world landmarks", "matches": "operationalises FMA2026 item 24 'fingertips touching the palm' (threshold proposed)"},
    "pinch_gap": {"formula": "dist(4, 8) / dist(0, 9)", "space": "3D hand world landmarks", "matches": "operationalises FMA2026 item 28 pad-to-pad opposition (threshold proposed)"},
    "trunk_flexion": {"formula": "angle between (hip_mid -> shoulder_mid) and image vertical; report change from start frame", "space": "2D pixel coords", "view": "sagittal",
                      "matches": "compensation check (FMA2026 no trunk compensation; GATES2016 healthy ADL trunk motion mostly < 10 deg)"},
    "trunk_lateral_lean": {"formula": "same as trunk_flexion in frontal view", "space": "2D pixel coords", "view": "frontal", "matches": "FMA2026 item 15 bans lateral trunk compensation"},
    "shoulder_girdle_elevation": {"formula": "((ear_y - shoulder_y)_start - (ear_y - shoulder_y)_now) / shoulder_width; positive = shoulder hikes toward ear", "space": "2D pixel coords", "view": "frontal",
                                  "matches": "FMA2026 item 15 bans shoulder girdle elevation; LEVIN2004 'excessive scapular elevation'"},
    "upper_arm_in_plane_ratio": {"formula": "len2D(shoulder->elbow) / len2D(shoulder->elbow at start with arm at side)", "space": "2D pixel coords",
                                 "matches": "proposed foreshortening check for 'without abduction' (FMA13/16) or 'without flexion' (FMA15); ratio cos(30 deg)=0.87 means ~30 deg out of camera plane"},
    "trunk_share_of_reach": {"formula": "forward displacement of shoulder_mid / forward displacement of wrist, from reach start to grasp", "space": "2D pixel coords", "view": "sagittal or 45 deg",
                             "matches": "LEVIN2004 trunk displacement component (share of hand displacement done by the trunk)"},
    "nose_touch_error": {"formula": "dist(hand index_tip, pose nose) / dist(pose left_eye, pose right_eye)", "space": "2D pixel coords", "view": "frontal", "matches": "operationalises FMA2026 item 32 dysmetria (thresholds proposed)"},
}


def build_tolerance():
    me = rows("measurement_error.csv")
    out = {}
    task_key = {"arm abduction (frontal view)": "abduction", "arm elevation / forward (sagittal view)": "forward_elevation",
                "hand to head (sagittal view)": "hand_to_head", "hand to mouth (sagittal view)": "hand_to_mouth"}
    for r in me:
        if r["source_id"] != "LAZEM2026" or r["side"] != "affected":
            continue
        k = f'{task_key[r["task"]]}.{r["joint"]}'
        d = out.setdefault(k, {"loa": {}, "mdc": {}})
        if r["metric"] == "concurrent_validity":
            d["loa"][r["setting"]] = [f(r["loa_low_deg"]), f(r["loa_high_deg"])]
        else:
            d["mdc"][r["setting"]] = f(r["mdc_deg"])
    floor = 5.0  # SOUCIE2011: trained therapists with goniometers differed by up to 5 deg
    for k, d in out.items():
        m = max(abs(v) for pair in d["loa"].values() for v in pair)
        d["tolerance_deg"] = max(floor, ceil_half(m))
        d["tolerance_rule"] = "max(5 deg goniometer inter-rater floor [SOUCIE2011], worst |95% LoA| affected side lab+home rounded up to 0.5 [LAZEM2026])"
        d["mdc_default_deg"] = d["mdc"].get("laboratory")
        d["mdc_rule"] = "use laboratory MDC when the guided camera setup is followed; home MDC is available as a stricter/looser alternative"
        d["source"] = ["LAZEM2026", "SOUCIE2011"]
    jay = {r["task"]: r for r in me if r["source_id"] == "JAYAVEL2025"}
    out["wrist_flexion_extension"] = {"loa": {"goniometer": [f(jay["wrist extension"]["loa_low_deg"]), f(jay["wrist extension"]["loa_high_deg"])],
                                              "goniometer_flexion": [f(jay["wrist flexion"]["loa_low_deg"]), f(jay["wrist flexion"]["loa_high_deg"])]},
                                      "tolerance_deg": floor, "tolerance_rule": "LoA < 2 deg vs goniometer (n=10) is below the 5 deg goniometer floor, so floor used",
                                      "mdc_default_deg": None, "mdc_rule": "no test-retest data; use 2 x tolerance as provisional change threshold", "provisional_change_threshold_deg": 2 * floor,
                                      "source": ["JAYAVEL2025", "SOUCIE2011"], "proposed_parts": ["provisional_change_threshold_deg"]}
    out["elbow_flexion_simple"] = {"loa": {"goniometer": [f(jay["elbow flexion"]["loa_low_deg"]), f(jay["elbow flexion"]["loa_high_deg"])]},
                                   "tolerance_deg": floor, "tolerance_rule": "goniometer floor (single-plane elbow flexion; JAYAVEL2025 LoA -2.7 to 1.62)",
                                   "mdc_default_deg": out["hand_to_mouth.elbow"]["mdc"]["laboratory"], "mdc_rule": "borrow LAZEM2026 hand-to-mouth elbow MDC", "source": ["JAYAVEL2025", "LAZEM2026", "SOUCIE2011"]}
    out["finger_joints"] = {"tolerance_deg": 10.0, "proposed": True, "tolerance_rule": "no MediaPipe finger-angle validation in the reviewed sources; 10 deg provisional, prefer normalised metrics relative to calibration",
                            "mdc_default_deg": None, "provisional_change_threshold_deg": 20.0, "proposed_parts": ["tolerance_deg", "provisional_change_threshold_deg"]}
    out["trunk"] = {"tolerance_deg": 5.0, "compensation_threshold_deg": 10.0,
                    "rule": "trunk change > 10 deg from start = compensation cue; GATES2016 found most healthy ADLs used < 10 deg trunk/pelvis motion; 5 deg floor as above",
                    "source": ["GATES2016", "SOUCIE2011"], "note": "derived use of a descriptive finding, not a published cut-off"}
    return out


def build_normative():
    out = {}
    for r in rows("soucie2011_normative_rom.csv"):
        if r["age_group_years"] != "45-69":
            continue
        key = r["motion"].lower().replace(" ", "_")
        out.setdefault(key, {})[r["sex"]] = {k: f(r[k]) for k in ("mean_deg", "sd_deg", "ci95_low_deg", "ci95_high_deg", "min_deg", "p25_deg", "p50_deg", "p75_deg", "max_deg")} | {"n": int(r["n"])}
        out[key]["position"] = r["position"]
    return {"source": "SOUCIE2011", "age_group": "45-69 (oldest available; Eleanor is 72)", "measurement": "passive, goniometer",
            "use": ["upper ceiling only (never a target)", "plausibility check: measured angle > group max + tolerance => probable tracking error"],
            "active_shoulder_flexion_reference_deg": {"value": 160, "source": "GILL2020_via_FMA2026", "note": "active flexion ~160 deg across age groups in non-disabled adults"},
            "motions": out}


def gates_lookup():
    g = {}
    for r in rows("gates2016_adl_peak_angles.csv"):
        g[(r["angle"], r["task"], r["limb_role"])] = {"mean": f(r["mean_deg"]), "ci95": [f(r["ci95_low_deg"]), f(r["ci95_high_deg"])]}
    return g


def build_functional():
    g = gates_lookup()
    def G(angle, task, role="ipsilateral"):
        v = g[(angle, task, role)]
        return {"deg": v["mean"], "ci95": v["ci95"], "task": task, "source": "GATES2016"}
    summary = {r["motion"]: {"neg": f(r["negative_limit_deg"]), "pos": f(r["positive_limit_deg"]), "notes": r["notes"]} for r in rows("gates2016_adl_minimum_rom_summary.csv")}
    bain = {(r["joint"], r["metric"]): f(r["value_deg"]) for r in rows("bain2015_finger_rom.csv") if r["finger"] == "all"}
    return {
        "upper_limb_milestones": {
            "shoulder_elevation": [G("humeral_elevation", "Drinking from a cup"), G("humeral_elevation", "Can off shelf (fixed height 1.48 m)"),
                                   G("humeral_elevation", "Can off shelf (head height)"), G("humeral_elevation", "Box off shelf (head height)")],
            "elbow_flexion": [G("elbow_flexion", "Box off ground", "bimanual"), G("elbow_flexion", "Can off shelf (fixed height 1.48 m)"), G("elbow_flexion", "Drinking from a cup")],
            "wrist_extension": [{"deg": 33, "ci95": [26, 40], "task": "Drinking from a cup", "source": "GATES2016", "note": "reported as -33 (extension negative); stored positive = extension"},
                                {"deg": 40, "task": "all 8 ADLs", "source": "GATES2016"}],
            "wrist_flexion": [{"deg": 38, "task": "all 8 ADLs", "source": "GATES2016"}],
            "forearm_supination": [{"deg": 22, "ci95": [13, 31], "task": "Drinking from a cup", "source": "GATES2016", "note": "reported negative"}, {"deg": 53, "task": "all 8 ADLs", "source": "GATES2016"}],
            "elbow_flexion_all_tasks_minimum_used_deg": {"value": 81, "source": "GATES2016", "note": "every task needed >= 81 deg"},
            "all_tasks_minimum_rom": summary,
        },
        "finger_functional_rom": {
            "source": "BAIN2015",
            "open_hand_pre_grasp_targets_deg": {"MCP_max_flexion": bain[("MCP", "functional90_min")], "PIP_max_flexion": bain[("PIP", "functional90_min")], "DIP_max_flexion": bain[("DIP", "functional90_min")],
                                                "meaning": "hand must open to at least this (flexion <= value) to pre-shape for 90% of Sollerman ADL tasks"},
            "closed_hand_grasp_targets_deg": {"MCP_min_flexion": bain[("MCP", "functional90_max")], "PIP_min_flexion": bain[("PIP", "functional90_max")], "DIP_min_flexion": bain[("DIP", "functional90_max")],
                                              "meaning": "fingers must flex to at least this to grasp in 90% of tasks"},
            "active_rom_healthy_deg": {"MCP": [bain[("MCP", "active_extension")], bain[("MCP", "active_flexion")]], "PIP": [bain[("PIP", "active_extension")], bain[("PIP", "active_flexion")]],
                                       "DIP": [bain[("DIP", "active_extension")], bain[("DIP", "active_flexion")]], "note": "negative = hyperextension"},
            "per_finger_functional_arc_deg": {r["joint"] + "." + r["finger"]: f(r["value_deg"]) for r in rows("bain2015_finger_rom.csv") if r["metric"] == "functional90_arc"},
        },
    }


def build_compensation():
    return {
        "general": {"rule": "Compensatory movements with other body parts (e.g., trunk) are not allowed when scoring", "source": "FMA2026"},
        "trunk_flexion_or_lean": {"threshold_change_deg": 10, "source": ["GATES2016"], "derived": True, "applies_to": ["shoulder_flexion", "shoulder_abduction", "hand_to_mouth", "hand_to_head", "wrist", "finger_to_nose"],
                                  "cue": "Keep your back against the chair."},
        "shoulder_girdle_elevation": {"threshold_ratio": 0.15, "proposed": True, "source_rule": "FMA2026 item 15 / LEVIN2004 'excessive scapular elevation'", "cue": "Let your shoulder relax down."},
        "elbow_bends_during_straight_arm_items": {"threshold_deg_above_start": "tolerance(elbow)", "source": "FMA2026 items 13, 15, 16", "cue": "Keep your elbow straight."},
        "arm_leaves_movement_plane": {"threshold_ratio": 0.87, "proposed": True, "source_rule": "FMA2026 13/16 'without abduction', 15 'forearm position maintained'", "cue_flexion": "Lift straight in front of you.", "cue_abduction": "Lift out to the side."},
        "reach_trunk_share": {
            "source": "LEVIN2004",
            "close_target": {"score3_max_share": 0.10, "score3_note": "'no or almost no' trunk displacement; 0.10 proposed", "score1_min_share": 0.50, "score0_share": 1.0},
            "far_target": {"appropriate_share": 0.25, "appropriate_tolerance": 0.10, "score1_share": 0.50, "score0_share": 0.75, "tolerance_proposed": True},
            "required_joint_motion_deg": {"shoulder_flexion": {"close": 20, "far": 40}, "shoulder_horizontal_adduction": {"close": 40, "far": 40}, "elbow_extension_excursion": {"close": 80, "far": 100}},
        },
    }


def build_fma_layer():
    items = {}
    for r in rows("fma_ue_2026_items.csv"):
        items[r["item"]] = {k: r[k] for k in ("section", "item_name", "score_0", "score_1", "score_2", "webcam_feasibility", "max_score_measurable_by_webcam", "signals_to_measure", "key_numeric_criteria", "start_position", "not_allowed_or_notes")}
    rules = {r["rule_id"]: {"rule": r["rule_text_paraphrase"], "engine": r["engine_translation"]} for r in rows("fma_ue_2026_general_rules.csv")}
    thresholds = {
        "FMA05_abduction_in_synergy_deg": 90, "FMA13_shoulder_flexion_deg": 90, "FMA15_shoulder_abduction_deg": 90, "FMA16_shoulder_flexion_full": "max available (~160 deg reference)",
        "FMA10_elbow_extension_deg": 0, "FMA13_15_16_elbow_contracture_allowed_deg": 30, "FMA19_21_wrist_dorsal_extension_deg": 15, "FMA20_22_min_full_cycles": 2,
        "FMA33_time_diff_s": {"score2_lt": 2.0, "score1_range": [2.0, 5.9], "score0_ge": 6.0}, "FMA31_33_reps": 5, "contracture_cap_fraction_of_normal_range": 0.25,
    }
    return {"source": "FMA2026", "general_rules": rules, "numeric_thresholds": thresholds, "items": items,
            "note": "Webcam output is an 'FMA-style' score for coaching and progress logs; items needing resistance are capped at 1; reflex and hand-to-lumbar items are not measurable."}


def build_exercises(tol):
    T = lambda k: tol[k]["tolerance_deg"]
    M = lambda k: tol[k].get("mdc_default_deg")
    return {
        "shoulder_flexion_raise": {
            "name": "Arm raise to the front (FMA 13 then 16)", "fma_items": ["13", "16"], "side_tested": "affected (left)", "camera_view": "sagittal (chair side-on, affected arm nearest camera)",
            "primary_metric": "shoulder_elevation", "direction": "increase",
            "start_posture": [{"metric": "shoulder_elevation", "op": "<=", "value": 20, "proposed": True}, {"metric": "elbow_flexion", "op": "<=", "value": "elbow_start_allowance"}],
            "form_rules": [{"id": "elbow_straight", "metric": "elbow_flexion", "op": "<=", "value": "start + tolerance(elbow)", "source": "FMA2026", "cue": "Keep your elbow straight."},
                           {"id": "in_plane", "metric": "upper_arm_in_plane_ratio", "op": ">=", "value": 0.87, "proposed": True, "cue": "Lift straight in front of you."}],
            "compensation_rules": ["trunk_flexion_or_lean"],
            "fma_thresholds_deg": {"13": 90, "16": "unaffected-side max (reference ~160)"},
            "milestones": "functional.upper_limb_milestones.shoulder_elevation",
            "ceiling": "min(unaffected_side_max, normative active 160)",
            "tolerance_deg": T("forward_elevation.shoulder"), "mdc_deg": M("forward_elevation.shoulder"),
            "elbow_tolerance_deg": T("forward_elevation.elbow"),
            "life_goal_link": "Reaching a shelf or cupboard.",
        },
        "shoulder_abduction_raise": {
            "name": "Arm raise to the side (FMA 15; FMA 05 in synergy)", "fma_items": ["15", "05"], "camera_view": "frontal", "primary_metric": "shoulder_elevation", "direction": "increase",
            "start_posture": [{"metric": "shoulder_elevation", "op": "<=", "value": 20, "proposed": True}, {"metric": "elbow_flexion", "op": "<=", "value": "elbow_start_allowance"}],
            "form_rules": [{"id": "elbow_straight", "metric": "elbow_flexion", "op": "<=", "value": "start + tolerance(elbow)", "source": "FMA2026", "cue": "Keep your elbow straight."},
                           {"id": "in_plane", "metric": "upper_arm_in_plane_ratio", "op": ">=", "value": 0.87, "proposed": True, "cue": "Lift out to the side."},
                           {"id": "no_shoulder_hike", "metric": "shoulder_girdle_elevation", "op": "<=", "value": 0.15, "proposed": True, "source_rule": "FMA2026", "cue": "Let your shoulder relax down."}],
            "compensation_rules": ["trunk_flexion_or_lean"], "fma_thresholds_deg": {"15": 90, "05": 90},
            "ceiling": "unaffected_side_max", "tolerance_deg": T("abduction.shoulder"), "mdc_deg": M("abduction.shoulder"), "elbow_tolerance_deg": T("forward_elevation.elbow"),
            "life_goal_link": "Putting on a cardigan.",
        },
        "hand_to_mouth": {
            "name": "Cup to mouth (Gates drinking task; Lazem task 4)", "fma_items": ["07 (partial)"], "camera_view": "sagittal", "primary_metric": "elbow_flexion", "direction": "increase",
            "secondary_metrics": ["shoulder_elevation"], "compensation_rules": ["trunk_flexion_or_lean"],
            "targets_deg": {"elbow_flexion": {"value": 121, "ci95": [115, 126], "source": "GATES2016"}, "shoulder_elevation": {"value": 71, "ci95": [67, 75], "source": "GATES2016"}},
            "tolerance_deg": T("hand_to_mouth.elbow"), "mdc_deg": M("hand_to_mouth.elbow"), "shoulder_tolerance_deg": T("hand_to_mouth.shoulder"), "shoulder_mdc_deg": M("hand_to_mouth.shoulder"),
            "safety": "Empty cup only until swallowing advice allows food/drink to mouth.", "life_goal_link": "Drinking your tea.",
        },
        "hand_to_head": {
            "name": "Hand to head / ear (FMA 07 hand touching ear; Lazem task 3)", "fma_items": ["07"], "camera_view": "sagittal", "primary_metric": "elbow_flexion", "direction": "increase",
            "secondary_metrics": ["shoulder_elevation"], "end_condition": {"metric": "wrist_to_ear_distance_norm", "op": "<=", "value": 0.5, "proposed": True, "note": "normalised by shoulder width"},
            "ceiling": "unaffected_side_max", "tolerance_deg": T("hand_to_head.elbow"), "mdc_deg": M("hand_to_head.elbow"), "shoulder_tolerance_deg": T("hand_to_head.shoulder"), "shoulder_mdc_deg": M("hand_to_head.shoulder"),
            "compensation_rules": ["trunk_flexion_or_lean"], "life_goal_link": "Brushing your hair.",
        },
        "elbow_extension": {
            "name": "Straighten the elbow (FMA 10)", "fma_items": ["10"], "camera_view": "sagittal", "primary_metric": "elbow_flexion", "direction": "decrease",
            "fma_thresholds_deg": {"10": 0}, "contracture_rule": "FMA2026: deficit up to 30 deg from contracture = max available; >= 30 deg = not testable",
            "tolerance_deg": T("elbow_flexion_simple"), "mdc_deg": M("elbow_flexion_simple"), "compensation_rules": ["trunk_flexion_or_lean"], "life_goal_link": "Reaching across the table.",
        },
        "wrist_extension": {
            "name": "Wrist up (FMA 19/21 then 20/22)", "fma_items": ["19", "21", "20", "22"], "camera_view": "side view of forearm, elbow ~90 deg (19/20) or ~0 deg (21/22)", "primary_metric": "wrist_extension", "direction": "increase",
            "fma_thresholds_deg": {"19": 15, "21": 15}, "fma_cycles": {"20": 2, "22": 2}, "fma_max_score_webcam": {"19": 1, "21": 1, "20": 2, "22": 2},
            "milestones_deg": [{"deg": 15, "source": "FMA2026"}, {"deg": 33, "task": "Drinking from a cup", "source": "GATES2016"}, {"deg": 40, "task": "all 8 ADLs", "source": "GATES2016"}],
            "form_rules": [{"id": "elbow_steady", "metric": "elbow_flexion", "op": "abs_change<=", "value": "tolerance(elbow)", "source": "FMA2026 (no elbow/shoulder compensation)", "cue": "Keep your elbow still."}],
            "tolerance_deg": T("wrist_flexion_extension"), "mdc_deg": None, "provisional_change_threshold_deg": tol["wrist_flexion_extension"]["provisional_change_threshold_deg"],
            "life_goal_link": "Holding a watering can steady.",
        },
        "hand_open_close": {
            "name": "Bloom: open and close the hand (FMA 25 then 24)", "fma_items": ["25", "24"], "camera_view": "palm facing camera, elbow ~90 deg", "primary_metric": "finger_flexion (MCP, PIP, DIP of digits 2-5)",
            "open_phase": {"fma25_full_extension_max_flexion_deg": "tolerance(finger)", "functional_open_max_flexion_deg": {"MCP": 19, "PIP": 23, "DIP": 10, "source": "BAIN2015"}},
            "close_phase": {"fma24_fingertip_to_palm_max": 0.35, "fma24_threshold_proposed": True, "thumb_outside_fist": True,
                            "functional_close_min_flexion_deg": {"MCP": 71, "PIP": 87, "DIP": 64, "source": "BAIN2015"}},
            "all_fingers_rule": "FMA2026: score 2 needs all 5 fingers; any finger short => 1",
            "normalised_metric": "hand_aperture as % of right-hand calibrated range", "tolerance_deg": T("finger_joints"), "provisional_change_threshold_deg": tol["finger_joints"]["provisional_change_threshold_deg"],
            "life_goal_link": "Opening your hand to hold the watering can.",
        },
        "pinch": {
            "name": "Bubble pinch (FMA 28 position)", "fma_items": ["28"], "fma_max_score_webcam": 1, "primary_metric": "pinch_gap", "direction": "decrease",
            "contact_threshold": {"value": 0.12, "proposed": True}, "rule": "digits 3-5 must not support the pinch (their tips stay away from thumb tip)", "source": "FMA2026",
            "life_goal_link": "Doing up a button.",
        },
        "tabletop_reach": {
            "name": "Light the lamps: reach to near and far targets (RPS)", "fma_items": [], "camera_view": "45 deg between frontal and sagittal on affected side", "source": "LEVIN2004",
            "targets_cm_from_table_edge": {"close": 1, "far": 30}, "setup": "seat 42 cm, backrest, no armrests; table 72 cm; chair at arm's length (wrist crease at 4 cm mark); feet flat; not leaning on backrest",
            "metrics": ["trunk_share_of_reach", "elbow_flexion", "shoulder_elevation"], "scoring": "compensation.reach_trunk_share",
            "life_goal_link": "Reaching for the salt on the table.",
        },
        "finger_to_nose_timed": {
            "name": "Finger to nose x5, both arms timed (FMA 31-33)", "fma_items": ["31", "32", "33"], "camera_view": "frontal", "reps": 5, "order": "non-affected arm first",
            "time_scoring_s": {"2": "diff < 2.0", "1": "2.0 <= diff <= 5.9", "0": "diff >= 6.0 or < 5 reps"}, "source": "FMA2026",
            "adaptation": "Eyes open for Eleanor (MCI, fall risk) => results NOT comparable to clinical FMA 31-33; log as 'adapted'.", "adaptation_proposed": True,
            "dysmetria_thresholds": {"on_nose_max": 0.35, "slight_max": 0.8, "proposed": True, "unit": "nose_touch_error"},
            "timing": "start when wrist leaves knee zone; stop when wrist returns to knee zone after 5th touch",
        },
    }


def build():
    tol = build_tolerance()
    bench = {
        "meta": {"name": "Eleanor rehab coach - movement benchmarks", "version": "1.0", "date": "2026-09-24",
                 "persona": {"name": "Eleanor", "age": 72, "sex": "female", "affected_side": "left", "unaffected_side": "right"},
                 "conventions": {"angles": "degrees; clinical zero (0 = neutral/straight); flexion positive", "sign_in_sources": "GATES2016 extension/external rotation/supination are negative; stored here as positive magnitudes where noted",
                                 "provenance": "every value has `source` or `proposed: true`"},
                 "disclaimer": "Benchmarks adapted from published clinical instruments and normative data for a student prototype. Not a clinical assessment and not validated for this system."},
        "sources": SOURCES,
        "landmarks": {"pose": POSE, "hand": HAND, "pose_side_convention": "MediaPipe Pose left/right = the person's own side; verify hand 'handedness' label during calibration because it depends on whether frames are mirrored"},
        "angle_definitions": ANGLES,
        "signal_processing": {
            "pixel_scaling": "multiply normalised x by frame width and y by frame height before any 2D angle (else angles distort with aspect ratio)",
            "low_pass": {"cutoff_hz": 6, "source": "GATES2016 used a 4th-order low-pass Butterworth at 6 Hz on marker data; use a causal equivalent (e.g., 2nd-order Butterworth or One Euro) in real time"},
            "visibility_min": {"value": 0.5, "proposed": True}, "invalid_frames_pause_s": {"value": 1.0, "proposed": True},
            "state_debounce_frames": {"value": 5, "proposed": True},
            "plausibility": "reject angles above SOUCIE2011 group max + tolerance as tracking errors",
        },
        "measurement_tolerance": tol,
        "layer1_fma": build_fma_layer(),
        "layer2_normative_rom": build_normative(),
        "layer2_functional_rom": build_functional(),
        "layer2_compensation": build_compensation(),
        "layer3_personal_baseline": {
            "calibration_protocol": [
                {"step": "setup check", "detail": "camera 1.5 m from chair, lens ~90 cm high, perpendicular to the movement plane; sleeveless/light clothing; even lighting", "source": ["LAZEM2026", "JAYAVEL2025"]},
                {"step": "seat", "detail": "chair without armrests is FMA standard; an armchair may be used for safety if logged", "source": "FMA2026"},
                {"step": "demo + practice", "detail": "demonstrate, then 2 practice trials", "source": ["FMA2026", "LAZEM2026", "LEVIN2004"]},
                {"step": "unaffected side first", "detail": "3 recorded reps right side, then 3 left side", "source": ["FMA2026", "SOUCIE2011"]},
                {"step": "rest", "detail": "~10 s between reps, ~1 min between tasks", "source": "LAZEM2026"},
                {"step": "store", "detail": "best of 3 (FMA best performance) and mean of 3 (for change detection) per metric and side", "source": "FMA2026"},
                {"step": "therapist limits", "detail": "optional passive ROM / contracture entries cap targets; elbow deficit < 30 deg = max available; >= 30 deg = item not testable", "source": "FMA2026"},
            ],
            "reference_values": {"unaffected_max": "best of 3, right side", "affected_baseline": "best of 3, left side, session 0", "symmetry_index": "affected / unaffected (Soucie: healthy left-right differences < 1 deg)"},
            "target_ladder": {
                "formula": "target_k = min(ceiling, affected_baseline + k * step); ceiling = min(unaffected_max, therapist_passive_limit, normative_ceiling)",
                "step_deg": {"value": "tolerance_deg of that joint/task", "proposed": True, "why": "a step smaller than measurement tolerance cannot be detected reliably"},
                "named_milestones": "insert FMA thresholds (e.g., 90 deg) and GATES2016/BAIN2015 task values that lie between baseline and ceiling; announce when crossed",
            },
            "progression_rules": {"proposed": True, "origin": "project Think-module defaults (Eleanor Extended Program doc), not from the reviewed papers",
                                  "rules": [{"if": "clean success >= 80% in 2 sessions in a row, pain not higher, compensation not rising", "then": "level +1"},
                                            {"if": "success 50-79%", "then": "stay"},
                                            {"if": "success < 50% OR compensation in most reps", "then": "level -1"},
                                            {"if": "pain +2 or new shoulder pain", "then": "pause exercise; flag therapist"},
                                            {"if": "2 missed sessions", "then": "restart one level lower"}]},
        },
        "scoring": {
            "modes": {"assessment": "1-3 attempts per item, best attempt counts, weekly (frequency proposed)", "training": "sets of reps with coaching feedback", "source": "FMA2026"},
            "rep_fma_style_score": [
                "0: movement from start < tolerance, OR start position cannot be obtained where the item requires it",
                "1: moved, but peak < FMA threshold, OR |peak - threshold| <= tolerance (uncertain => lower score, FMA2026), OR form rule broken before threshold, OR any compensation flag",
                "2: peak > threshold + tolerance (outside the uncertainty band), all form rules held, no compensation",
            ],
            "rep_coaching_success": "peak >= personal_target - tolerance AND form rules held; compensation => success_with_cue (counts, but logged and slows progression)",
            "progress_claim_rule": "say 'you improved' only if session value (best of 3) minus reference >= MDC for that task/joint; otherwise praise effort/consistency",
            "feedback_map": {"2": "specific praise naming the gain or the life goal", "1": "one cue about the failed form rule, or 'a little higher'", "0": "encouragement + replay demo", "tracking_lost": "blame the system: 'I can't see your arm, move into the box'"},
            "elbow_start_allowance": "start elbow flexion must be <= max(tolerance(elbow), therapist contracture deficit < 30 deg) (FMA2026: full extension of available passive range at start)",
            "max_technique_cues_per_set": {"value": 1, "proposed": True, "origin": "project design rule"},
        },
        "exercises": build_exercises(tol),
    }
    with open(os.path.join(ROOT, "benchmarks.json"), "w") as fh:
        json.dump(bench, fh, indent=2, ensure_ascii=False)
    return bench


if __name__ == "__main__":
    b = build()
    print("wrote benchmarks.json;", len(b["exercises"]), "exercises;", len(b["measurement_tolerance"]), "tolerance entries")
