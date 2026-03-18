# -*- coding: utf-8 -*-
"""
Baseline EEG training session (imperative style).

Design:
- 16 blocks total, alternating task type
- 8 RSA blocks x 125 trials = 1000 RSA trials
- 8 SD blocks x 30 trials = 240 SD trials

Outputs:
- rsa_anchor_stimuli.csv
- sd_anchor_stimuli.csv
- sd_pair_table.csv
- rsa_trials_<participant>_<sessionday>_<datetime>.csv
- sd_trials_<participant>_<sessionday>_<datetime>.csv
"""

from datetime import datetime
import os

import numpy as np
import pandas as pd
from psychopy import core, visual  # type: ignore
from psychopy.hardware import keyboard  # type: ignore

from util_func import make_stim_cats


# -----------------------------
# Fixed settings
# -----------------------------
SCHEMA_VERSION = "1.0.0"
SOURCE_MODE = "baseline_eeg_training"

DATA_DIR = "../data_baseline"
SEED = 20260318

EEG_ENABLED = False
EEG_PORT_ADDRESS = "0x3FD8"
EEG_DEFAULT_PULSE_MS = 10

N_CANDIDATES_PER_CATEGORY = 4000
N_RSA_UNIQUE = 50
N_RSA_TOTAL = 1000
N_SD_UNIQUE = 60
N_SD_TRIALS_TOTAL = 240
N_SD_PRACTICE_UNIQUE = 20
MAKE_PRACTICE_POOL = True

N_RSA_BLOCKS = 8
N_SD_BLOCKS = 8
RSA_TRIALS_PER_BLOCK = 125
SD_TRIALS_PER_BLOCK = 30

WIN_SIZE = (1920, 1080)
FULLSCR = True
WIN_COLOR = (0.494, 0.494, 0.494)
GRATING_SIZE_CM = 5.0

RSA_SOA_MS = 500
RSA_STIM_MS = 250

SD_FIX_MS = 500
SD_STIM1_MS = 120
SD_ISI_MS = 250
SD_STIM2_MS = 120
SD_RESP_WINDOW_MS = 2000
SD_ITI_MIN_MS = 400
SD_ITI_MAX_MS = 800

# Default placeholders; actual SD mapping is set per participant from randomized tables below.
KEY_SAME = "q"
KEY_DIFF = "p"

PARTICIPANT_IDS = [
    "002", "077", "134", "189", "213", "268", "303", "358", "482",
    "527", "594", "639", "662", "707", "729", "875", "943", "998",
]

# Randomized, balanced key-map assignment built once from the participant list.
_rng_keymap = np.random.default_rng(SEED + 9001)
_ids_shuffled = PARTICIPANT_IDS.copy()
_rng_keymap.shuffle(_ids_shuffled)
_half = len(_ids_shuffled) // 2
KEYMAP_A_IDS = set(_ids_shuffled[:_half])   # Q=Same, P=Different
KEYMAP_B_IDS = set(_ids_shuffled[_half:])   # P=Same, Q=Different

ALLOWED_IDS = set(PARTICIPANT_IDS)

TRIG = {
    "EXP_START": 10,
    "EXP_END": 15,
    "BLOCK_START_RSA": 16,
    "BLOCK_END_RSA": 17,
    "BLOCK_START_SD": 18,
    "BLOCK_END_SD": 19,
    "TRIAL_START": 20,
    "RSA_STIM_ONSET": 21,
    "SD_STIM1_ONSET": 22,
    "SD_STIM2_ONSET": 23,
    "RESP_SAME": 30,
    "RESP_DIFFERENT": 31,
}


# -----------------------------
# Minimal EEG helper (kept for port safety)
# -----------------------------
class EEGPort:
    def __init__(self, win, address, enabled, default_ms):
        self.win = win
        self.enabled = enabled
        self.default_ms = default_ms
        self._port = None
        self._clear_at = None

        if not self.enabled:
            return

        try:
            from psychopy import parallel  # type: ignore
            self._port = parallel.ParallelPort(address=address)
        except Exception as exc:
            print(f"[EEG] Parallel port unavailable ({exc}). Running without triggers.")
            self.enabled = False
            self._port = None

    def flip_pulse(self, code, global_clock=None, width_ms=None):
        if not (self.enabled and self._port):
            return
        if width_ms is None:
            width_ms = self.default_ms
        self.win.callOnFlip(self._port.setData, int(code) & 0xFF)
        if global_clock is not None:
            self._clear_at = global_clock.getTime() + (width_ms / 1000.0)

    def pulse_now(self, code, global_clock=None, width_ms=None):
        if not (self.enabled and self._port):
            return
        if width_ms is None:
            width_ms = self.default_ms
        self._port.setData(int(code) & 0xFF)
        if global_clock is not None:
            self._clear_at = global_clock.getTime() + (width_ms / 1000.0)

    def update(self, global_clock=None):
        if not (self.enabled and self._port):
            return
        if self._clear_at is not None and global_clock is not None:
            if global_clock.getTime() >= self._clear_at:
                self._port.setData(0)
                self._clear_at = None

    def close(self):
        try:
            if self._port:
                self._port.setData(0)
        except Exception:
            pass


# -----------------------------
# Small utility for on-screen text entry
# -----------------------------
def text_entry_screen(win, kb_task, kb_default, title, subtitle=""):
    value = ""
    title_stim = visual.TextStim(win, text=title, color="white", height=36, pos=(0, 120), wrapWidth=1600)
    subtitle_stim = visual.TextStim(win, text=subtitle, color="white", height=24, pos=(0, 50), wrapWidth=1600)
    value_stim = visual.TextStim(win, text="", color="white", height=40, pos=(0, -30), wrapWidth=1600)
    hint_stim = visual.TextStim(
        win,
        text="Type, BACKSPACE to edit, ENTER to confirm, ESC to quit",
        color="white",
        height=22,
        pos=(0, -120),
        wrapWidth=1600,
    )

    while True:
        title_stim.draw()
        subtitle_stim.draw()
        value_stim.text = value if value else "_"
        value_stim.draw()
        hint_stim.draw()
        win.flip()

        if kb_default.getKeys(keyList=["escape"], waitRelease=False):
            raise KeyboardInterrupt("Escape pressed")

        keys = kb_task.getKeys(waitRelease=False)
        if not keys:
            core.wait(0.01)
            continue

        for key in keys:
            name = key.name
            if name in ("return", "num_enter"):
                if value.strip() != "":
                    return value.strip()
            elif name == "backspace":
                value = value[:-1]
            elif name == "space":
                value += " "
            elif len(name) == 1:
                value += name


# -----------------------------
# Main top-to-bottom flow
# -----------------------------
if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    rng = np.random.default_rng(SEED)

    # ---- Build/load anchor tables
    rsa_anchor_path = os.path.join(DATA_DIR, "rsa_anchor_stimuli.csv")
    sd_anchor_path = os.path.join(DATA_DIR, "sd_anchor_stimuli.csv")

    if os.path.exists(rsa_anchor_path) and os.path.exists(sd_anchor_path):
        rsa_anchors = pd.read_csv(rsa_anchor_path)
        sd_anchors = pd.read_csv(sd_anchor_path)
    else:
        ds, _, _ = make_stim_cats(N_CANDIDATES_PER_CATEGORY)
        ds = ds[["x", "y", "xt", "yt"]].copy().reset_index(drop=True)

        # Add category tags for 0/90/180 with explicit loops
        cat_train = []
        cat_rot90 = []
        cat_rot180 = []

        for _, row in ds.iterrows():
            x = float(row["x"])
            y = float(row["y"])

            # no rotation means at (40,60) and (60,40)
            da0 = (x - 40.0) ** 2 + (y - 60.0) ** 2
            db0 = (x - 60.0) ** 2 + (y - 40.0) ** 2
            cat_train.append("A" if da0 <= db0 else "B")

            # 90 rotation around center (50,50)
            ax90 = 50.0 - (60.0 - 50.0)
            ay90 = 50.0 + (40.0 - 50.0)
            bx90 = 50.0 - (40.0 - 50.0)
            by90 = 50.0 + (60.0 - 50.0)
            da90 = (x - ax90) ** 2 + (y - ay90) ** 2
            db90 = (x - bx90) ** 2 + (y - by90) ** 2
            cat_rot90.append("A" if da90 <= db90 else "B")

            # 180 rotation around center (50,50)
            ax180 = 100.0 - 40.0
            ay180 = 100.0 - 60.0
            bx180 = 100.0 - 60.0
            by180 = 100.0 - 40.0
            da180 = (x - ax180) ** 2 + (y - ay180) ** 2
            db180 = (x - bx180) ** 2 + (y - by180) ** 2
            cat_rot180.append("A" if da180 <= db180 else "B")

        ds["cat_train"] = cat_train
        ds["cat_rot90"] = cat_rot90
        ds["cat_rot180"] = cat_rot180

        # Max-min uniform sample for RSA anchors (explicit)
        feats = ds[["xt", "yt"]].to_numpy(dtype=float)
        mu = feats.mean(axis=0)
        sdv = feats.std(axis=0)
        sdv[sdv == 0] = 1.0
        z = (feats - mu) / sdv

        start_idx = int(rng.integers(0, len(ds)))
        picked = [start_idx]
        dist2 = np.sum((z - z[start_idx]) ** 2, axis=1)
        for _ in range(1, N_RSA_UNIQUE):
            next_idx = int(np.argmax(dist2))
            picked.append(next_idx)
            d2_new = np.sum((z - z[next_idx]) ** 2, axis=1)
            dist2 = np.minimum(dist2, d2_new)

        rsa_anchors = ds.iloc[picked].copy().reset_index(drop=True)
        rsa_anchors.insert(0, "stim_id", [f"rsa_{i:03d}" for i in range(len(rsa_anchors))])
        rsa_anchors["seed"] = SEED
        rsa_anchors["created_at_iso"] = datetime.now().isoformat()

        # Remaining pool for SD
        remaining = ds.merge(rsa_anchors[["x", "y"]], on=["x", "y"], how="left", indicator=True)
        remaining = remaining.loc[
            remaining["_merge"] == "left_only",
            ["x", "y", "xt", "yt", "cat_train", "cat_rot90", "cat_rot180"],
        ].reset_index(drop=True)

        feats_sd = remaining[["xt", "yt"]].to_numpy(dtype=float)
        mu_sd = feats_sd.mean(axis=0)
        sd_sd = feats_sd.std(axis=0)
        sd_sd[sd_sd == 0] = 1.0
        z_sd = (feats_sd - mu_sd) / sd_sd

        start_idx_sd = int(rng.integers(0, len(remaining)))
        picked_sd = [start_idx_sd]
        dist2_sd = np.sum((z_sd - z_sd[start_idx_sd]) ** 2, axis=1)
        for _ in range(1, N_SD_UNIQUE):
            next_idx = int(np.argmax(dist2_sd))
            picked_sd.append(next_idx)
            d2_new_sd = np.sum((z_sd - z_sd[next_idx]) ** 2, axis=1)
            dist2_sd = np.minimum(dist2_sd, d2_new_sd)

        sd_main = remaining.iloc[picked_sd].copy().reset_index(drop=True)
        sd_main.insert(0, "stim_id", [f"sd_{i:03d}" for i in range(len(sd_main))])
        sd_main["pool_type"] = "main"

        if MAKE_PRACTICE_POOL and N_SD_PRACTICE_UNIQUE > 0:
            rem2 = remaining.merge(sd_main[["x", "y"]], on=["x", "y"], how="left", indicator=True)
            rem2 = rem2.loc[
                rem2["_merge"] == "left_only",
                ["x", "y", "xt", "yt", "cat_train", "cat_rot90", "cat_rot180"],
            ].reset_index(drop=True)

            feats_pr = rem2[["xt", "yt"]].to_numpy(dtype=float)
            mu_pr = feats_pr.mean(axis=0)
            sd_pr = feats_pr.std(axis=0)
            sd_pr[sd_pr == 0] = 1.0
            z_pr = (feats_pr - mu_pr) / sd_pr

            start_idx_pr = int(rng.integers(0, len(rem2)))
            picked_pr = [start_idx_pr]
            dist2_pr = np.sum((z_pr - z_pr[start_idx_pr]) ** 2, axis=1)
            for _ in range(1, min(N_SD_PRACTICE_UNIQUE, len(rem2))):
                next_idx = int(np.argmax(dist2_pr))
                picked_pr.append(next_idx)
                d2_new_pr = np.sum((z_pr - z_pr[next_idx]) ** 2, axis=1)
                dist2_pr = np.minimum(dist2_pr, d2_new_pr)

            sd_practice = rem2.iloc[picked_pr].copy().reset_index(drop=True)
            sd_practice.insert(0, "stim_id", [f"sdp_{i:03d}" for i in range(len(sd_practice))])
            sd_practice["pool_type"] = "practice"
            sd_anchors = pd.concat([sd_main, sd_practice], ignore_index=True)
        else:
            sd_anchors = sd_main.copy()

        sd_anchors["seed"] = SEED
        sd_anchors["created_at_iso"] = datetime.now().isoformat()

        rsa_anchors.to_csv(rsa_anchor_path, index=False)
        sd_anchors.to_csv(sd_anchor_path, index=False)

    # ---- Build/load SD pair table
    sd_pair_path = os.path.join(DATA_DIR, "sd_pair_table.csv")
    if os.path.exists(sd_pair_path):
        sd_pairs = pd.read_csv(sd_pair_path)
    else:
        sd_main = sd_anchors.loc[sd_anchors["pool_type"] == "main"].copy().reset_index(drop=True)
        if len(sd_main) < 60:
            raise ValueError("Need at least 60 SD anchors in main pool.")

        # Same trials: exactly 60
        pair_rows = []
        for i in range(60):
            sid = sd_main.iloc[i]["stim_id"]
            pair_rows.append({
                "pair_id": f"sdpair_{i:04d}",
                "stim1_id": sid,
                "stim2_id": sid,
                "pair_type": "same",
                "distance_metric": "euclidean_z_xt_yt",
                "distance_value": 0.0,
            })

        # Different pairs and distance bins
        feat = sd_main[["xt", "yt"]].to_numpy(dtype=float)
        mu = feat.mean(axis=0)
        sdv = feat.std(axis=0)
        sdv[sdv == 0] = 1.0
        z = (feat - mu) / sdv

        all_pairs = []
        n = len(sd_main)
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(z[i] - z[j]))
                all_pairs.append((i, j, d))

        diff_df = pd.DataFrame(all_pairs, columns=["i", "j", "dist"])
        q1 = float(diff_df["dist"].quantile(1 / 3))
        q2 = float(diff_df["dist"].quantile(2 / 3))

        near_df = diff_df.loc[diff_df["dist"] <= q1].copy()
        mid_df = diff_df.loc[(diff_df["dist"] > q1) & (diff_df["dist"] <= q2)].copy()
        far_df = diff_df.loc[diff_df["dist"] > q2].copy()

        near_take = near_df.sample(n=60, random_state=SEED + 11)
        mid_take = mid_df.sample(n=60, random_state=SEED + 12)
        far_take = far_df.sample(n=60, random_state=SEED + 13)

        diff_take = pd.concat([near_take, mid_take, far_take], ignore_index=True)
        labels = (["near"] * 60) + (["mid"] * 60) + (["far"] * 60)

        for k in range(len(diff_take)):
            i = int(diff_take.iloc[k]["i"])
            j = int(diff_take.iloc[k]["j"])
            d = float(diff_take.iloc[k]["dist"])
            pair_rows.append({
                "pair_id": f"sdpair_{60 + k:04d}",
                "stim1_id": sd_main.iloc[i]["stim_id"],
                "stim2_id": sd_main.iloc[j]["stim_id"],
                "pair_type": labels[k],
                "distance_metric": "euclidean_z_xt_yt",
                "distance_value": d,
            })

        sd_pairs = pd.DataFrame(pair_rows)
        sd_pairs["distance_bin_rule"] = f"terciles_main_pool_q1={q1:.6f}_q2={q2:.6f}"
        sd_pairs["seed"] = SEED + 1
        sd_pairs["created_at_iso"] = datetime.now().isoformat()
        sd_pairs = sd_pairs.sample(frac=1, random_state=SEED + 2).reset_index(drop=True)
        sd_pairs.to_csv(sd_pair_path, index=False)

    # ---- Make full RSA trial list (50 x 20 = 1000)
    n_unique = len(rsa_anchors)
    repeats = N_RSA_TOTAL // n_unique
    remainder = N_RSA_TOTAL % n_unique
    id_list = []
    for i in range(n_unique):
        k = repeats + (1 if i < remainder else 0)
        for _ in range(k):
            id_list.append(i)
    rng.shuffle(id_list)
    rsa_trials_all = rsa_anchors.iloc[id_list].copy().reset_index(drop=True)

    if N_RSA_BLOCKS * RSA_TRIALS_PER_BLOCK != N_RSA_TOTAL:
        raise ValueError("RSA blocks do not sum to 1000 trials.")
    if N_SD_BLOCKS * SD_TRIALS_PER_BLOCK != N_SD_TRIALS_TOTAL:
        raise ValueError("SD blocks do not sum to 240 trials.")

    # ---- PsychoPy window and stimuli
    pixels_per_inch = 227 / 2
    px_per_cm = pixels_per_inch / 2.54
    size_px = int(GRATING_SIZE_CM * px_per_cm)

    win = visual.Window(
        size=WIN_SIZE,
        fullscr=FULLSCR,
        units="pix",
        color=WIN_COLOR,
        colorSpace="rgb",
        winType="pyglet",
        useRetina=True,
        waitBlanking=True,
    )
    win.mouseVisible = False

    grating = visual.GratingStim(
        win,
        tex="sin",
        mask="circle",
        interpolate=True,
        size=(size_px, size_px),
        units="pix",
        sf=0.02,
        ori=0.0,
    )

    fix_h = visual.Line(win, start=(0, -10), end=(0, 10), lineColor="white", lineWidth=8)
    fix_v = visual.Line(win, start=(-10, 0), end=(10, 0), lineColor="white", lineWidth=8)

    prompt_text = visual.TextStim(
        win,
        text="Same / Different",
        color="white",
        height=32,
    )

    kb_task = keyboard.Keyboard()
    kb_default = keyboard.Keyboard()
    global_clock = core.Clock()
    eeg = EEGPort(win, EEG_PORT_ADDRESS, EEG_ENABLED, EEG_DEFAULT_PULSE_MS)

    # ---- On-screen participant/session entry
    participant_id = ""
    session_day = -1

    try:
        while participant_id == "":
            candidate_id = text_entry_screen(win, kb_task, kb_default, "Enter participant number", "Press ENTER when done")
            if candidate_id in ALLOWED_IDS:
                participant_id = candidate_id
            else:
                err = visual.TextStim(
                    win,
                    text=(
                        f"Participant ID '{candidate_id}' is not in the allowed list.\n\n"
                        "Please re-enter a valid 3-digit participant number.\n\n"
                        "Press SPACE to continue."
                    ),
                    color="white",
                    height=28,
                    wrapWidth=1500,
                )
                waiting = True
                while waiting:
                    err.draw()
                    win.flip()
                    if kb_default.getKeys(keyList=["escape"], waitRelease=False):
                        raise KeyboardInterrupt("Escape pressed")
                    if kb_task.getKeys(keyList=["space"], waitRelease=False):
                        waiting = False
                    core.wait(0.01)

        while session_day not in (0, 21):
            day_text = text_entry_screen(win, kb_task, kb_default, "Enter session day", "Valid values: 0 or 21")
            try:
                session_day = int(day_text)
            except ValueError:
                session_day = -1

        session_dt = datetime.now()
        session_datetime_iso = session_dt.isoformat()
        run_id = f"{participant_id}_{session_dt.strftime('%Y%m%d_%H%M%S')}"

        # ---- Task order: always RSA first
        sequence_label = "RSA_FIRST"

        if participant_id in KEYMAP_A_IDS:
            key_same = "q"
            key_diff = "p"
            keymap_label = "A_Q_SAME_P_DIFF"
        else:
            key_same = "p"
            key_diff = "q"
            keymap_label = "B_P_SAME_Q_DIFF"

        prompt_text.text = f"Same ({key_same.upper()}) / Different ({key_diff.upper()})"

        total_blocks = N_RSA_BLOCKS + N_SD_BLOCKS
        sequence = []
        for i in range(total_blocks):
            sequence.append("RSA" if i % 2 == 0 else "SD")

        # ---- Start instruction screen
        inst_text = visual.TextStim(
            win,
            text=(
                "Press SPACE to begin.\n\n"
                "Alternating RSA and same/different blocks.\n"
                "No feedback is provided.\n"
                f"Same key: {key_same.upper()}    Different key: {key_diff.upper()}\n"
                "Press ESC at any time to stop."
            ),
            color="white",
            height=30,
            wrapWidth=1400,
        )

        waiting = True
        while waiting:
            inst_text.draw()
            win.flip()
            if kb_default.getKeys(keyList=["escape"], waitRelease=False):
                raise KeyboardInterrupt("Escape pressed")
            if kb_task.getKeys(keyList=["space"], waitRelease=False):
                waiting = False
            core.wait(0.01)

        # ---- Precompute SD anchor lookup
        sd_main = sd_anchors.loc[sd_anchors["pool_type"] == "main"].copy().reset_index(drop=True)
        anchors_by_id = {}
        for _, row in sd_main.iterrows():
            anchors_by_id[str(row["stim_id"])] = row.to_dict()

        rsa_rows = []
        sd_rows = []

        exp_start_time = np.nan
        exp_end_time = np.nan
        exp_start_code = np.nan
        exp_end_code = np.nan

        aborted = False

        # ---- EXP start trigger
        eeg.flip_pulse(TRIG["EXP_START"], global_clock=global_clock)
        win.flip()
        exp_start_time = float(global_clock.getTime())
        exp_start_code = int(TRIG["EXP_START"])

        rsa_ptr = 0
        sd_ptr = 0
        trial_index_global = 0

        # ---- Run alternating blocks
        for b_idx, task in enumerate(sequence, start=1):
            if task == "RSA":
                # Block start
                eeg.flip_pulse(TRIG["BLOCK_START_RSA"], global_clock=global_clock)
                win.flip()
                block_start_time = float(global_clock.getTime())

                # Slice and randomize within block
                block_trials = rsa_trials_all.iloc[rsa_ptr:rsa_ptr + RSA_TRIALS_PER_BLOCK].copy().reset_index(drop=True)
                block_trials = block_trials.sample(frac=1, random_state=SEED + 1000 + b_idx).reset_index(drop=True)

                # Trial loop
                for t_idx, trial in block_trials.iterrows():
                    eeg.flip_pulse(TRIG["TRIAL_START"], global_clock=global_clock)
                    # Intentional consistent blank before each RSA stim
                    win.flip()
                    trial_start_time = float(global_clock.getTime())

                    sf_cycles_per_pix = float(trial["xt"]) / px_per_cm
                    ori_deg = float(trial["yt"]) * 180.0 / np.pi
                    grating.sf = sf_cycles_per_pix
                    grating.ori = ori_deg
                    grating.draw()

                    eeg.flip_pulse(TRIG["RSA_STIM_ONSET"], global_clock=global_clock)
                    win.flip()
                    stim_onset_time = float(global_clock.getTime())

                    # Stim on duration
                    t0 = global_clock.getTime()
                    while (global_clock.getTime() - t0) < (RSA_STIM_MS / 1000.0):
                        if kb_default.getKeys(keyList=["escape"], waitRelease=False):
                            raise KeyboardInterrupt("Escape pressed")
                        eeg.update(global_clock)
                        core.wait(0.005)

                    # Blank for remaining SOA
                    win.flip()
                    blank_ms = max(0, RSA_SOA_MS - RSA_STIM_MS)
                    t0 = global_clock.getTime()
                    while (global_clock.getTime() - t0) < (blank_ms / 1000.0):
                        if kb_default.getKeys(keyList=["escape"], waitRelease=False):
                            raise KeyboardInterrupt("Escape pressed")
                        eeg.update(global_clock)
                        core.wait(0.005)

                    rsa_rows.append({
                        "participant_id": participant_id,
                        "session_day": session_day,
                        "session_datetime_iso": session_datetime_iso,
                        "run_id": run_id,
                        "sequence_label": sequence_label,
                        "key_same": key_same,
                        "key_diff": key_diff,
                        "keymap_label": keymap_label,
                        "block_index": b_idx,
                        "block_type": "RSA",
                        "trial_index": int(t_idx),
                        "trial_index_global": int(trial_index_global),
                        "trial_start_time": trial_start_time,
                        "stim_id": trial["stim_id"],
                        "x": float(trial["x"]),
                        "y": float(trial["y"]),
                        "xt": float(trial["xt"]),
                        "yt": float(trial["yt"]),
                        "cat_train": trial["cat_train"],
                        "cat_rot90": trial["cat_rot90"],
                        "cat_rot180": trial["cat_rot180"],
                        "soa_ms": int(RSA_SOA_MS),
                        "stim_onset_time": stim_onset_time,
                        "block_start_time": block_start_time,
                        "block_end_time": np.nan,
                        "trigger_block_start": int(TRIG["BLOCK_START_RSA"]),
                        "trigger_block_end": int(TRIG["BLOCK_END_RSA"]),
                        "trigger_trial_start": int(TRIG["TRIAL_START"]),
                        "trigger_stim_onset": int(TRIG["RSA_STIM_ONSET"]),
                        "trigger_exp_start": np.nan,
                        "trigger_exp_end": np.nan,
                        "exp_start_time": np.nan,
                        "exp_end_time": np.nan,
                        "eeg_enabled": int(bool(EEG_ENABLED)),
                        "schema_version": SCHEMA_VERSION,
                    })

                    trial_index_global += 1

                rsa_ptr += RSA_TRIALS_PER_BLOCK

                # Block end
                eeg.flip_pulse(TRIG["BLOCK_END_RSA"], global_clock=global_clock)
                win.flip()
                block_end_time = float(global_clock.getTime())
                for row in rsa_rows:
                    if row["block_index"] == b_idx and np.isnan(row["block_end_time"]):
                        row["block_end_time"] = block_end_time

            else:
                # Block start
                eeg.flip_pulse(TRIG["BLOCK_START_SD"], global_clock=global_clock)
                win.flip()
                block_start_time = float(global_clock.getTime())

                # Slice and randomize within block
                block_trials = sd_pairs.iloc[sd_ptr:sd_ptr + SD_TRIALS_PER_BLOCK].copy().reset_index(drop=True)
                block_trials = block_trials.sample(frac=1, random_state=SEED + 2000 + b_idx).reset_index(drop=True)

                for t_idx, trial in block_trials.iterrows():
                    stim1 = anchors_by_id[str(trial["stim1_id"])]
                    stim2 = anchors_by_id[str(trial["stim2_id"])]

                    eeg.flip_pulse(TRIG["TRIAL_START"], global_clock=global_clock)
                    fix_h.draw()
                    fix_v.draw()
                    win.flip()
                    trial_start_time = float(global_clock.getTime())

                    t0 = global_clock.getTime()
                    while (global_clock.getTime() - t0) < (SD_FIX_MS / 1000.0):
                        if kb_default.getKeys(keyList=["escape"], waitRelease=False):
                            raise KeyboardInterrupt("Escape pressed")
                        eeg.update(global_clock)
                        core.wait(0.005)

                    sf1 = float(stim1["xt"]) / px_per_cm
                    ori1 = float(stim1["yt"]) * 180.0 / np.pi
                    grating.sf = sf1
                    grating.ori = ori1
                    grating.draw()
                    eeg.flip_pulse(TRIG["SD_STIM1_ONSET"], global_clock=global_clock)
                    win.flip()
                    stim1_onset_time = float(global_clock.getTime())

                    t0 = global_clock.getTime()
                    while (global_clock.getTime() - t0) < (SD_STIM1_MS / 1000.0):
                        if kb_default.getKeys(keyList=["escape"], waitRelease=False):
                            raise KeyboardInterrupt("Escape pressed")
                        eeg.update(global_clock)
                        core.wait(0.005)

                    win.flip()
                    t0 = global_clock.getTime()
                    while (global_clock.getTime() - t0) < (SD_ISI_MS / 1000.0):
                        if kb_default.getKeys(keyList=["escape"], waitRelease=False):
                            raise KeyboardInterrupt("Escape pressed")
                        eeg.update(global_clock)
                        core.wait(0.005)

                    sf2 = float(stim2["xt"]) / px_per_cm
                    ori2 = float(stim2["yt"]) * 180.0 / np.pi
                    grating.sf = sf2
                    grating.ori = ori2
                    grating.draw()
                    eeg.flip_pulse(TRIG["SD_STIM2_ONSET"], global_clock=global_clock)
                    win.flip()
                    stim2_onset_time = float(global_clock.getTime())

                    t0 = global_clock.getTime()
                    while (global_clock.getTime() - t0) < (SD_STIM2_MS / 1000.0):
                        if kb_default.getKeys(keyList=["escape"], waitRelease=False):
                            raise KeyboardInterrupt("Escape pressed")
                        eeg.update(global_clock)
                        core.wait(0.005)

                    kb_task.clearEvents()
                    prompt_text.draw()
                    win.flip()

                    response_key = ""
                    response_label = ""
                    rt_ms = np.nan
                    response_time = np.nan
                    trig_response = np.nan

                    correct_answer = "same" if str(trial["pair_type"]) == "same" else "different"
                    accuracy = 0

                    resp_clock = core.Clock()
                    responded = False
                    while resp_clock.getTime() * 1000.0 < SD_RESP_WINDOW_MS:
                        if kb_default.getKeys(keyList=["escape"], waitRelease=False):
                            raise KeyboardInterrupt("Escape pressed")

                        eeg.update(global_clock)
                        keys = kb_task.getKeys(keyList=[key_same, key_diff], waitRelease=False)
                        if keys:
                            k = keys[-1]
                            response_key = k.name
                            rt_ms = float(k.rt * 1000.0)
                            response_label = "same" if k.name == key_same else "different"
                            accuracy = int(response_label == correct_answer)
                            trig_response = TRIG["RESP_SAME"] if response_label == "same" else TRIG["RESP_DIFFERENT"]
                            eeg.pulse_now(trig_response, global_clock=global_clock)
                            response_time = float(global_clock.getTime())
                            responded = True
                            break
                        core.wait(0.005)

                    if not responded:
                        response_label = "no_response"
                        accuracy = 0

                    win.flip()
                    iti_ms = int(np.random.randint(SD_ITI_MIN_MS, SD_ITI_MAX_MS + 1))
                    t0 = global_clock.getTime()
                    while (global_clock.getTime() - t0) < (iti_ms / 1000.0):
                        if kb_default.getKeys(keyList=["escape"], waitRelease=False):
                            raise KeyboardInterrupt("Escape pressed")
                        eeg.update(global_clock)
                        core.wait(0.005)

                    sd_rows.append({
                        "participant_id": participant_id,
                        "session_day": session_day,
                        "session_datetime_iso": session_datetime_iso,
                        "run_id": run_id,
                        "sequence_label": sequence_label,
                        "key_same": key_same,
                        "key_diff": key_diff,
                        "keymap_label": keymap_label,
                        "block_index": b_idx,
                        "block_type": "SD",
                        "trial_index": int(t_idx),
                        "trial_index_global": int(trial_index_global),
                        "trial_start_time": trial_start_time,
                        "pair_id": trial["pair_id"],
                        "stim1_id": trial["stim1_id"],
                        "stim2_id": trial["stim2_id"],
                        "stim1_x": float(stim1["x"]),
                        "stim1_y": float(stim1["y"]),
                        "stim1_xt": float(stim1["xt"]),
                        "stim1_yt": float(stim1["yt"]),
                        "stim2_x": float(stim2["x"]),
                        "stim2_y": float(stim2["y"]),
                        "stim2_xt": float(stim2["xt"]),
                        "stim2_yt": float(stim2["yt"]),
                        "pair_type": trial["pair_type"],
                        "distance_value": float(trial["distance_value"]),
                        "correct_answer": correct_answer,
                        "response_key": response_key,
                        "response_label": response_label,
                        "accuracy": int(accuracy),
                        "rt_ms": rt_ms,
                        "stim1_onset_time": stim1_onset_time,
                        "stim2_onset_time": stim2_onset_time,
                        "response_time": response_time,
                        "fixation_ms": int(SD_FIX_MS),
                        "stim1_ms": int(SD_STIM1_MS),
                        "isi_ms": int(SD_ISI_MS),
                        "stim2_ms": int(SD_STIM2_MS),
                        "response_window_ms": int(SD_RESP_WINDOW_MS),
                        "iti_ms": int(iti_ms),
                        "block_start_time": block_start_time,
                        "block_end_time": np.nan,
                        "trigger_block_start": int(TRIG["BLOCK_START_SD"]),
                        "trigger_block_end": int(TRIG["BLOCK_END_SD"]),
                        "trigger_trial_start": int(TRIG["TRIAL_START"]),
                        "trigger_stim1_onset": int(TRIG["SD_STIM1_ONSET"]),
                        "trigger_stim2_onset": int(TRIG["SD_STIM2_ONSET"]),
                        "trigger_response": trig_response,
                        "trigger_exp_start": np.nan,
                        "trigger_exp_end": np.nan,
                        "exp_start_time": np.nan,
                        "exp_end_time": np.nan,
                        "eeg_enabled": int(bool(EEG_ENABLED)),
                        "schema_version": SCHEMA_VERSION,
                    })

                    trial_index_global += 1

                sd_ptr += SD_TRIALS_PER_BLOCK

                # Block end
                eeg.flip_pulse(TRIG["BLOCK_END_SD"], global_clock=global_clock)
                win.flip()
                block_end_time = float(global_clock.getTime())
                for row in sd_rows:
                    if row["block_index"] == b_idx and np.isnan(row["block_end_time"]):
                        row["block_end_time"] = block_end_time

            # Break screen between blocks
            if b_idx < len(sequence):
                break_stim = visual.TextStim(
                    win,
                    text=f"End of block {b_idx}/{len(sequence)}.\n\nPress SPACE to continue.",
                    color="white",
                    height=32,
                    wrapWidth=1400,
                )
                waiting = True
                while waiting:
                    break_stim.draw()
                    win.flip()
                    if kb_default.getKeys(keyList=["escape"], waitRelease=False):
                        raise KeyboardInterrupt("Escape pressed")
                    if kb_task.getKeys(keyList=["space"], waitRelease=False):
                        waiting = False
                    core.wait(0.01)

        # ---- EXP end trigger
        eeg.flip_pulse(TRIG["EXP_END"], global_clock=global_clock)
        win.flip()
        exp_end_time = float(global_clock.getTime())
        exp_end_code = int(TRIG["EXP_END"])

        done_stim = visual.TextStim(win, text="Session complete. Thank you.\n\nPress SPACE to finish.", color="white", height=32)
        waiting = True
        while waiting:
            done_stim.draw()
            win.flip()
            if kb_default.getKeys(keyList=["escape"], waitRelease=False):
                waiting = False
            if kb_task.getKeys(keyList=["space"], waitRelease=False):
                waiting = False
            core.wait(0.01)

    except KeyboardInterrupt:
        aborted = True
        print("Session stopped early by user (ESC). Saving collected data.")

    finally:
        # Best-effort EXP_END if aborted
        if 'aborted' in locals() and aborted:
            try:
                eeg.flip_pulse(TRIG["EXP_END"], global_clock=global_clock)
                win.flip()
                exp_end_time = float(global_clock.getTime())
                exp_end_code = int(TRIG["EXP_END"])
            except Exception:
                pass

        # Stamp experiment metadata into both csv row types
        if 'rsa_rows' in locals():
            for row in rsa_rows:
                row["exp_start_time"] = exp_start_time
                row["exp_end_time"] = exp_end_time
                row["trigger_exp_start"] = exp_start_code
                row["trigger_exp_end"] = exp_end_code

        if 'sd_rows' in locals():
            for row in sd_rows:
                row["exp_start_time"] = exp_start_time
                row["exp_end_time"] = exp_end_time
                row["trigger_exp_start"] = exp_start_code
                row["trigger_exp_end"] = exp_end_code

        # Save trial CSVs (two files only)
        if 'participant_id' in locals() and participant_id != "" and 'session_day' in locals() and session_day in (0, 21):
            dt_key = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"{participant_id}_day{session_day}_{dt_key}"
            rsa_path = os.path.join(DATA_DIR, f"rsa_trials_{base}.csv")
            sd_path = os.path.join(DATA_DIR, f"sd_trials_{base}.csv")

            if 'rsa_rows' in locals():
                pd.DataFrame(rsa_rows).to_csv(rsa_path, index=False)
                print(f"Saved RSA trials: {rsa_path}")
            if 'sd_rows' in locals():
                pd.DataFrame(sd_rows).to_csv(sd_path, index=False)
                print(f"Saved SD trials: {sd_path}")

        eeg.close()
        win.close()
        core.quit()
