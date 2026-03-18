# -*- coding: utf-8 -*-
"""
Standalone SD staircase pilot (single track, different-only updates).
"""

from datetime import datetime
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from psychopy import core, visual  # type: ignore
from psychopy.hardware import keyboard  # type: ignore

from util_func import make_stim_cats


# -----------------------------
# Settings
# -----------------------------
DATA_DIR = "../data_staircase_pilot"
SEED = 20260319
SOURCE_MODE = "sd_staircase_pilot_diff_only"
SCHEMA_VERSION = "1.0.0"

N_CANDIDATES_PER_CATEGORY = 3000
N_ANCHORS = 80
N_DIFFICULTY_BINS = 12

MIN_TRIALS = 60
MAX_TRIALS = 300
STOP_REVERSALS = 12
THRESH_FROM_LAST_N_REV = 8

SD_FIX_MS = 500
SD_STIM1_MS = 120
SD_ISI_MS = 250
SD_STIM2_MS = 120
SD_RESP_WINDOW_MS = 2000
SD_ITI_MIN_MS = 400
SD_ITI_MAX_MS = 800

WIN_SIZE = (1920, 1080)
FULLSCR = True
WIN_COLOR = (0.494, 0.494, 0.494)
GRATING_SIZE_CM = 5.0

KEY_SAME = "q"
KEY_DIFF = "p"


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
                if value.strip():
                    return value.strip()
            elif name == "backspace":
                value = value[:-1]
            elif name == "space":
                value += " "
            elif len(name) == 1:
                value += name


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # -----------------------------
    # Build anchor pool from no-rotation stimulus space
    # -----------------------------
    ds, _, _ = make_stim_cats(N_CANDIDATES_PER_CATEGORY)
    ds = ds[["x", "y", "xt", "yt"]].copy().reset_index(drop=True)

    feats = ds[["xt", "yt"]].to_numpy(dtype=float)
    mu = feats.mean(axis=0)
    sdv = feats.std(axis=0)
    sdv[sdv == 0] = 1.0
    z = (feats - mu) / sdv

    start_idx = int(rng.integers(0, len(ds)))
    picked = [start_idx]
    dist2 = np.sum((z - z[start_idx]) ** 2, axis=1)
    for _ in range(1, N_ANCHORS):
        next_idx = int(np.argmax(dist2))
        picked.append(next_idx)
        d2_new = np.sum((z - z[next_idx]) ** 2, axis=1)
        dist2 = np.minimum(dist2, d2_new)

    anchors = ds.iloc[picked].copy().reset_index(drop=True)
    anchors.insert(0, "stim_id", [f"a_{i:03d}" for i in range(len(anchors))])

    # All different-pair distances and difficulty bins
    az = anchors[["xt", "yt"]].to_numpy(dtype=float)
    mu_a = az.mean(axis=0)
    sd_a = az.std(axis=0)
    sd_a[sd_a == 0] = 1.0
    az = (az - mu_a) / sd_a

    pair_rows = []
    n = len(anchors)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(az[i] - az[j]))
            pair_rows.append((i, j, d))
    diff_df = pd.DataFrame(pair_rows, columns=["i", "j", "dist"])

    # qcut produces easy->hard by distance ascending; invert to make higher index harder.
    diff_df["bin_easy_to_hard"] = pd.qcut(diff_df["dist"], q=N_DIFFICULTY_BINS, labels=False, duplicates="drop")
    max_bin = int(diff_df["bin_easy_to_hard"].max())
    diff_df["difficulty_bin"] = max_bin - diff_df["bin_easy_to_hard"]
    bin_ids = sorted(int(x) for x in diff_df["difficulty_bin"].unique())
    min_bin = min(bin_ids)
    max_bin = max(bin_ids)

    bin_to_pairs = {}
    for b in bin_ids:
        bin_to_pairs[b] = diff_df.loc[diff_df["difficulty_bin"] == b].copy().reset_index(drop=True)

    # -----------------------------
    # PsychoPy setup
    # -----------------------------
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
        text=f"Same ({KEY_SAME.upper()}) / Different ({KEY_DIFF.upper()})",
        color="white",
        height=32,
    )

    kb_task = keyboard.Keyboard()
    kb_default = keyboard.Keyboard()
    global_clock = core.Clock()

    participant_id = ""
    session_label = ""
    trial_rows = []
    reversal_levels = []
    reversal_trials = []
    stop_reason = "aborted"
    run_start_iso = datetime.now().isoformat()

    try:
        while participant_id == "":
            participant_id = text_entry_screen(win, kb_task, kb_default, "Enter participant ID", "Press ENTER when done")
        while session_label == "":
            session_label = text_entry_screen(win, kb_task, kb_default, "Enter session label", "Example: pilot_1")

        inst = visual.TextStim(
            win,
            text=(
                "Same/Different Staircase Pilot (Diff-only updates)\n\n"
                f"Press {KEY_SAME.upper()} for SAME, {KEY_DIFF.upper()} for DIFFERENT.\n"
                "Difficulty adapts only from DIFFERENT trials.\n\n"
                "Press SPACE to begin."
            ),
            color="white",
            height=30,
            wrapWidth=1450,
        )
        waiting = True
        while waiting:
            inst.draw()
            win.flip()
            if kb_default.getKeys(keyList=["escape"], waitRelease=False):
                raise KeyboardInterrupt("Escape pressed")
            if kb_task.getKeys(keyList=["space"], waitRelease=False):
                waiting = False
            core.wait(0.01)

        # Staircase state (different-only updates)
        current_level = int(round((min_bin + max_bin) * 0.35))
        current_level = max(min_bin, min(max_bin, current_level))
        consecutive_correct_diff = 0
        last_direction = 0
        reversal_count = 0
        trial_index = 0
        run_done = False

        while not run_done:
            trial_index += 1
            level_pre = int(current_level)

            # 50/50 same/different
            is_same = bool(rng.random() < 0.5)
            adaptive_trial_flag = 0

            if is_same:
                i = int(rng.integers(0, len(anchors)))
                stim1 = anchors.iloc[i]
                stim2 = anchors.iloc[i]
                trial_type = "same"
                distance_value = 0.0
                distance_bin = np.nan
            else:
                adaptive_trial_flag = 1
                pool = bin_to_pairs[level_pre]
                p = pool.iloc[int(rng.integers(0, len(pool)))]
                i = int(p["i"])
                j = int(p["j"])
                stim1 = anchors.iloc[i]
                stim2 = anchors.iloc[j]
                trial_type = "different"
                distance_value = float(p["dist"])
                distance_bin = int(level_pre)

            # Fixation
            fix_h.draw()
            fix_v.draw()
            win.flip()
            t0 = global_clock.getTime()
            while (global_clock.getTime() - t0) < (SD_FIX_MS / 1000.0):
                if kb_default.getKeys(keyList=["escape"], waitRelease=False):
                    raise KeyboardInterrupt("Escape pressed")
                core.wait(0.005)

            # Stim1
            grating.sf = float(stim1["xt"]) / px_per_cm
            grating.ori = float(stim1["yt"]) * 180.0 / np.pi
            grating.draw()
            win.flip()
            t0 = global_clock.getTime()
            while (global_clock.getTime() - t0) < (SD_STIM1_MS / 1000.0):
                if kb_default.getKeys(keyList=["escape"], waitRelease=False):
                    raise KeyboardInterrupt("Escape pressed")
                core.wait(0.005)

            # ISI
            win.flip()
            t0 = global_clock.getTime()
            while (global_clock.getTime() - t0) < (SD_ISI_MS / 1000.0):
                if kb_default.getKeys(keyList=["escape"], waitRelease=False):
                    raise KeyboardInterrupt("Escape pressed")
                core.wait(0.005)

            # Stim2
            grating.sf = float(stim2["xt"]) / px_per_cm
            grating.ori = float(stim2["yt"]) * 180.0 / np.pi
            grating.draw()
            win.flip()
            t0 = global_clock.getTime()
            while (global_clock.getTime() - t0) < (SD_STIM2_MS / 1000.0):
                if kb_default.getKeys(keyList=["escape"], waitRelease=False):
                    raise KeyboardInterrupt("Escape pressed")
                core.wait(0.005)

            # Response
            kb_task.clearEvents()
            prompt_text.draw()
            win.flip()
            resp_clock = core.Clock()
            response_key = ""
            response_label = ""
            rt_ms = np.nan
            responded = False
            while resp_clock.getTime() * 1000.0 < SD_RESP_WINDOW_MS:
                if kb_default.getKeys(keyList=["escape"], waitRelease=False):
                    raise KeyboardInterrupt("Escape pressed")
                keys = kb_task.getKeys(keyList=[KEY_SAME, KEY_DIFF], waitRelease=False)
                if keys:
                    k = keys[-1]
                    response_key = k.name
                    rt_ms = float(k.rt * 1000.0)
                    response_label = "same" if k.name == KEY_SAME else "different"
                    responded = True
                    break
                core.wait(0.005)
            if not responded:
                response_label = "no_response"

            correct_answer = "same" if trial_type == "same" else "different"
            accuracy = int(response_label == correct_answer)

            # DIFFERENT-only staircase update
            direction = 0
            if adaptive_trial_flag == 1:
                if accuracy == 1:
                    consecutive_correct_diff += 1
                    if consecutive_correct_diff >= 2:
                        current_level = min(max_bin, current_level + 1)
                        direction = +1
                        consecutive_correct_diff = 0
                else:
                    consecutive_correct_diff = 0
                    current_level = max(min_bin, current_level - 1)
                    direction = -1

            level_post = int(current_level)

            reversal_flag = 0
            if adaptive_trial_flag == 1 and direction != 0 and last_direction != 0 and direction != last_direction:
                reversal_flag = 1
                reversal_count += 1
                reversal_levels.append(level_pre)
                reversal_trials.append(trial_index)
            if adaptive_trial_flag == 1 and direction != 0:
                last_direction = direction

            # ITI
            win.flip()
            iti_ms = int(rng.integers(SD_ITI_MIN_MS, SD_ITI_MAX_MS + 1))
            t0 = global_clock.getTime()
            while (global_clock.getTime() - t0) < (iti_ms / 1000.0):
                if kb_default.getKeys(keyList=["escape"], waitRelease=False):
                    raise KeyboardInterrupt("Escape pressed")
                core.wait(0.005)

            trial_rows.append({
                "participant_id": participant_id,
                "session_label": session_label,
                "session_datetime_iso": run_start_iso,
                "source_mode": SOURCE_MODE,
                "schema_version": SCHEMA_VERSION,
                "trial_index": int(trial_index),
                "trial_type": trial_type,
                "stim1_id": str(stim1["stim_id"]),
                "stim2_id": str(stim2["stim_id"]),
                "stim1_xt": float(stim1["xt"]),
                "stim1_yt": float(stim1["yt"]),
                "stim2_xt": float(stim2["xt"]),
                "stim2_yt": float(stim2["yt"]),
                "distance_value": float(distance_value),
                "distance_bin": distance_bin,
                "staircase_level_pre": level_pre,
                "staircase_level_post": level_post,
                "adaptive_trial_flag": int(adaptive_trial_flag),
                "response_key": response_key,
                "response_label": response_label,
                "correct_answer": correct_answer,
                "accuracy": int(accuracy),
                "rt_ms": rt_ms,
                "consecutive_correct_diff": int(consecutive_correct_diff),
                "reversal_flag": int(reversal_flag),
                "reversal_count": int(reversal_count),
                "stop_reason": "",
            })

            # Stop logic
            if trial_index >= MAX_TRIALS:
                stop_reason = "max_trials"
                run_done = True
            elif trial_index >= MIN_TRIALS and reversal_count >= STOP_REVERSALS:
                stop_reason = "reversal_target"
                run_done = True

        # Fill stop reason
        for row in trial_rows:
            row["stop_reason"] = stop_reason

    except KeyboardInterrupt:
        stop_reason = "aborted"
        for row in trial_rows:
            row["stop_reason"] = stop_reason
        print("Pilot stopped by user (ESC). Saving partial data.")
    finally:
        # Summary + save
        run_end = datetime.now()
        run_end_iso = run_end.isoformat()
        duration_sec = (run_end - datetime.fromisoformat(run_start_iso)).total_seconds()

        rev_arr = np.array(reversal_levels[-THRESH_FROM_LAST_N_REV:] if len(reversal_levels) >= THRESH_FROM_LAST_N_REV else reversal_levels, dtype=float)
        if rev_arr.size == 0:
            threshold_last8_mean = np.nan
            threshold_last8_sd = np.nan
        else:
            threshold_last8_mean = float(np.nanmean(rev_arr))
            threshold_last8_sd = float(np.nanstd(rev_arr))

        df = pd.DataFrame(trial_rows)
        n_trials = int(len(df))
        n_trials_same = int((df["trial_type"] == "same").sum()) if n_trials > 0 else 0
        n_trials_diff = int((df["trial_type"] == "different").sum()) if n_trials > 0 else 0
        overall_acc = float(df["accuracy"].mean()) if n_trials > 0 else np.nan
        same_acc = float(df.loc[df["trial_type"] == "same", "accuracy"].mean()) if n_trials_same > 0 else np.nan
        diff_acc = float(df.loc[df["trial_type"] == "different", "accuracy"].mean()) if n_trials_diff > 0 else np.nan
        same_fa = float((df.loc[df["trial_type"] == "same", "response_label"] == "different").mean()) if n_trials_same > 0 else np.nan
        diff_miss = float((df.loc[df["trial_type"] == "different", "response_label"] != "different").mean()) if n_trials_diff > 0 else np.nan

        summary = pd.DataFrame([{
            "participant_id": participant_id,
            "session_label": session_label,
            "n_trials": n_trials,
            "n_trials_different": n_trials_diff,
            "n_trials_same": n_trials_same,
            "n_reversals": int(len(reversal_levels)),
            "threshold_last8_mean": threshold_last8_mean,
            "threshold_last8_sd": threshold_last8_sd,
            "overall_accuracy": overall_acc,
            "same_accuracy": same_acc,
            "different_accuracy": diff_acc,
            "same_false_alarm_rate": same_fa,
            "different_miss_rate": diff_miss,
            "stop_reason": stop_reason,
            "started_at_iso": run_start_iso,
            "ended_at_iso": run_end_iso,
            "duration_sec": float(duration_sec),
        }])

        dt_key = run_end.strftime("%Y%m%d_%H%M%S")
        base = f"{participant_id if participant_id else 'unknown'}_{session_label if session_label else 'nosession'}_{dt_key}"
        trials_path = os.path.join(DATA_DIR, f"sd_staircase_pilot_diff_only_trials_{base}.csv")
        summary_path = os.path.join(DATA_DIR, f"sd_staircase_pilot_diff_only_summary_{base}.csv")
        plot_path = os.path.join(DATA_DIR, f"sd_staircase_pilot_diff_only_trace_{base}.png")

        df.to_csv(trials_path, index=False)
        summary.to_csv(summary_path, index=False)

        plt.figure(figsize=(10, 4))
        if n_trials > 0:
            plt.plot(df["trial_index"], df["staircase_level_pre"], lw=1.5, label="Level")
        if len(reversal_trials) > 0:
            plt.scatter(np.array(reversal_trials), np.array(reversal_levels), s=35, label="Reversals")
        plt.xlabel("Trial")
        plt.ylabel("Staircase Level (higher = harder)")
        plt.title("SD Staircase (Diff-only updates)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=140)
        plt.close()

        print(f"Saved trials: {trials_path}")
        print(f"Saved summary: {summary_path}")
        print(f"Saved plot: {plot_path}")

        try:
            win.close()
        except Exception:
            pass
        core.quit()
