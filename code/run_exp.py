# -*- coding: utf-8 -*-
"""
Run category learning experiment (lab mode, PsychoPy).
"""

from datetime import datetime, timedelta
import re
import os
import sys
import uuid
import re
import json
import numpy as np
import pandas as pd
from psychopy import visual, core, event, logging  # type: ignore
from psychopy.hardware import keyboard  # type: ignore
from util_func import *

SCHEMA_VERSION = "1.0.0"
SOURCE_MODE = "lab"


# --------------------------- EEG (Parallel Port) helper ---------------------------
# Flip-locked rising edges; non-blocking clear to zero a few ms later.
EEG_ENABLED = False
EEG_PORT_ADDRESS = '0x3FD8'
EEG_DEFAULT_PULSE_MS = 10

TRIG = {

    # -------------------- Experiment structure --------------------
    "EXP_START": 10,
    "ITI_ONSET": 11,
    "EXP_END": 15,

    # -------------------- Stimulus onset --------------------
    # Training trials
    "STIM_ONSET_A_TRAIN": 20,
    "STIM_ONSET_B_TRAIN": 21,

    # Probe trials
    "STIM_ONSET_A_PROBE": 22,
    "STIM_ONSET_B_PROBE": 23,

    # -------------------- Responses --------------------
    # Training trials
    "RESP_A_TRAIN": 30,
    "RESP_B_TRAIN": 31,

    # Probe trials
    "RESP_A_PROBE": 32,
    "RESP_B_PROBE": 33,

    # -------------------- Feedback --------------------
    # Training trials
    "FB_COR_TRAIN": 40,
    "FB_INC_TRAIN": 41,

    # Probe trials
    "FB_COR_PROBE": 42,
    "FB_INC_PROBE": 43,
}


def normalize_category_label(value):
    """Normalize category codes to the task labels used by response keys."""
    if isinstance(value, str):
        v = value.strip().upper()
        if v in ("A", "1"):
            return "A"
        if v in ("B", "2"):
            return "B"

    try:
        v = int(value)
    except (TypeError, ValueError):
        v = None

    if v == 1:
        return "A"
    if v == 2:
        return "B"

    raise ValueError(f"Unexpected category label: {value!r}")


class EEGPort:

    def __init__(self,
                 win,
                 address=EEG_PORT_ADDRESS,
                 enabled=EEG_ENABLED,
                 default_ms=EEG_DEFAULT_PULSE_MS):
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
        except Exception as e:
            print(
                f"[EEG] Parallel port unavailable ({e}). Running without triggers."
            )
            self.enabled = False
            self._port = None

    def flip_pulse(self, code, width_ms=None, global_clock=None):
        """Schedule a flip-locked pulse: set code on next win.flip, clear after width_ms."""
        if not (self.enabled and self._port):
            return
        width_ms = self.default_ms if width_ms is None else width_ms
        # rising edge exactly on next flip:
        self.win.callOnFlip(self._port.setData, int(code) & 0xFF)
        # schedule a timed clear to 0 after the flip:
        if global_clock is not None:
            # record when to clear (relative to global clock)
            self._clear_at = global_clock.getTime() + (width_ms / 1000.0)

    def pulse_now(self, code, width_ms=None, global_clock=None):
        """Immediate pulse (not flip-locked) -- useful for response events."""
        if not (self.enabled and self._port):
            return
        width_ms = self.default_ms if width_ms is None else width_ms
        self._port.setData(int(code) & 0xFF)
        if global_clock is not None:
            self._clear_at = global_clock.getTime() + (width_ms / 1000.0)

    def update(self, global_clock=None):
        """Call every frame: clears the port to 0 if a pulse has expired."""
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


# ----------------------------------------------------------------------------------

if __name__ == "__main__":

    # --------------------------- Experiment parameters ---------------------------
    n_train = 10
    n_test = 10
    n_total = n_train + n_test

    # --------------------------- Subject handling --------------------------------
    dir_data = "../data"
    os.makedirs(dir_data, exist_ok=True)

    info_path = os.path.join(dir_data, "participant_info.json")

    if os.path.exists(info_path):
        try:
            with open(info_path, "r") as f:
                info = json.load(f)
            subject = info["subject"]
            condition = int(info["condition"])
        except Exception:
            print("Could not read participant_info.json. Aborting.")
            sys.exit()
    else:
        subject = uuid.uuid4().hex[:12]
        condition = int(np.random.choice([90, 180]))
        info = {"subject": subject, "condition": condition}
        try:
            with open(info_path, "x") as f:
                json.dump(info, f, indent=4)
        except FileExistsError:
            print("participant_info.json already exists. Aborting.")
            sys.exit()

    # ---------------------------  session handling -------------------------------
    WINDOW = timedelta(hours=12)
    now = datetime.now()

    fn_re = re.compile(
        rf"^sub_{re.escape(str(subject))}_date_(\d{{4}}_\d{{2}}_\d{{2}})_data\.csv$"
    )

    def load_n_done(path):
        return pd.read_csv(path).shape[0]

    # ------------------------------------------------------------------
    # Gather all files for this subject
    # ------------------------------------------------------------------
    candidates = []
    for fn in os.listdir(dir_data):
        if fn_re.match(fn):
            full = os.path.join(dir_data, fn)
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(full))
                candidates.append((mtime, fn, full))
            except OSError:
                pass

    candidates.sort(reverse=True, key=lambda x: x[0])
    candidates_asc = sorted(candidates, key=lambda x: x[0])
    file_to_session_num = {fn: i + 1 for i, (_, fn, _) in enumerate(candidates_asc)}

    resume = None
    most_recent = None
    session_num = len(candidates) + 1
    today_key = now.strftime("%Y_%m_%d")

    # ------------------------------------------------------------------
    # Identify most recent file and most recent incomplete file
    # ------------------------------------------------------------------
    for i, (mtime, fn, full) in enumerate(candidates):
        try:
            n = load_n_done(full)
        except Exception:
            continue

        if i == 0:
            most_recent = (mtime, fn, full, n)

        if n < n_total and resume is None:
            resume = (mtime, fn, full, n)

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------
    if resume is not None:
        mtime, f_name, full_path, n_done = resume
        age = now - mtime
        m = fn_re.match(f_name)
        if m:
            today_key = m.group(1)

        if age <= WINDOW:
            session_num = file_to_session_num.get(f_name, len(candidates) + 1)
            remaining = n_total - n_done
            print(
                f"Resuming your last incomplete session "
                f"(last saved {mtime:%Y-%m-%d %H:%M})."
            )
            print(
                f"You have {remaining} trials remaining in this session. "
                "Please try to finish today’s trials so you can stay on track."
            )
        else:
            session_num = len(candidates) + 1
            today_key = now.strftime("%Y_%m_%d")
            f_name = f"sub_{subject}_date_{today_key}_data.csv"
            full_path = os.path.join(dir_data, f_name)
            n_done = 0
            print(
                "Your last incomplete session was more than 12 hours ago. "
                "Starting a new session."
            )

    else:
        if most_recent is not None:
            mtime, fn, path, last_n = most_recent
            age = now - mtime

            if last_n >= n_total and age < WINDOW:
                next_ok = mtime + WINDOW
                print(
                    "It has been fewer than 12 hours since your last completed session.\n"
                    f"Please wait until {next_ok:%Y-%m-%d %H:%M} before trying again."
                )
                sys.exit()

        session_num = len(candidates) + 1
        today_key = now.strftime("%Y_%m_%d")
        f_name = f"sub_{subject}_date_{today_key}_data.csv"
        full_path = os.path.join(dir_data, f_name)
        n_done = 0

    # ------------------------------------------------------------------
    # Final guard + status print
    # ------------------------------------------------------------------
    if n_done >= n_total:
        print(f"Session is already complete ({n_done} trials). Aborting.")
        sys.exit()

    print(
        f"Subject: {subject} | Condition: {condition} | "
        f"Session: {session_num} | Date: {today_key} | Resuming at trial: {n_done}"
    )

    trial = n_done - 1

    # --------------------------- Stimuli and Categories  ---------------------------
    n_stimuli_per_category = n_total // 2
    ds, ds_90, ds_180 = make_stim_cats(n_stimuli_per_category)

    ds_train = ds.copy()
    ds_train = ds_train.sample(frac=1).reset_index(drop=True)
    ds_train = ds_train.iloc[:n_train, :]
    ds_train["phase"] = "train"

    if condition == 90:
        ds_test = ds_90.copy()
    elif condition == 180:
        ds_test = ds_180.copy()

    ds_test = ds_test.sample(frac=1).reset_index(drop=True)
    ds_test = ds_test.iloc[:n_test, :]
    ds_test["phase"] = "test"

    ds = pd.concat([ds_train, ds_test]).reset_index(drop=True)

    # NOTE: Uncomment to visualize stimulus space scatter
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # fig, ax = plt.subplots(1, 3, squeeze=False, figsize=(6, 6))
    # sns.scatterplot(data=ds, x='x', y='y', hue='cat', ax=ax[0, 0])
    # sns.scatterplot(data=ds, x='xt', y='yt', hue='cat', ax=ax[0, 1])
    # ds['yt_deg'] = ds['yt'] * 180.0 / np.pi
    # sns.scatterplot(data=ds, x='xt', y='yt_deg', hue='cat', ax=ax[0, 2])
    # plt.show()

    # --------------------------- Display / geometry -------------------------------
    pixels_per_inch = 227 / 2
    px_per_cm = pixels_per_inch / 2.54
    size_cm = 5
    size_px = int(size_cm * px_per_cm)

    win = visual.Window(size=(1920, 1080),
                        fullscr=True,
                        units='pix',
                        color=(0.494, 0.494, 0.494),
                        colorSpace='rgb',
                        winType='pyglet',
                        useRetina=True,
                        waitBlanking=True)

    win.mouseVisible = False
    frame_rate = win.getActualFrameRate()
    print(f"[Info] Frame rate: {frame_rate}")

    center_x, center_y = 0, 0

    # --------------------------- Stim objects ------------------------------------
    fix_h = visual.Line(win,
                        start=(0, -10),
                        end=(0, 10),
                        lineColor='white',
                        lineWidth=8)
    fix_v = visual.Line(win,
                        start=(-10, 0),
                        end=(10, 0),
                        lineColor='white',
                        lineWidth=8)

    init_text = visual.TextStim(win,
                                text="Please press the space bar to begin",
                                color='white',
                                height=32)

    finished_text = visual.TextStim(
        win,
        text="You finished! Thank you for participating!",
        color='white',
        height=32)

    grating = visual.GratingStim(win,
                                 tex='sin',
                                 mask='circle',
                                 interpolate=True,
                                 size=(size_px, size_px),
                                 units='pix',
                                 sf=0.02,
                                 ori=0.0)

    fb_ring = visual.Circle(win,
                            radius=(size_px // 2 + 10),
                            edges=128,
                            fillColor=None,
                            lineColor='white',
                            lineWidth=10,
                            units='pix',
                            pos=(center_x, center_y))

    kb = keyboard.Keyboard()
    default_kb = keyboard.Keyboard()

    global_clock = core.Clock()
    state_clock = core.Clock()
    stim_clock = core.Clock()

    # --------------------------- EEG init ----------------------------------------
    eeg = EEGPort(win)

    # --------------------------- State machine setup ------------------------------
    time_state = 0.0
    state_current = "state_init"
    state_entry = True

    resp = -1
    resp_key = ""
    action_label = ""
    fb = ""
    rt = -1
    trial = n_done - 1
    phase = ""
    cat = ""
    sf_cycles_per_pix = np.nan
    ori_deg = np.nan
    trig_stim = np.nan
    trig_resp = np.nan
    trig_fb = np.nan

    # Record keeping
    trial_data = {
        "schema_version": [],
        "source_mode": [],
        "subject_id": [],
        "session_num": [],
        "trial_index": [],
        "phase": [],
        "category": [],
        "stim_sf": [],
        "stim_ori_deg": [],
        "resp_key": [],
        "action_label": [],
        "correct": [],
        "rt_ms": [],
        "ts_iso": [],
        "data_file": [],
        "eeg_enabled": [],
        "trigger_stim": [],
        "trigger_resp": [],
        "trigger_feedback": [],
        "port_address": [],
        "probe_condition": [],
        # Backward-compatible legacy columns:
        "subject": [],
        "day": [],
        "trial": [],
        "cat": [],
        "x": [],
        "y": [],
        "xt": [],
        "yt": [],
        "resp": [],
        "rt": [],
        "fb": []
    }

    # --------------------------- Main loop ---------------------------------------
    running = True
    while running:

        if default_kb.getKeys(keyList=['escape'], waitRelease=False):
            running = False
            break

        eeg.update(global_clock)

        # --------------------- STATE: INIT ---------------------
        if state_current == "state_init":
            if state_entry:
                state_clock.reset()
                win.color = (0.494, 0.494, 0.494)
                state_entry = False

            time_state = state_clock.getTime() * 1000.0
            init_text.draw()

            keys = kb.getKeys(keyList=['space'], waitRelease=False, clear=True)
            if keys:
                eeg.flip_pulse(TRIG["EXP_START"], global_clock=global_clock)
                state_current = "state_iti"
                state_entry = True

            win.flip()

        # --------------------- STATE: FINISHED ---------------------
        elif state_current == "state_finished":
            if state_entry:
                eeg.flip_pulse(TRIG["EXP_END"], global_clock=global_clock)
                state_clock.reset()
                state_entry = False

            time_state = state_clock.getTime() * 1000.0
            finished_text.draw()
            win.flip()

        # --------------------- STATE: ITI ---------------------
        elif state_current == "state_iti":
            if state_entry:
                state_clock.reset()
                eeg.flip_pulse(TRIG["ITI_ONSET"], global_clock=global_clock)
                state_entry = False

            time_state = state_clock.getTime() * 1000.0

            fix_h.draw()
            fix_v.draw()

            if time_state > 1000:
                resp = -1
                resp_key = ""
                action_label = ""
                fb = ""
                rt = -1
                state_clock.reset()
                trial += 1
                if trial >= n_total:
                    state_current = "state_finished"
                    state_entry = True
                else:
                    sf_cycles_per_cm = ds['xt'].iloc[trial]
                    sf_cycles_per_pix = sf_cycles_per_cm / px_per_cm
                    ori_deg = ds['yt'].iloc[trial] * 180.0 / np.pi
                    cat_raw = ds['cat'].iloc[trial]
                    cat = normalize_category_label(cat_raw)
                    phase = ds['phase'].iloc[trial]
                    trig_stim = np.nan
                    trig_resp = np.nan
                    trig_fb = np.nan

                    grating.sf = sf_cycles_per_pix
                    grating.ori = ori_deg
                    grating.pos = (center_x, center_y)

                    kb.clearEvents()
                    state_current = "state_stim"
                    state_entry = True

                    jitter = np.random.randint(200, 401)
                    core.wait(jitter / 1000.0)

            win.flip()

        # --------------------- STATE: STIM ---------------------
        elif state_current == "state_stim":
            if state_entry:
                if phase == 'train':
                    if cat == "A":
                        trig = TRIG["STIM_ONSET_A_TRAIN"]
                    else:
                        trig = TRIG["STIM_ONSET_B_TRAIN"]
                elif phase == 'test':
                    if cat == "A":
                        trig = TRIG["STIM_ONSET_A_PROBE"]
                    else:
                        trig = TRIG["STIM_ONSET_B_PROBE"]
                else:
                    trig = np.nan

                if not np.isnan(trig):
                    eeg.flip_pulse(trig, global_clock=global_clock)
                    trig_stim = int(trig)

                state_clock.reset()
                stim_clock.reset()

                win.callOnFlip(kb.clock.reset)
                state_entry = False

            time_state = state_clock.getTime() * 1000.0

            grating.draw()

            keys = kb.getKeys(keyList=['d', 'k'], waitRelease=False)
            if keys:
                k = keys[-1]
                resp_key = k.name
                rt = k.rt * 1000.0
                if phase == 'train':
                    if k.name == 'd':
                        resp_label = "A"
                        trig = TRIG["RESP_A_TRAIN"]
                    else:
                        resp_label = "B"
                        trig = TRIG["RESP_B_TRAIN"]
                elif phase == 'test':
                    if k.name == 'd':
                        resp_label = "A"
                        trig = TRIG["RESP_A_PROBE"]
                    else:
                        resp_label = "B"
                        trig = TRIG["RESP_B_PROBE"]
                else:
                    resp_label = "none"
                    trig = np.nan

                if not np.isnan(trig):
                    eeg.pulse_now(trig, global_clock=global_clock)
                    trig_resp = int(trig)

                if cat == resp_label:
                    fb = "Correct"
                else:
                    fb = "Incorrect"

                resp = resp_label
                action_label = resp_label

                state_clock.reset()
                state_current = "state_feedback"
                state_entry = True

                jitter = np.random.randint(200, 401)
                core.wait(jitter / 1000.0)

            win.flip()

        # --------------------- STATE: FEEDBACK ---------------------
        elif state_current == "state_feedback":
            if state_entry:
                if phase == 'train':
                    if fb == "Correct":
                        fb_ring.lineColor = 'green'
                        trig = TRIG["FB_COR_TRAIN"]
                    else:
                        fb_ring.lineColor = 'red'
                        trig = TRIG["FB_INC_TRAIN"]
                elif phase == 'test':
                    if fb == "Correct":
                        fb_ring.lineColor = 'green'
                        trig = TRIG["FB_COR_PROBE"]
                    else:
                        fb_ring.lineColor = 'red'
                        trig = TRIG["FB_INC_PROBE"]
                else:
                    trig = np.nan

                if not np.isnan(trig):
                    eeg.flip_pulse(trig, global_clock=global_clock)
                    trig_fb = int(trig)

                state_clock.reset()
                state_entry = False

            time_state = state_clock.getTime() * 1000.0

            grating.draw()
            fb_ring.draw()

            if time_state > 1000:
                correct = int(fb == "Correct")
                ts_iso = datetime.now().isoformat()
                probe_condition = condition if phase == "test" else np.nan

                trial_data["schema_version"].append(SCHEMA_VERSION)
                trial_data["source_mode"].append(SOURCE_MODE)
                trial_data["subject_id"].append(subject)
                trial_data["session_num"].append(session_num)
                trial_data["trial_index"].append(trial)
                trial_data["phase"].append(phase)
                trial_data["category"].append(cat)
                trial_data["stim_sf"].append(sf_cycles_per_pix)
                trial_data["stim_ori_deg"].append(ori_deg)
                trial_data["resp_key"].append(resp_key)
                trial_data["action_label"].append(action_label)
                trial_data["correct"].append(correct)
                trial_data["rt_ms"].append(rt)
                trial_data["ts_iso"].append(ts_iso)
                trial_data["data_file"].append(f_name)
                trial_data["eeg_enabled"].append(int(bool(EEG_ENABLED)))
                trial_data["trigger_stim"].append(trig_stim)
                trial_data["trigger_resp"].append(trig_resp)
                trial_data["trigger_feedback"].append(trig_fb)
                trial_data["port_address"].append(EEG_PORT_ADDRESS if EEG_ENABLED else "")
                trial_data["probe_condition"].append(probe_condition)

                trial_data["subject"].append(subject)
                trial_data["day"].append(today_key)
                trial_data["trial"].append(trial)
                trial_data["cat"].append(ds["cat"].iloc[trial])
                trial_data["x"].append(ds["x"].iloc[trial])
                trial_data["y"].append(ds["y"].iloc[trial])
                trial_data["xt"].append(ds["xt"].iloc[trial])
                trial_data["yt"].append(ds["yt"].iloc[trial])
                trial_data["resp"].append(resp)
                trial_data["rt"].append(rt)
                trial_data["fb"].append(fb)

                pd.DataFrame(trial_data).to_csv(full_path, index=False)

                state_current = "state_iti"
                state_entry = True
                resp = -1
                rt = -1

                jitter = np.random.randint(200, 401)
                core.wait(jitter / 1000.0)

            win.flip()

    # --------------------------- Cleanup ------------------------------------------
    eeg.close()
    win.close()
    core.quit()
