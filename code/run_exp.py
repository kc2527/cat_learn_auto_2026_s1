# -*- coding: utf-8 -*-
"""
Run category learning experiment using PsychoPy.
"""

from datetime import timedelta
import os
import numpy as np
import pandas as pd
from psychopy import visual, core  # type: ignore
from psychopy.hardware import keyboard
from util_func_eeg import EEGPort
from util_func_pid import prompt_for_pid
from util_func_session_man import resolve_session
from util_func_stimcat import *

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

PID_DIGITS = 3
CONDITION_BY_SUBJECT = {
    "002": 90,
    "077": 90,
    "134": 90,
    "189": 90,
    "213": 90,
    "268": 90,
    "303": 90,
    "358": 90,
    "482": 90,
    "527": 180,
    "594": 180,
    "639": 180,
    "662": 180,
    "707": 180,
    "729": 180,
    "875": 180,
    "943": 180,
    "998": 180,
}
RESUME_WINDOW = timedelta(hours=12)
NEW_SESSION_COOLDOWN = timedelta(hours=8)

# ----------------------------------------------------------------------------------

if __name__ == "__main__":

    # --------------------------- Experiment parameters ---------------------------
    if EEG_ENABLED:
        n_train = 550
        n_test = 100
    else:
        n_train = 300
        n_test = 0

    n_total = n_train + n_test

    # --------------------------- Display / geometry -------------------------------
    pixels_per_inch = 227 / 2
    px_per_cm = pixels_per_inch / 2.54
    size_cm = 5
    size_px = int(size_cm * px_per_cm)

    win = visual.Window(
        size=(1920, 1080),
        fullscr=True,
        units='pix',
        color=(0.494, 0.494, 0.494),
        colorSpace='rgb',
        winType='pyglet',
        useRetina=True,
        waitBlanking=True,
    )
    win.mouseVisible = False
    frame_rate = win.getActualFrameRate()
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
                                 texRes=256,
                                 interpolate=True,
                                 size=(size_px, size_px),
                                 units='pix',
                                 sf=0.0,
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

    # --------------------------- Subject handling --------------------------------
    dir_data = "../data"
    os.makedirs(dir_data, exist_ok=True)

    subject, condition = prompt_for_pid(win, PID_DIGITS, CONDITION_BY_SUBJECT)

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

    # NOTE: Uncomment to visualise gratings in stim space
    plot_stim_space_examples(ds, win, grating, px_per_cm)

    # ---------------------------  session handling -------------------------------
    session_info = resolve_session(
        dir_data,
        subject,
        n_total,
        resume_window=RESUME_WINDOW,
        new_session_cooldown=NEW_SESSION_COOLDOWN,
    )
    session_num = session_info["session_num"]
    part_num = session_info["part_num"]
    today_key = session_info["today_key"]
    f_name = session_info["f_name"]
    full_path = session_info["full_path"]
    n_done = session_info["n_done"]

    trial = n_done - 1

    # --------------------------- EEG init ----------------------------------------
    eeg = EEGPort(
        win,
        address=EEG_PORT_ADDRESS,
        enabled=EEG_ENABLED,
        default_ms=EEG_DEFAULT_PULSE_MS,
    )

    # --------------------------- State machine setup ------------------------------
    time_state = 0.0
    state_current = "state_init"
    state_entry = True

    resp_key = ""
    resp = ""
    fb = ""
    rt = -1
    trial = n_done - 1
    phase = ""
    cat = ""
    gap_ms = 0
    sf_cycles_per_pix = np.nan
    ori_deg = np.nan
    trig_stim = np.nan
    trig_resp = np.nan
    trig_fb = np.nan

    # Record keeping
    trial_data = {
        "subject_id": [],
        "session_num": [],
        "session_part": [],
        "trial": [],
        "phase": [],
        "cat": [],
        "resp_key": [],
        "resp": [],
        "fb": [],
        "rt": [],
        "ts_iso": [],
        "eeg_enabled": [],
        "trigger_stim": [],
        "trigger_resp": [],
        "trigger_fb": [],
        "port_address": [],
        "probe_condition": [],
        "x": [],
        "y": [],
        "xt": [],
        "yt": []
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
                resp_key = ""
                resp = ""
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
                    cat = str(ds['cat'].iloc[trial]).upper()
                    if cat not in {"A", "B"}:
                        raise ValueError(
                            f"Category labels must be 'A' or 'B'. Got: {cat}"
                        )
                    phase = ds['phase'].iloc[trial]
                    trig_stim = np.nan
                    trig_resp = np.nan
                    trig_fb = np.nan

                    grating.sf = sf_cycles_per_pix
                    grating.ori = ori_deg
                    grating.pos = (center_x, center_y)

                    kb.clearEvents()
                    gap_ms = np.random.randint(200, 401)
                    state_current = "state_pre_stim_gap"
                    state_entry = True

            win.flip()

        # --------------------- STATE: PRE-STIM GAP ---------------------
        elif state_current == "state_pre_stim_gap":
            if state_entry:
                state_clock.reset()
                state_entry = False

            time_state = state_clock.getTime() * 1000.0

            fix_h.draw()
            fix_v.draw()

            if time_state >= gap_ms:
                state_current = "state_stim"
                state_entry = True

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

                state_clock.reset()
                gap_ms = np.random.randint(200, 401)
                state_current = "state_pre_feedback_gap"
                state_entry = True

            win.flip()

        # --------------------- STATE: PRE-FEEDBACK GAP ---------------------
        elif state_current == "state_pre_feedback_gap":
            if state_entry:
                state_clock.reset()
                state_entry = False

            time_state = state_clock.getTime() * 1000.0

            grating.draw()

            if time_state >= gap_ms:
                state_current = "state_feedback"
                state_entry = True

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
                ts_iso = datetime.now().isoformat()
                probe_condition = condition

                trial_data["subject_id"].append(subject)
                trial_data["session_num"].append(session_num)
                trial_data["session_part"].append(part_num)
                trial_data["trial"].append(trial)
                trial_data["phase"].append(phase)
                trial_data["cat"].append(cat)
                trial_data["resp_key"].append(resp_key)
                trial_data["resp"].append(resp)
                trial_data["fb"].append(fb)
                trial_data["rt"].append(rt)
                trial_data["ts_iso"].append(ts_iso)
                trial_data["eeg_enabled"].append(int(bool(EEG_ENABLED)))
                trial_data["trigger_stim"].append(trig_stim)
                trial_data["trigger_resp"].append(trig_resp)
                trial_data["trigger_fb"].append(trig_fb)
                trial_data["port_address"].append(EEG_PORT_ADDRESS if EEG_ENABLED else "")
                trial_data["probe_condition"].append(probe_condition)
                trial_data["x"].append(ds["x"].iloc[trial])
                trial_data["y"].append(ds["y"].iloc[trial])
                trial_data["xt"].append(ds["xt"].iloc[trial])
                trial_data["yt"].append(ds["yt"].iloc[trial])

                pd.DataFrame(trial_data).to_csv(full_path, index=False)

                state_current = "state_iti"
                state_entry = True
                rt = -1

            win.flip()

    # --------------------------- Cleanup ------------------------------------------
    eeg.close()
    win.close()
    core.quit()
