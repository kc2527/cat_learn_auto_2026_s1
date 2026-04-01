# -*- coding: utf-8 -*-
"""
Run representational similarity analysis task using PsychoPy.
"""

from datetime import datetime, timedelta
import os
import sys
import numpy as np
import pandas as pd
from psychopy import core, visual
from psychopy.hardware import keyboard
from util_func_eeg import EEGPort
from util_func_pid import prompt_for_pid_in_set
from util_func_session_man import resolve_session
from util_func_stimcat import make_rsa_pool_grid
from util_func_stimcat import make_rsa_schedule_table
from util_func_stimcat import plot_stim_space_examples
from util_func_stimcat import stim_xy_to_sf_ori_deg
from util_func_stimcat import transform_stim

EEG_ENABLED = False
EEG_PORT_ADDRESS = "0x3FB8"
EEG_DEFAULT_PULSE_MS = 50

TRIG = {
    "EXP_START": 10,
    "EXP_END": 15,
    "RSA_BLOCK_START": 21,
    "RSA_BLOCK_END": 22,
    "RSA_READY_ONSET": 23,
    "RSA_STIM_ONSET": 24,
}

PID_DIGITS = 3
MODE = "rsa"
ALLOWED_SUBJECT_IDS = {
    "002",
    "077",
    "134",
    "189",
    "213",
    "268",
    "303",
    "358",
    "482",
    "527",
    "594",
    "639",
    "662",
    "707",
    "729",
    "875",
    "943",
    "998",
}

PIXELS_PER_INCH = 227 / 2
PX_PER_CM = PIXELS_PER_INCH / 2.54
SIZE_CM = 5
SIZE_PX = int(SIZE_CM * PX_PER_CM)
RSA_BLOCKS = 8
RSA_POOL_GRID_N = 7
RSA_POOL_SIZE = RSA_POOL_GRID_N * RSA_POOL_GRID_N
RSA_REPEATS_PER_BLOCK = 2
RSA_TOTAL_TRIALS = RSA_POOL_SIZE * RSA_REPEATS_PER_BLOCK * RSA_BLOCKS
RSA_SOA_SEC = 0.5
RESUME_WINDOW = timedelta(hours=12)
NEW_SESSION_COOLDOWN = timedelta(hours=8)

if __name__ == "__main__":

    # --------------------------- Display / geometry -------------------------------

    win = visual.Window(
        size=(1920, 1080),
        fullscr=True,
        units="pix",
        color=(0.494, 0.494, 0.494),
        colorSpace="rgb",
        winType="pyglet",
        useRetina=True,
        waitBlanking=True,
    )
    win.mouseVisible = False

    # --------------------------- Stim objects ------------------------------------

    msg_text = visual.TextStim(win,
                               text="",
                               color="white",
                               height=32,
                               wrapWidth=1600)

    fix_h = visual.ShapeStim(win,
                             vertices=[(-20, 0), (20, 0)],
                             lineWidth=6,
                             lineColor="white",
                             closeShape=False)

    fix_v = visual.ShapeStim(win,
                             vertices=[(0, -20), (0, 20)],
                             lineWidth=6,
                             lineColor="white",
                             closeShape=False)

    grating = visual.GratingStim(
        win,
        tex="sin",
        mask="circle",
        texRes=256,
        interpolate=True,
        size=(SIZE_PX, SIZE_PX),
        units="pix",
        sf=0.02,
        ori=0.0,
        phase=0.0,
        pos=(0, 0),
    )

    intro_text = (
        "You will now see flashing stimuli.\n"
        "Please stay relaxed, keep your eyes on the center of the screen, and minimize movement.\n"
        "This may feel repetitive or boring, but please keep looking at the centre and try to stay awake!\n\n"
        "Press SPACE to start.")

    ready_text = (
        "Another block is about to start.\n"
        "Feel free to wriggle, blink, and get comfortable before you begin.\n\n"
        "Press SPACE when you are ready.")

    # --------------------------- response and clocks -----------------------------
    kb = keyboard.Keyboard()
    default_kb = keyboard.Keyboard()
    global_clock = core.Clock()
    state_clock = core.Clock()

    # --------------------------- Subject handling --------------------------------
    dir_data = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(dir_data, exist_ok=True)

    participant = prompt_for_pid_in_set(win, PID_DIGITS, ALLOWED_SUBJECT_IDS)

    # ---------------------------  session handling -------------------------------
    session_info = resolve_session(
        dir_data,
        participant,
        RSA_TOTAL_TRIALS,
        resume_window=RESUME_WINDOW,
        new_session_cooldown=NEW_SESSION_COOLDOWN,
        task_tag=MODE,
    )
    session_num = session_info["session_num"]
    session_part = session_info["part_num"]
    full_path = session_info["full_path"]
    n_done = session_info["n_done"]

    # --------------------------- Stimuli and Categories  ---------------------------
    pool = make_rsa_pool_grid(grid_n=RSA_POOL_GRID_N)

    # NOTE: Uncomment to visualise gratings in stim space
    # x_min = pool["x"].min() + 5
    # x_max = pool["x"].max() - 5
    # y_min = pool["y"].min() + 5
    # y_max = pool["y"].max() - 5
    # x = np.linspace(x_min, x_max, 3)
    # y = np.linspace(y_min, y_max, 2)
    # dss = pd.DataFrame({'x':x.repeat(2), 'y':np.tile(y, 3)})
    # plot_stim_space_examples(dss, win, grating, PX_PER_CM)

    seed = f"{participant}_{session_num:03d}_{MODE}"
    schedule = make_rsa_schedule_table(
        pool,
        repeats_per_block=RSA_REPEATS_PER_BLOCK,
        n_blocks=RSA_BLOCKS,
        schedule_seed=seed,
    )

    # --------------------------- EEG init ----------------------------------------
    eeg = EEGPort(
        win,
        address=EEG_PORT_ADDRESS,
        enabled=EEG_ENABLED,
        default_ms=EEG_DEFAULT_PULSE_MS,
    )

    # --------------------------- data init ---------------------------------------
    trial_data = {
        "subject_id": [],
        "session_num": [],
        "session_part": [],
        "trial": [],
        "block_id": [],
        "block_trial": [],
        "rsa_item_id": [],
        "rsa_x": [],
        "rsa_y": [],
        "rsa_xt": [],
        "rsa_yt": [],
        "ts_iso": [],
        "eeg_enabled": [],
        "trigger_stim": [],
        "trigger_block_start": [],
        "trigger_block_end": [],
        "port_address": [],
        "t_stim": [],
    }

    flip_times = {"t_stim": np.nan}

    trial = n_done - 1
    current_row = None
    current_block = 0
    next_block = int(
        schedule["block_id"].iloc[n_done]) if n_done < len(schedule) else 0
    sf_cycles_per_pix = np.nan
    ori_deg = np.nan
    trig_stim = np.nan
    trig_block_start = np.nan
    trig_block_end = np.nan
    state_current = "state_init"
    state_entry = True
    running = True

    # --------------------------- State machine setup ------------------------------
    while running:
        if default_kb.getKeys(keyList=["escape"], waitRelease=False):
            eeg.pulse_now(TRIG["EXP_END"], global_clock=global_clock)
            break

        eeg.update(global_clock)

        # --------------------- STATE: INIT ---------------------
        if state_current == "state_init":
            if state_entry:
                state_clock.reset()
                msg_text.text = "Experiment ready.\n\nPress SPACE to begin."
                state_entry = False

            msg_text.draw()
            keys = kb.getKeys(keyList=["space"], waitRelease=False, clear=True)
            if keys:
                eeg.flip_pulse(TRIG["EXP_START"], global_clock=global_clock)
                if n_done >= len(schedule):
                    state_current = "state_finished"
                else:
                    state_current = "state_block_intro"
                state_entry = True
            win.flip()

        # --------------------- STATE: BLOCK INTRO ---------------------
        elif state_current == "state_block_intro":
            if state_entry:
                state_clock.reset()
                msg_text.text = intro_text if current_block == 0 else ready_text
                state_entry = False

            msg_text.draw()
            keys = kb.getKeys(keyList=["space"], waitRelease=False, clear=True)
            if keys:
                if current_block != 0:
                    eeg.flip_pulse(TRIG["RSA_READY_ONSET"],
                                   global_clock=global_clock)
                    win.flip()
                eeg.pulse_now(TRIG["RSA_BLOCK_START"],
                              global_clock=global_clock)
                trig_block_start = TRIG["RSA_BLOCK_START"]
                trig_block_end = np.nan
                current_block = next_block
                state_current = "state_trial_prep"
                state_entry = True
            else:
                win.flip()

        # --------------------- STATE: TRIAL PREP ---------------------
        elif state_current == "state_trial_prep":
            if state_entry:
                state_clock.reset()
                trial += 1
                if trial >= len(schedule):
                    eeg.pulse_now(TRIG["RSA_BLOCK_END"],
                                  global_clock=global_clock)
                    trig_block_end = int(TRIG["RSA_BLOCK_END"])
                    state_current = "state_finished"
                    state_entry = True
                else:
                    current_row = schedule.iloc[trial]
                    if int(current_row["block_id"]) != current_block:
                        eeg.pulse_now(TRIG["RSA_BLOCK_END"],
                                      global_clock=global_clock)
                        trig_block_end = int(TRIG["RSA_BLOCK_END"])
                        next_block = int(current_row["block_id"])
                        trial -= 1
                        state_current = "state_block_intro"
                        state_entry = True
                    else:
                        sf_cycles_per_pix, ori_deg = stim_xy_to_sf_ori_deg(
                            current_row["x"],
                            current_row["y"],
                            PX_PER_CM,
                        )
                        sf_cycles_per_pix = float(
                            np.asarray(sf_cycles_per_pix))
                        ori_deg = float(np.asarray(ori_deg))
                        trig_stim = np.nan
                        flip_times["t_stim"] = np.nan
                        grating.sf = sf_cycles_per_pix
                        grating.ori = ori_deg
                        grating.phase = 0.0
                        state_current = "state_stim"
                        state_entry = True

        # --------------------- STATE: STIM ---------------------
        elif state_current == "state_stim":
            if state_entry:
                state_clock.reset()
                eeg.flip_pulse(TRIG["RSA_STIM_ONSET"], global_clock=global_clock)
                win.callOnFlip(lambda: flip_times.__setitem__("t_stim", global_clock.getTime()))
                trig_stim = int(TRIG["RSA_STIM_ONSET"])
                state_entry = False

            fix_h.draw()
            fix_v.draw()
            grating.draw()

            if state_clock.getTime() >= RSA_SOA_SEC:

                xt, yt = transform_stim(current_row["x"], current_row["y"])

                trial_data["subject_id"].append(participant)
                trial_data["session_num"].append(session_num)
                trial_data["session_part"].append(session_part)
                trial_data["trial"].append(trial)
                trial_data["block_id"].append(int(current_row["block_id"]))
                trial_data["block_trial"].append(int(current_row["block_trial"]))
                trial_data["rsa_item_id"].append(int(current_row["item_id"]))
                trial_data["rsa_x"].append(float(current_row["x"]))
                trial_data["rsa_y"].append(float(current_row["y"]))
                trial_data["rsa_xt"].append(xt)
                trial_data["rsa_yt"].append(yt)
                trial_data["ts_iso"].append(datetime.now().isoformat())
                trial_data["eeg_enabled"].append(int(bool(EEG_ENABLED)))
                trial_data["trigger_stim"].append(trig_stim)
                trial_data["trigger_block_start"].append(trig_block_start)
                trial_data["trigger_block_end"].append(trig_block_end)
                trial_data["port_address"].append(
                    EEG_PORT_ADDRESS if EEG_ENABLED else "")
                trial_data["t_stim"].append(flip_times["t_stim"])
                pd.DataFrame(trial_data).to_csv(full_path, index=False)

                trig_block_start = np.nan
                trig_block_end = np.nan
                state_current = "state_trial_prep"
                state_entry = True

            win.flip()

        # --------------------- STATE: FINISHED ---------------------
        elif state_current == "state_finished":
            if state_entry:
                state_clock.reset()
                msg_text.text = "Thank you for being awesome!\nPress SPACE to exit."
                state_entry = False

            msg_text.draw()
            keys = kb.getKeys(keyList=["space"], waitRelease=False, clear=True)
            if keys:
                eeg.flip_pulse(TRIG["EXP_END"], global_clock=global_clock)
                running = False
            win.flip()

    # --------------------------- Cleanup ------------------------------------------
    eeg.close()
    win.close()
    core.quit()
    sys.exit()
