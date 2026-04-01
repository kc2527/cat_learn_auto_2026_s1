# -*- coding: utf-8 -*-
"""
Run categorical perception task using PsychoPy.
"""

from datetime import datetime, timedelta
import os
import sys
import random
import numpy as np
import pandas as pd
from psychopy import core, visual
from psychopy.hardware import keyboard
from util_func_eeg import EEGPort
from util_func_pid import prompt_for_pid_in_set
from util_func_session_man import resolve_session
from util_func_stimcat import build_cp_trial_runtime_from_pairs
from util_func_stimcat import key_to_interval
from util_func_stimcat import make_cp_pair_tables
from util_func_stimcat import make_cp_trial_table
from util_func_stimcat import plot_stim_space_examples
from util_func_stimcat import stim_xy_to_sf_ori_deg
from util_func_stimcat import transform_stim

EEG_ENABLED = False
EEG_PORT_ADDRESS = "0x3FB8"
EEG_DEFAULT_PULSE_MS = 50

TRIG = {
    # -------------------- Experiment structure --------------------
    "EXP_START": 10,
    "EXP_END": 15,
    "CP_MAIN_BLOCK_START": 31,
    "CP_MAIN_BLOCK_END": 32,
    "CP_PRACTICE_START": 33,
    "CP_PRACTICE_END": 34,
    "CP_ITI_ONSET": 40,
    "CP_INTERVAL2_ONSET": 42,
    # -------------------- Responses --------------------
    "CP_RESPONSE_PROMPT_ONSET": 43,
    "CP_RESP_1": 44,
    "CP_RESP_2": 45,
    "CP_RESP_TIMEOUT": 46,
    # -------------------- Stimulus onset --------------------
    "CP_STIM_WITHIN_A_NEAR_ONSET": 60,
    "CP_STIM_WITHIN_A_FAR_ONSET": 61,
    "CP_STIM_WITHIN_B_NEAR_ONSET": 62,
    "CP_STIM_WITHIN_B_FAR_ONSET": 63,
    "CP_STIM_ACROSS_NEAR_ONSET": 64,
    "CP_STIM_ACROSS_FAR_ONSET": 65,
}

PID_DIGITS = 3
MODE = "cp"
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
CP_PRACTICE_N = 24
CP_PRACTICE_FAR_N = 16
CP_PRACTICE_MODERATE_N = 8
CP_MAIN_REPS_PER_CELL = 34
CP_DIST_SMALL = 6.0
CP_DIST_LARGE = 15.0
CP_TOTAL_TRIALS = CP_PRACTICE_N + (6 * CP_MAIN_REPS_PER_CELL)
ITI_SEC = 0.8
ITI_JITTER_SEC = (0.0, 0.4)
INTERVAL_SEC = 0.2
PAIR_GAP_SEC = 0.15
ISI_SEC = 0.4
RESP_WINDOW_SEC = 1.5
PRACTICE_FEEDBACK_SEC = 0.6
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

    prompt_text = visual.TextStim(win,
                                  text="",
                                  color="white",
                                  height=30,
                                  wrapWidth=1600,
                                  pos=(0, 0))

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

    instructions_text = (
        "In each trial two stimulus pairs will flash on the screen (Interval 1 and Interval 2).\n"
        "In one interval the two stimuli are different. In the other interval they are the same.\n"
        "The stimuli can differ in bar thickness, angle, or both.\n"
        "Press 1 if the different pair was in interval 1.\n"
        "Press 2 if the different pair was in interval 2.\n"
        "Please keep your eyes centered on the middle of the stimulus.\n"
        "Try to respond as accurately as possible.\n\n"
        "Press SPACE to continue.")

    # --------------------------- response and clocks -----------------------------
    kb = keyboard.Keyboard()
    default_kb = keyboard.Keyboard()
    global_clock = core.Clock()
    state_clock = core.Clock()
    resp_clock = core.Clock()

    # --------------------------- Subject handling --------------------------------
    dir_data = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(dir_data, exist_ok=True)

    participant = prompt_for_pid_in_set(win, PID_DIGITS, ALLOWED_SUBJECT_IDS)

    # ---------------------------  session handling -------------------------------
    session_info = resolve_session(
        dir_data,
        participant,
        CP_TOTAL_TRIALS,
        resume_window=RESUME_WINDOW,
        new_session_cooldown=NEW_SESSION_COOLDOWN,
        task_tag=MODE,
    )
    session_num = session_info["session_num"]
    session_part = session_info["part_num"]
    full_path = session_info["full_path"]
    n_done = session_info["n_done"]

    # --------------------------- Stimuli  ----------------------------------------
    seed = f"{participant}_{session_num:03d}_{MODE}"
    pair_tables = make_cp_pair_tables(pool_seed=f"{seed}_pool")
    trials = make_cp_trial_table(
        practice_far_n=CP_PRACTICE_FAR_N,
        practice_moderate_n=CP_PRACTICE_MODERATE_N,
        main_reps_per_cell=CP_MAIN_REPS_PER_CELL,
        near_dist=CP_DIST_SMALL,
        far_dist=CP_DIST_LARGE,
        schedule_seed=seed,
    )

    preview_rows = []
    preview_rng = random.Random(f"{seed}_preview")
    for preview_idx in np.linspace(0, len(trials) - 1, 6, dtype=int):
        preview_runtime = build_cp_trial_runtime_from_pairs(
            trials.iloc[int(preview_idx)],
            pair_tables,
            preview_rng,
        )
        for stim_name in ["int1a", "int1b", "int2a", "int2b"]:
            preview_rows.append({
                "x": preview_runtime[stim_name]["x"],
                "y": preview_runtime[stim_name]["y"],
            })
    cp_preview = pd.DataFrame(preview_rows).drop_duplicates().reset_index(
        drop=True)

    # NOTE: Uncomment to visualise gratings in stim space
    # plot_stim_space_examples(cp_preview, win, grating, PX_PER_CM)

    # --------------------------- EEG init ----------------------------------------
    eeg = EEGPort(
        win,
        address=EEG_PORT_ADDRESS,
        enabled=EEG_ENABLED,
        default_ms=EEG_DEFAULT_PULSE_MS,
    )

    # --------------------------- EEG init ----------------------------------------
    trial_data = {
        "subject_id": [],
        "session_num": [],
        "session_part": [],
        "trial": [],
        "phase": [],
        "block_id": [],
        "condition_id": [],
        "cp_family": [],
        "pair_type": [],
        "distance_level": [],
        "distance_value": [],
        "different_interval": [],
        "resp": [],
        "corr": [],
        "rt_ms": [],
        "response_key_raw": [],
        "feedback": [],
        "trigger_i1": [],
        "trigger_i2": [],
        "trigger_resp": [],
        "t_i1": [],
        "t_i2": [],
        "t_resp": [],
        "i1a_x": [],
        "i1a_y": [],
        "i1b_x": [],
        "i1b_y": [],
        "i2a_x": [],
        "i2a_y": [],
        "i2b_x": [],
        "i2b_y": [],
        "ts_iso": [],
        "eeg_enabled": [],
        "port_address": [],
    }

    flip_times = {
        "t_i1": np.nan,
        "t_i2": np.nan,
    }

    def make_trial_rng(trial_index, stream_name):
        return random.Random(f"{seed}_{stream_name}_{int(trial_index):03d}")

    current_trial = None
    runtime = None
    iti_sec = 0.0
    trial = n_done - 1
    practice_started = n_done > 0
    practice_finished = n_done >= CP_PRACTICE_N
    main_started = n_done >= CP_PRACTICE_N

    response_key_raw = ""
    resp = ""
    corr = 0
    rt_ms = np.nan
    feedback = ""
    trig_i1 = np.nan
    trig_i2 = np.nan
    trig_resp = np.nan
    t_resp = np.nan

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
                state_current = "state_instructions"
                state_entry = True
            win.flip()

        # --------------------- STATE: INSTRUCTIONS ---------------------
        elif state_current == "state_instructions":
            if state_entry:
                state_clock.reset()
                msg_text.text = instructions_text
                state_entry = False

            msg_text.draw()
            keys = kb.getKeys(keyList=["space"], waitRelease=False, clear=True)
            if keys:
                if n_done < CP_PRACTICE_N:
                    state_current = "state_practice_intro"
                elif n_done < len(trials):
                    state_current = "state_main_intro"
                else:
                    state_current = "state_finished"
                state_entry = True
            win.flip()

        # --------------------- STATE: PRACTICE INTRO ---------------------
        elif state_current == "state_practice_intro":
            if state_entry:
                state_clock.reset()
                msg_text.text = "You will begin with a short practice block.\n\nPress SPACE to start practice."
                state_entry = False

            msg_text.draw()
            keys = kb.getKeys(keyList=["space"], waitRelease=False, clear=True)
            if keys:
                eeg.flip_pulse(TRIG["CP_PRACTICE_START"],
                               global_clock=global_clock)
                practice_started = True
                state_current = "state_iti"
                state_entry = True
            win.flip()

        # --------------------- STATE: MAIN INTRO ---------------------
        elif state_current == "state_main_intro":
            if state_entry:
                if practice_started and not practice_finished:
                    eeg.pulse_now(TRIG["CP_PRACTICE_END"],
                                  global_clock=global_clock)
                    practice_finished = True
                state_clock.reset()
                msg_text.text = (
                    "Practice complete.\n"
                    "Now the main task will begin.\n"
                    "The main trials will be harder. Please try and respond as accurately as possible.\n"
                    "Keep your eyes centered on the middle of the stimulus.\n"
                    "Remember: 1 = interval 1, 2 = interval 2.\n\n"
                    "Press SPACE to begin.")
                state_entry = False

            msg_text.draw()
            keys = kb.getKeys(keyList=["space"], waitRelease=False, clear=True)
            if keys:
                eeg.flip_pulse(TRIG["CP_MAIN_BLOCK_START"],
                               global_clock=global_clock)
                main_started = True
                state_current = "state_iti"
                state_entry = True
            win.flip()

        # --------------------- STATE: ITI ---------------------
        elif state_current == "state_iti":
            if state_entry:
                state_clock.reset()
                eeg.flip_pulse(TRIG["CP_ITI_ONSET"], global_clock=global_clock)
                next_trial = trial + 1
                iti_rng = make_trial_rng(next_trial, "iti")
                iti_sec = ITI_SEC + iti_rng.uniform(ITI_JITTER_SEC[0],
                                                    ITI_JITTER_SEC[1])
                state_entry = False

            fix_h.draw()
            fix_v.draw()

            if state_clock.getTime() >= iti_sec:
                trial += 1
                if trial >= len(trials):
                    if not practice_finished and practice_started:
                        eeg.pulse_now(TRIG["CP_PRACTICE_END"],
                                      global_clock=global_clock)
                    if main_started:
                        eeg.pulse_now(TRIG["CP_MAIN_BLOCK_END"],
                                      global_clock=global_clock)
                    state_current = "state_finished"
                    state_entry = True
                else:
                    current_trial = trials.iloc[trial]
                    runtime_rng = make_trial_rng(trial, "runtime")
                    runtime = build_cp_trial_runtime_from_pairs(
                        current_trial,
                        pair_tables,
                        runtime_rng,
                    )
                    response_key_raw = "none"
                    resp = ""
                    corr = 0
                    rt_ms = np.nan
                    feedback = ""
                    trig_resp = np.nan
                    t_resp = np.nan
                    flip_times["t_i1"] = np.nan
                    flip_times["t_i2"] = np.nan

                    trig_i1 = np.nan
                    if current_trial["phase"] == "main":
                        family = current_trial["family"]
                        level = current_trial["distance_level"]
                        if family == "within_A" and level == "near":
                            trig_i1 = TRIG["CP_STIM_WITHIN_A_NEAR_ONSET"]
                        elif family == "within_A" and level == "far":
                            trig_i1 = TRIG["CP_STIM_WITHIN_A_FAR_ONSET"]
                        elif family == "within_B" and level == "near":
                            trig_i1 = TRIG["CP_STIM_WITHIN_B_NEAR_ONSET"]
                        elif family == "within_B" and level == "far":
                            trig_i1 = TRIG["CP_STIM_WITHIN_B_FAR_ONSET"]
                        elif family == "between_AB" and level == "near":
                            trig_i1 = TRIG["CP_STIM_ACROSS_NEAR_ONSET"]
                        else:
                            trig_i1 = TRIG["CP_STIM_ACROSS_FAR_ONSET"]

                    sf_cycles_per_pix, ori_deg = stim_xy_to_sf_ori_deg(
                        runtime["int1a"]["x"],
                        runtime["int1a"]["y"],
                        PX_PER_CM,
                    )
                    grating.sf = float(np.asarray(sf_cycles_per_pix))
                    grating.ori = float(np.asarray(ori_deg))
                    grating.phase = 0.0
                    state_current = "state_interval1a"
                    state_entry = True
            win.flip()

        # --------------------- STATE: INTERVAL 1A ---------------------
        elif state_current == "state_interval1a":
            if state_entry:
                state_clock.reset()
                if not np.isnan(trig_i1):
                    eeg.flip_pulse(int(trig_i1), global_clock=global_clock)
                win.callOnFlip(lambda: flip_times.__setitem__("t_i1", global_clock.getTime()))
                state_entry = False

            grating.draw()

            if state_clock.getTime() >= INTERVAL_SEC:
                state_current = "state_gap1"
                state_entry = True
            win.flip()

        # --------------------- STATE: GAP 1 ---------------------
        elif state_current == "state_gap1":
            if state_entry:
                state_clock.reset()
                sf_cycles_per_pix, ori_deg = stim_xy_to_sf_ori_deg(
                    runtime["int1b"]["x"],
                    runtime["int1b"]["y"],
                    PX_PER_CM,
                )
                grating.sf = float(np.asarray(sf_cycles_per_pix))
                grating.ori = float(np.asarray(ori_deg))
                grating.phase = 0.0
                state_entry = False

            if state_clock.getTime() >= PAIR_GAP_SEC:
                state_current = "state_interval1b"
                state_entry = True
            win.flip()

        # --------------------- STATE: INTERVAL 1B ---------------------
        elif state_current == "state_interval1b":
            if state_entry:
                state_clock.reset()
                state_entry = False

            grating.draw()

            if state_clock.getTime() >= INTERVAL_SEC:
                state_current = "state_isi"
                state_entry = True
            win.flip()

        # --------------------- STATE: ISI ---------------------
        elif state_current == "state_isi":
            if state_entry:
                state_clock.reset()
                sf_cycles_per_pix, ori_deg = stim_xy_to_sf_ori_deg(
                    runtime["int2a"]["x"],
                    runtime["int2a"]["y"],
                    PX_PER_CM,
                )
                grating.sf = float(np.asarray(sf_cycles_per_pix))
                grating.ori = float(np.asarray(ori_deg))
                grating.phase = 0.0
                state_entry = False

            fix_h.draw()
            fix_v.draw()

            if state_clock.getTime() >= ISI_SEC:
                state_current = "state_interval2a"
                state_entry = True
            win.flip()

        # --------------------- STATE: INTERVAL 2A ---------------------
        elif state_current == "state_interval2a":
            if state_entry:
                state_clock.reset()
                eeg.flip_pulse(TRIG["CP_INTERVAL2_ONSET"],
                               global_clock=global_clock)
                win.callOnFlip(lambda: flip_times.__setitem__("t_i2", global_clock.getTime()))
                trig_i2 = int(TRIG["CP_INTERVAL2_ONSET"])
                state_entry = False

            grating.draw()

            if state_clock.getTime() >= INTERVAL_SEC:
                state_current = "state_gap2"
                state_entry = True
            win.flip()

        # --------------------- STATE: GAP 2 ---------------------
        elif state_current == "state_gap2":
            if state_entry:
                state_clock.reset()
                sf_cycles_per_pix, ori_deg = stim_xy_to_sf_ori_deg(
                    runtime["int2b"]["x"],
                    runtime["int2b"]["y"],
                    PX_PER_CM,
                )
                grating.sf = float(np.asarray(sf_cycles_per_pix))
                grating.ori = float(np.asarray(ori_deg))
                grating.phase = 0.0
                state_entry = False

            if state_clock.getTime() >= PAIR_GAP_SEC:
                state_current = "state_interval2b"
                state_entry = True
            win.flip()

        # --------------------- STATE: INTERVAL 2B ---------------------
        elif state_current == "state_interval2b":
            if state_entry:
                state_clock.reset()
                state_entry = False

            grating.draw()

            if state_clock.getTime() >= INTERVAL_SEC:
                state_current = "state_response"
                state_entry = True
            win.flip()

        # --------------------- STATE: RESPONSE ---------------------
        elif state_current == "state_response":
            if state_entry:
                state_clock.reset()
                resp_clock.reset()
                kb.clearEvents()
                prompt_text.text = "Which interval had the different pair?\n1 = Interval 1, 2 = Interval 2"
                eeg.flip_pulse(TRIG["CP_RESPONSE_PROMPT_ONSET"],
                               global_clock=global_clock)
                win.callOnFlip(kb.clock.reset)
                win.callOnFlip(resp_clock.reset)
                state_entry = False

            prompt_text.draw()

            keys = kb.getKeys(keyList=["1", "2", "num_1", "num_2"],
                              waitRelease=False,
                              clear=False)
            if keys:
                response_key_raw = keys[-1].name
                picked = key_to_interval(response_key_raw)
                if picked is not None:
                    resp = picked
                rt_ms = keys[-1].rt * 1000.0
                corr = 1 if picked == runtime["diff_interval"] else 0
                if picked == 1:
                    eeg.pulse_now(TRIG["CP_RESP_1"], global_clock=global_clock)
                    trig_resp = int(TRIG["CP_RESP_1"])
                    t_resp = global_clock.getTime()
                elif picked == 2:
                    eeg.pulse_now(TRIG["CP_RESP_2"], global_clock=global_clock)
                    trig_resp = int(TRIG["CP_RESP_2"])
                    t_resp = global_clock.getTime()
                state_current = "state_feedback"
                state_entry = True
            elif resp_clock.getTime() >= RESP_WINDOW_SEC:
                eeg.pulse_now(TRIG["CP_RESP_TIMEOUT"],
                              global_clock=global_clock)
                trig_resp = int(TRIG["CP_RESP_TIMEOUT"])
                t_resp = global_clock.getTime()
                state_current = "state_feedback"
                state_entry = True

            win.flip()

        # --------------------- STATE: FEEDBACK ---------------------
        elif state_current == "state_feedback":
            if state_entry:
                state_clock.reset()
                if current_trial["phase"] == "practice":
                    if response_key_raw == "none":
                        feedback = "too_slow"
                        prompt_text.text = "Too slow"
                    elif corr == 1:
                        feedback = "correct"
                        prompt_text.text = "Correct"
                    else:
                        feedback = "incorrect"
                        prompt_text.text = "Incorrect"
                else:
                    feedback = ""

                trial_data["subject_id"].append(participant)
                trial_data["session_num"].append(session_num)
                trial_data["session_part"].append(session_part)
                trial_data["trial"].append(trial)
                trial_data["phase"].append(current_trial["phase"])
                trial_data["block_id"].append(int(current_trial["block_id"]))
                trial_data["condition_id"].append(current_trial["condition_id"])
                trial_data["cp_family"].append(runtime["cp_family"])
                trial_data["pair_type"].append(runtime["pair_type"])
                trial_data["distance_level"].append(runtime["cp_distance_level"])
                trial_data["distance_value"].append(runtime["distance"])
                trial_data["different_interval"].append(runtime["diff_interval"])
                trial_data["resp"].append(resp)
                trial_data["corr"].append(corr)
                trial_data["rt_ms"].append(rt_ms)
                trial_data["response_key_raw"].append(response_key_raw)
                trial_data["feedback"].append(feedback)
                trial_data["trigger_i1"].append(trig_i1)
                trial_data["trigger_i2"].append(trig_i2)
                trial_data["trigger_resp"].append(trig_resp)
                trial_data["t_i1"].append(flip_times["t_i1"])
                trial_data["t_i2"].append(flip_times["t_i2"])
                trial_data["t_resp"].append(t_resp)
                trial_data["i1a_x"].append(runtime["int1a"]["x"])
                trial_data["i1a_y"].append(runtime["int1a"]["y"])
                trial_data["i1b_x"].append(runtime["int1b"]["x"])
                trial_data["i1b_y"].append(runtime["int1b"]["y"])
                trial_data["i2a_x"].append(runtime["int2a"]["x"])
                trial_data["i2a_y"].append(runtime["int2a"]["y"])
                trial_data["i2b_x"].append(runtime["int2b"]["x"])
                trial_data["i2b_y"].append(runtime["int2b"]["y"])
                trial_data["ts_iso"].append(datetime.now().isoformat())
                trial_data["eeg_enabled"].append(int(bool(EEG_ENABLED)))
                trial_data["port_address"].append(
                    EEG_PORT_ADDRESS if EEG_ENABLED else "")

                pd.DataFrame(trial_data).to_csv(full_path, index=False)

                state_entry = False

            if current_trial["phase"] == "practice":
                prompt_text.draw()
                if state_clock.getTime() >= PRACTICE_FEEDBACK_SEC:
                    if trial >= len(trials) - 1:
                        eeg.pulse_now(TRIG["CP_PRACTICE_END"],
                                      global_clock=global_clock)
                        state_current = "state_finished"
                    elif trials.iloc[trial + 1]["phase"] == "main":
                        eeg.pulse_now(TRIG["CP_PRACTICE_END"],
                                      global_clock=global_clock)
                        practice_finished = True
                        state_current = "state_main_intro"
                    else:
                        state_current = "state_iti"
                    state_entry = True
            else:
                if trial >= len(trials) - 1:
                    eeg.pulse_now(TRIG["CP_MAIN_BLOCK_END"],
                                  global_clock=global_clock)
                    state_current = "state_finished"
                else:
                    state_current = "state_iti"
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
