import math
import random
from datetime import datetime
import numpy as np
import pandas as pd
from psychopy import core, event, visual

# define function to create df of numerical stroop number pairs
def make_stroop_pairs(n_total, p_incongruent, random_seed=None):
    rng = np.random.default_rng(random_seed)
    ds_ns_rec = []

    num_list = [2, 3, 4, 5, 6, 7, 8]

    # congruency counts
    n_incongruent = int(np.ceil(p_incongruent * n_total))
    n_congruent = n_total - n_incongruent

    # cue split (same style as before)
    half_incon = n_incongruent // 2
    half_con = n_congruent // 2
    incon_count = 0
    con_count = 0

    # determine number of correct trials per side
    # determine number of correct trials per condition
    target_left_total = n_total // 2
    incon_left_corr = n_incongruent // 2
    con_left_corr = target_left_total - incon_left_corr
    incon_right_corr = n_incongruent - incon_left_corr
    con_right_corr = n_congruent - con_left_corr

    # create a list for correct sides ["left"] and ["right"] per condition
    corr_side_incon = ["left"] * incon_left_corr + ["right"] * incon_right_corr
    corr_side_con = ["left"] * con_left_corr + ["right"] * con_right_corr

    # shuffle rows
    rng.shuffle(corr_side_incon)
    rng.shuffle(corr_side_con)

    incon_idx = 0
    con_idx = 0

    for trl in range(n_total):
        num_left, num_right = rng.choice(num_list, size=2, replace=False)

        if trl < n_incongruent:
            congruency = "incongruent"

            # setting correct side to the next ["left"] or ["right"] in the list
            corr_side_target = corr_side_incon[incon_idx]
            incon_idx += 1

            # setting first half of cues to size, other half to value
            if incon_count < half_incon:
                cue = "Size"
            else:
                cue = "Value"
            incon_count += 1

            # if cue is "Value", correct side is side of larger/bigger value
            if cue == "Value":
                big_value_side = corr_side_target
            else:  
                # cue == "Size"
                # for incongruent trials, if correct target is on the left (meaning
                # smaller value in bigger font) then the number with bigger value
                # shuold be on the right
                if corr_side_target == "left":
                    big_value_side = "right"
                else:
                    # else larger value should go on the left
                    big_value_side = "left"

        else:
            congruency = "congruent"
            corr_side_target = corr_side_con[con_idx]
            con_idx += 1

            if con_count < half_con:
                cue = "Size"
            else:
                cue = "Value"
            con_count += 1

            # CONGRUENT: correct side must equal larger-value side
            # set big value side to the correct target side because in both size and
            # value trials, the bigger value is the right number
            big_value_side = corr_side_target

        # find small/large value and put on correct side 
        if big_value_side == "left":
            value_left = max(num_left, num_right)
            value_right = min(num_left, num_right)
        else:
            value_left = min(num_left, num_right)
            value_right = max(num_left, num_right)

        if congruency == "incongruent":
            if value_left > value_right:
                size_left = "small"
                size_right = "big"
            else:
                size_left = "big"
                size_right = "small"
        else:  # congruent
            if value_left > value_right:
                size_left = "big"
                size_right = "small"
            else:
                size_left = "small"
                size_right = "big"

        ds_ns_rec.append({
            "value_left": value_left,
            "size_left": size_left,
            "value_right": value_right,
            "size_right": size_right,
            "congruency": congruency,
            "cue": cue,
        })

    ds_ns = pd.DataFrame(ds_ns_rec)

    if random_seed is None:
        ds_ns = ds_ns.sample(frac=1).reset_index(drop=True)
    else:
        ds_ns = ds_ns.sample(
            frac=1,
            random_state=int(rng.integers(0, 2**32 - 1))
        ).reset_index(drop=True)

    return ds_ns

# check = ds_ns["value_left"] == ds_ns["value_right"]
