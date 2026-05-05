import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

dir_data = '../at_home_data'
dir_data_lab = '../behavioural_data'

df_train_rec = []
df_lab_rec = []

# not reading in cp task
for fd in os.listdir(dir_data):
    dir_data_fd = os.path.join(dir_data, fd)
    if os.path.isdir(dir_data_fd):
        for fs in os.listdir(dir_data_fd):
            f_full_path = os.path.join(dir_data_fd, fs)
            if os.path.isfile(f_full_path) and "task_cp_" not in fs:
                
                df = pd.read_csv(f_full_path)
                df['f_name'] = fs
                df_train_rec.append(df)

for fd in os.listdir(dir_data_lab):
    dir_data_lab_fd = os.path.join(dir_data_lab, fd)
    if os.path.isdir(dir_data_lab_fd):
        for fs in os.listdir(dir_data_lab_fd):
            f_full_path = os.path.join(dir_data_lab_fd, fs)
            if os.path.isfile(f_full_path) and fs.endswith(".csv"):

                # in session 1, participant 875 did 10 train trials and 60
                # probe trials taking first 10 train trials and adding it to
                # part 2 (540 train, 100 probe) -- excluding for the moment

                # in session 4, ActiView had a syncing error and crached 30
                # trials in with participant 875, restarted experiment clean --
                # removing extra data file
                if fs not in ['sub_875_sess_001_part_001_date_2026_04_03_data.csv',
                              'sub_875_sess_001_part_002_date_2026_04_03_data.csv',
                              'sub_875_sess_004_part_001_date_2026_04_24_data (1).csv'
                              ]:

                    df = pd.read_csv(f_full_path)
                    df['f_name'] = fs
                    df_lab_rec.append(df)

d_home = pd.concat(df_train_rec, ignore_index=True)
d_lab = pd.concat(df_lab_rec, ignore_index=True)

block_size = 25

# NOTE: Setting dfs up 
# at-home data
# 300 trials -- train
d_home = d_home.sort_values(["subject_id", "session_num", "session_part",
                               "trial"]).reset_index(drop=True)
d_home = d_home[d_home['session_num'] != 17]

d_home['acc'] = (d_home['cat'] == d_home['resp']).astype(int)
d_home['trial'] = d_home.groupby(['subject_id', 'session_num']).cumcount()
d_home['n_trials'] = d_home.groupby(['subject_id', 'session_num'])['trial'].transform('count')
d_home['block'] = d_home.groupby(['subject_id', 'session_num'])['trial'].transform(lambda x: x // block_size)
d_home = d_home.drop(columns=['value_left', 'size_left', 'value_right',
                                'size_right', 'congruency', 'cue',
                                'resp_key_ns', 'resp_ns', 'fb_ns', 'rt_ns',
                                't_cue_ns', 't_fb_ns'])

# dual task day (17)
# 300 trials -- train + numerical stroop
d_dt = d_home.sort_values(["subject_id", "session_num", "session_part",
                             "trial"]).reset_index(drop=True)
d_dt = d_dt[d_dt['session_num'] == 17]

# TODO: next line raises error
d_dt['acc'] = (d_dt['cat'] == d_home['resp']).astype(int)
d_dt['trial'] = d_dt.groupby(['subject_id', 'session_num']).cumcount()
d_dt['n_trials'] = d_dt.groupby(['subject_id', 'session_num'])['trial'].transform('count')
d_dt['block'] = d_dt.groupby(['subject_id', 'session_num'])['trial'].transform(lambda x: x // block_size)

# lab data 
# 550 trials -- train, 100 trials -- test
d_lab = d_lab.sort_values(["subject_id", "session_num", "session_part",
                             "trial"]).reset_index(drop=True)

d_lab['acc'] = (d_lab['cat'] == d_lab['resp']).astype(int)
d_lab['trial'] = d_lab.groupby(['subject_id', 'session_num']).cumcount()
d_lab['n_trials'] = d_lab.groupby(['subject_id', 'session_num'])['trial'].transform('count')
d_lab['block'] = d_lab.groupby(['subject_id', 'session_num'])['trial'].transform(lambda x: x // block_size)

# ds for all, train, and test trials 
d_lab_all = (d_lab.groupby(["subject_id", "session_num", "block",
                              "probe_condition", "phase"],
                             as_index=False)["acc"].mean().sort_values(["session_num",
                                                                        "subject_id",
                                                                        "block"]))

d_lab_train = d_lab[d_lab['phase'] == 'train'].groupby(['subject_id',
                                                           'session_num',
                                                           'probe_condition',
                                                           'block']).agg({'acc':
                                                                          'mean'}).reset_index()

d_lab_test = d_lab[d_lab['phase'] == 'test'].groupby(['subject_id',
                                                         'session_num',
                                                         'probe_condition',
                                                         'block']).agg({'acc':
                                                                        'mean'}).reset_index()

# NOTE: Inspect performance
# average accuracy per lab day
d_lab_pd_avg = d_lab_train.groupby(['subject_id', 'session_num']).agg({'acc': 'mean'}).reset_index()

# looking for average day accuracies below 75% after the first lab session 
below_exp = d_lab_pd_avg[(d_lab_pd_avg['acc'] < 0.75) & (d_lab_pd_avg['session_num'] != 1)]

# participants 2, 189, and 639 have below 70% accuracy by the end of lab day 2
# (after 6 sessions) -- inspecting at home performance 
home_inspect = d_home[(d_home['subject_id'] == 2) |
                       (d_home['subject_id'] == 189) |
                       (d_home['subject_id'] == 639)]

# at-home data shows that they are not breaking 80% at home by days 7, 6, and 4
# respectively
home_pd_avg = home_inspect.groupby(['subject_id', 'session_num']).agg({'acc': 'mean'}).reset_index()

# NOTE: Plots 
# -- HOME -- 
days_home = d_home["session_num"].unique()[:16]

# accuracy across task across days
fig, axes = plt.subplots(2, len(days_home), squeeze = False)

# average accuracy in task across days


# -- DUAL TASK --
# average accuracy in task


# average accuracy compared to last at home day and last lab day


# -- LAB --
days_lab = d_lab["session_num"].unique()[:5]

# accuracy across task across days



# average accuracy in task across days


# average accuracy across participants across days


# 90 vs 180 cost
# take participants 134, 213, 268, 358, and 482 session 1 out of d as they
# completed 650 trials of train, no test trials were completed
d_cost = d_lab_all.copy() 

drop_subs = [134, 213, 268, 358, 482]
d_cost = d_cost[~((d_cost['session_num'] == 1) & (d_cost['subject_id'].isin(drop_subs)))]

pre_block = d_cost.loc[d_cost["phase"] == "train", "block"].max()
post_block = d_cost.loc[d_cost["phase"] == "test", "block"].min()

pre_90 = d_cost[(d_cost["block"] == pre_block) &
                 (d_cost["probe_condition"] == 90)].groupby("session_num")["acc"].mean()
post_90 = d_cost[(d_cost["block"] == post_block) &
                  (d_cost["probe_condition"] == 90)].groupby("session_num")["acc"].mean()

pre_180 = d_cost[(d_cost["block"] == pre_block) &
                  (d_cost["probe_condition"] == 180)].groupby("session_num")["acc"].mean()
post_180 = d_cost[(d_cost["block"] == post_block) &
                   (d_cost["probe_condition"] == 180)].groupby("session_num")["acc"].mean()

cost_90 = pre_90 - post_90
cost_180 = pre_180 - post_180

cost = pd.concat(
    [cost_90.rename("cost").reset_index().assign(probe_condition="90"),
     cost_180.rename("cost").reset_index().assign(probe_condition="180")],
     ignore_index=True)

# plot
fig, ax = plt.subplots(1, 1, squeeze = False)
sns.barplot(data=cost, x='session_num', y='cost', hue='probe_condition', ax=ax[0,0])
plt.tight_layout()
plt.show()

### HOW MATT WOULD DO IT (JUST THE PANDAS VERSION OF THE DATA.TABLE APPROACH IN 2020)
d_cost = d_lab_all.copy() 
d_cost = d_cost[~((d_cost['session_num'] == 1) & (d_cost['subject_id'].isin(drop_subs)))]

d = d_cost[d_cost['block'] > 17] # equating number of train and test blocks for fair compare
dd = d.groupby(['subject_id', 'session_num', 'phase',
                           'probe_condition'])['acc'].mean().reset_index()

dd_wide = (
  dd.pivot_table(
      index=["subject_id", "session_num", "probe_condition"],
      columns="phase",
      values="acc",
      aggfunc="mean"
  )
  .reset_index()
)

dd_wide['diff_score'] = dd_wide['train'] - dd_wide['test']

dd_wide['probe_condition'] = dd_wide['probe_condition'].astype('category')
dd_wide['subject_id'] = dd_wide['subject_id'].astype('category')

sns.set_palette('rocket', 2)

fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(6, 6))
sns.pointplot(data=dd_wide,
              x = 'session_num',
              y = 'diff_score',
              hue = 'probe_condition',
              linestyle='none',
              dodge=True
)
plt.show()

fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(10, 5))
sns.lineplot(data=dd_wide[dd_wide['probe_condition'] == 90],
             x = 'session_num',
             y = 'diff_score',
             hue = 'subject_id',
             ax=ax[0, 0]
)
sns.lineplot(data=dd_wide[dd_wide['probe_condition'] == 180],
             x = 'session_num',
             y = 'diff_score',
             hue = 'subject_id',
             ax=ax[0, 1]
)
sns.move_legend(ax[0, 0], "upper left", bbox_to_anchor=(1, 1))
sns.move_legend(ax[0, 1], "upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()
