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
            if os.path.isfile(f_full_path) and 'task_cp_' not in fs:
                
                df = pd.read_csv(f_full_path)
                df['f_name'] = fs
                df_train_rec.append(df)

for fd in os.listdir(dir_data_lab):
    dir_data_lab_fd = os.path.join(dir_data_lab, fd)
    if os.path.isdir(dir_data_lab_fd):
        for fs in os.listdir(dir_data_lab_fd):
            f_full_path = os.path.join(dir_data_lab_fd, fs)
            if os.path.isfile(f_full_path) and fs.endswith('.csv'):

                # in session 1, participant 875 did 10 train trials and 60
                # probe trials taking first 10 train trials and adding it to
                # part 2 (540 train, 100 probe) -- excluding for the moment

                # in session 4, ActiView had a syncing error and crached 30
                # trials in with participant 875, restarted experiment clean --
                # removing extra data file
                if fs not in ['sub_875_sess_001_part_001_date_2026_04_03_data (1).csv',
                              'sub_875_sess_001_part_002_date_2026_04_03_data.csv',
                              'sub_875_sess_004_part_001_date_2026_04_24_data (1).csv'
                              ]:

                    df = pd.read_csv(f_full_path)
                    df['f_name'] = fs
                    df_lab_rec.append(df)

d_lab = pd.concat(df_lab_rec, ignore_index=True)
d_home = pd.concat(df_train_rec, ignore_index=True)

block_size = 25

# NOTE: Setting dfs up 
# lab data 
# 550 trials -- train, 100 trials -- test
dd_lab = d_lab.sort_values(['subject_id', 'session_num', 'session_part',
                             'trial']).reset_index(drop=True)

dd_lab['acc'] = (dd_lab['cat'] == dd_lab['resp']).astype(int)
dd_lab['trial'] = dd_lab.groupby(['subject_id', 'session_num']).cumcount()
dd_lab['n_trials'] = dd_lab.groupby(['subject_id', 'session_num'])['trial'].transform('count')
dd_lab['block'] = dd_lab.groupby(['subject_id', 'session_num'])['trial'].transform(lambda x: x // block_size)

# ds for all, train, and test trials 
dd_lab_all = (dd_lab.groupby(['subject_id', 'session_num', 'block',
                              'probe_condition', 'phase'],
                             as_index=False)['acc'].mean().sort_values(['session_num',
                                                                        'subject_id',
                                                                        'block']))

dd_lab_train = dd_lab[dd_lab['phase'] == 'train'].groupby(['subject_id',
                                                           'session_num',
                                                           'probe_condition',
                                                           'block'])['acc'].mean().reset_index()

dd_lab_test = dd_lab[dd_lab['phase'] == 'test'].groupby(['subject_id',
                                                         'session_num',
                                                         'probe_condition',
                                                         'block'])['acc'].mean().reset_index()
# at-home data
# 300 trials -- train
dd_home = d_home.sort_values(['subject_id', 'session_num', 'session_part',
                               'trial']).reset_index(drop=True)
dd_home = dd_home[dd_home['session_num'] != 17]

dd_home['acc'] = (dd_home['cat'] == dd_home['resp']).astype(int)
dd_home['trial'] = dd_home.groupby(['subject_id', 'session_num']).cumcount()
dd_home['n_trials'] = dd_home.groupby(['subject_id', 'session_num'])['trial'].transform('count')
dd_home['block'] = dd_home.groupby(['subject_id', 'session_num'])['trial'].transform(lambda x: x // block_size)
dd_home = dd_home.drop(columns=['value_left', 'size_left', 'value_right',
                                'size_right', 'congruency', 'cue',
                                'resp_key_ns', 'resp_ns', 'fb_ns', 'rt_ns',
                                't_cue_ns', 't_fb_ns'])

# dual task day (17)
# 300 trials -- train + numerical stroop
d_dt = d_home.sort_values(['subject_id', 'session_num', 'session_part',
                             'trial']).reset_index(drop=True)
d_dt = d_dt[(d_dt['session_num'] == 16) | (d_dt['session_num'] == 17)]

d_dt['acc'] = (d_dt['cat'] == d_dt['resp']).astype(int)
d_dt['trial'] = d_dt.groupby(['subject_id', 'session_num']).cumcount()
d_dt['n_trials'] = d_dt.groupby(['subject_id', 'session_num'])['trial'].transform('count')
d_dt['block'] = d_dt.groupby(['subject_id', 'session_num'])['trial'].transform(lambda x: x // block_size)

# NOTE: Inspect performance
# -- LAB -- 
# average accuracy per lab day (train trials)
dd_lab_pd_avg = dd_lab_train.groupby(['subject_id', 'session_num'])['acc'].mean().reset_index()

# looking for average day accuracies below 75% after the first lab session 
below_exp = dd_lab_pd_avg[(dd_lab_pd_avg['acc'] < 0.75) & (dd_lab_pd_avg['session_num'] != 1)]

# -- HOME -- 
dd_home_pd_avg = dd_home.groupby(['subject_id', 'session_num'])['acc'].mean().reset_index()

# participants 2, 189, and 639 have below 70% accuracy by the end of lab day 2
# (after 6 sessions) -- at-home data shows that they are not breaking 80% at
# home by days 7, 6, and 4 respectively
home_inspect = dd_home[(dd_home['subject_id'] == 2) |
                       (dd_home['subject_id'] == 189) |
                       (dd_home['subject_id'] == 639)]

dd_inspect_avg = home_inspect.groupby(['subject_id', 'session_num'])['acc'].mean().reset_index()

# -- DUAL TASK -- 
d_dt_acc = d_dt.groupby(['subject_id', 'session_num'])['acc'].mean().reset_index()

# NOTE: Plots 
# -- LAB --
days_lab = sorted(d_lab['session_num'].unique()[:5])

# accuracy across task across days
fig, ax = plt.subplots(1, len(days_lab), squeeze = False, figsize=(24, 3.5), sharey=True)
for a, day in zip(ax.flat, days_lab):
      sns.lineplot(
          data=dd_lab[dd_lab['session_num'] == day],
          x='block',
          y='acc',
          hue='subject_id',
          legend=False,
          errorbar=None,
          ax=a
      )
      a.set_title(f'Day {day}')
      a.set_ylim(0, 1)
plt.tight_layout()
plt.show()

# average accuracy in task (in train trials) across days
dd_avg_total_lab = dd_lab_pd_avg.groupby(['session_num'])['acc'].mean().reset_index()

fig, ax = plt.subplots(1, 1, squeeze = False)
sns.pointplot(data=dd_avg_total_lab,
              x='session_num',
              y='acc',
              )
plt.tight_layout()
plt.show()

# average accuracy across participants across days
dd_lab_pd_avg['subject_id'] = dd_lab_pd_avg['subject_id'].astype('category')

fig, ax = plt.subplots(1, 1, squeeze = False)
sns.pointplot(data=dd_lab_pd_avg,
              x='session_num',
              y='acc',
              hue='subject_id',
              )
plt.tight_layout()
plt.show()

# -- HOME -- 
days_home = dd_home['session_num'].unique()[:16]

# accuracy across whole task across days
fig, ax = plt.subplots(1, len(days_home), squeeze = False, figsize=(24, 3.5), sharey=True)
for a, day in zip(ax.flat, days_home):
      sns.lineplot(
          data=dd_home[dd_home['session_num'] == day],
          x='block',
          y='acc',
          hue='subject_id',
          legend=False,
          errorbar=None,
          ax=a
      )
      a.set_title(f'Day {day}')
      a.set_ylim(0, 1)
plt.tight_layout()
plt.show()

# total average accuracy per day 
dd_avg_total_home = dd_home_pd_avg.groupby(['session_num'])['acc'].mean().reset_index()

fig, ax = plt.subplots(1, 1, squeeze = False)
sns.pointplot(data=dd_avg_total_home,
              x='session_num',
              y='acc',
              )
plt.tight_layout()
plt.show()

# average accuracy in task per day across days across all participants
dd_home_pd_avg['subject_id'] = dd_home_pd_avg['subject_id'].astype('category')

fig, ax = plt.subplots(1, 1, squeeze = False)
sns.pointplot(data=dd_home_pd_avg,
              x='session_num',
              y='acc',
              hue='subject_id',
              )
plt.tight_layout()
plt.show()

# -- DUAL TASK --
# average accuracy in task
d_dt_acc['subject_id'] = d_dt_acc['subject_id'].astype('category')
    
fig, ax = plt.subplots(1, 1, squeeze = False)
sns.pointplot(data=d_dt_acc,
              x='session_num',
              y='acc',
              hue='subject_id',
              )
plt.tight_layout()
plt.show()

# -- 90 vs 180 COST -- 
# take participants 134, 213, 268, 358, and 482 session 1 out of d as they
# completed 650 trials of train, no test trials were completed
d_cost = dd_lab_all.copy() 

drop_subs = [134, 213, 268, 358, 482]
d_cost = d_cost[~((d_cost['session_num'] == 1) & (d_cost['subject_id'].isin(drop_subs)))]

# dropping non-learners
drop_subs_exc = [2, 189, 639]
d_cost = d_cost[~((d_cost['subject_id'].isin(drop_subs_exc)))]

pre_block = d_cost.loc[d_cost['phase'] == 'train', 'block'].max()
post_block = d_cost.loc[d_cost['phase'] == 'test', 'block'].min()

pre_90 = d_cost[(d_cost['block'] == pre_block) &
                 (d_cost['probe_condition'] == 90)].groupby('session_num')['acc'].mean()
post_90 = d_cost[(d_cost['block'] == post_block) &
                  (d_cost['probe_condition'] == 90)].groupby('session_num')['acc'].mean()

pre_180 = d_cost[(d_cost['block'] == pre_block) &
                  (d_cost['probe_condition'] == 180)].groupby('session_num')['acc'].mean()
post_180 = d_cost[(d_cost['block'] == post_block) &
                   (d_cost['probe_condition'] == 180)].groupby('session_num')['acc'].mean()

cost_90 = pre_90 - post_90
cost_180 = pre_180 - post_180

cost = pd.concat(
    [cost_90.rename('cost').reset_index().assign(probe_condition='90'),
     cost_180.rename('cost').reset_index().assign(probe_condition='180')],
     ignore_index=True)

# plot
fig, ax = plt.subplots(1, 1, squeeze = False)
sns.barplot(data=cost, x='session_num', y='cost', hue='probe_condition', ax=ax[0,0])
plt.tight_layout()
plt.show()

### HOW MATT WOULD DO IT (JUST THE PANDAS VERSION OF THE DATA.TABLE APPROACH IN 2020)
d_cost = dd_lab_all.copy() 

drop_subs = [134, 213, 268, 358, 482]
d_cost = d_cost[~((d_cost['session_num'] == 1) & (d_cost['subject_id'].isin(drop_subs)))]

# dropping non-learners
drop_subs_exc = [2, 189, 639]
d_cost = d_cost[~((d_cost['subject_id'].isin(drop_subs_exc)))]

d = d_cost[d_cost['block'] > 17] # equating number of train and test blocks for fair compare
dd = d.groupby(['subject_id', 'session_num', 'phase',
                           'probe_condition'])['acc'].mean().reset_index()

dd_wide = (
  dd.pivot_table(
      index=['subject_id', 'session_num', 'probe_condition'],
      columns='phase',
      values='acc',
      aggfunc='mean'
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
sns.move_legend(ax[0, 0], 'upper left', bbox_to_anchor=(1, 1))
sns.move_legend(ax[0, 1], 'upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
barplotplt.show()
plt.savefig('90vs180.png')
