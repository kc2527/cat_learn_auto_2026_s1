import numpy as np
import scipy
import pingouin as pg
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from util_func_dbm import *

dir_data = '../at_home_data'
dir_data_lab = '../behavioural_data'

df_lab_rec = []
df_train_rec = []
df_dt_rec = []

sns.set_palette('rocket')

for fd in os.listdir(dir_data_lab):
    dir_data_lab_fd = os.path.join(dir_data_lab, fd)
    if os.path.isdir(dir_data_lab_fd):
        for fs in os.listdir(dir_data_lab_fd):
            f_full_path = os.path.join(dir_data_lab_fd, fs)
            if os.path.isfile(f_full_path) and fs.endswith('.csv'):

                # in session 4, ActiView had a syncing error and crashed 30
                # trials in with participant 875, restarted experiment clean --
                # removing extra data file
                if fs not in ['sub_875_sess_004_part_001_date_2026_04_24_data (1).csv'
                              ]:

                    df = pd.read_csv(f_full_path)

                    # subject 594 missed lab day 4 due to illness, made up
                    # session at home under id 444, changing id to 594 and
                    # session_num to 4
                    if fs == 'sub_444_sess_001_part_001_date_2026_05_24_data.csv':
                        df['subject_id'] = 594
                        df['session_num'] = 4

                    # subject 594 completed sessions across 2 lab computers
                    # throughout the experiment so relabelling 'session 3' as
                    # 'session 5'
                    if fs == 'sub_594_sess_003_part_001_date_2026_05_29_data.csv':
                        df['session_num'] = 5

                    df['f_name'] = fs
                    df_lab_rec.append(df)

# not reading in cp task
for fd in os.listdir(dir_data):
    dir_data_fd = os.path.join(dir_data, fd)
    if os.path.isdir(dir_data_fd):
        for fs in os.listdir(dir_data_fd):
            f_full_path = os.path.join(dir_data_fd, fs)
            if os.path.isfile(f_full_path) and 'task_cp_' not in fs:
                
                df = pd.read_csv(f_full_path)
                df['f_name'] = fs

                # subject 943: made an exact copy of session 2 and relabelled it
                # to session 6 -- session excluded
                if fs == 'sub_943_sess_006_part_001_date_2026_04_29_data.csv':
                    continue

                session = df['session_num'].unique()

                # training days
                if ~np.isin(session, 17):
                    df_train_rec.append(df)

                # dual task day
                if session == 17:
                    df_dt_rec.append(df)

d_lab = pd.concat(df_lab_rec, ignore_index=True)
d_home = pd.concat(df_train_rec, ignore_index=True)
d_dt = pd.concat(df_dt_rec, ignore_index=True)
  
# in session 1, sub_875 completed 10 train trials and 50 probe trials (part 1),
# then completed 540 train and 100 probe (part 2) -- adding 10 train trials from
# part 1 to part 2
f1 = 'sub_875_sess_001_part_001_date_2026_04_03_data (1).csv'
f2 = 'sub_875_sess_001_part_002_date_2026_04_03_data.csv'

p1_875 = d_lab[d_lab['f_name'] == f1]
p2_875 = d_lab[d_lab['f_name'] == f2]

p875 = pd.concat([p1_875[p1_875['phase'] == 'train'].head(10), p2_875], ignore_index=True)

d_lab = d_lab[(d_lab['f_name'] != f1) & (d_lab['f_name'] != f2)]
d_lab = pd.concat([d_lab, p875], ignore_index=True)

# NOTE: create dfs
block_size = 25

d_lab = d_lab.sort_values(['subject_id', 'session_num', 'session_part',
                             'trial']).reset_index(drop=True)
d_lab['acc'] = (d_lab['cat'] == d_lab['resp']).astype(int)
d_lab['trial'] = d_lab.groupby(['subject_id', 'session_num']).cumcount()
d_lab['n_trials'] = d_lab.groupby(['subject_id', 'session_num'])['trial'].transform('count')
d_lab['block'] = d_lab.groupby(['subject_id', 'session_num'])['trial'].transform(lambda x: x // block_size)
d_lab['session_num'] = d_lab['session_num'].map({1: 0.5, 2:4.5, 3:8.5, 4:12.5, 5:21})
d_lab['session_type'] = 'Lab'

d_home = d_home.sort_values(['subject_id', 'session_num', 'session_part',
                               'trial']).reset_index(drop=True)
d_home['acc'] = (d_home['cat'] == d_home['resp']).astype(int)
d_home['trial'] = d_home.groupby(['subject_id', 'session_num']).cumcount()
d_home['n_trials'] = d_home.groupby(['subject_id', 'session_num'])['trial'].transform('count')
d_home['block'] = d_home.groupby(['subject_id', 'session_num'])['trial'].transform(lambda x: x // block_size)
d_home['session_type'] = 'Training'

# merge all dataframes inserting np.nan into columns that don't exist in a particular dataframe
d_all = pd.concat([d_home, d_lab], ignore_index=True, sort=False)
d_all['session_num'] = d_all.groupby('subject_id')['session_num'].rank(method='dense').astype(int)

# excluding known non-learners
d_all = d_all.loc[~d_all['subject_id'].isin([2, 189, 639])].copy()

# removing first session probes for 134, 213, 268, 358, 492
d_all = d_all.loc[~(d_all['session_num'].eq(1)
                  & d_all['phase'].eq('test')
                  & d_all['probe_condition'].eq(90)
                  & ~d_all['subject_id'].isin([77, 303]))].copy()

# plotting % of trials kept at different rts
fig, ax = plt.subplots(figsize=(8, 5))

sns.ecdfplot(data=d_all, x="rt", ax=ax)
ax.set_xscale("log")
ax.set_xlabel("Reaction time (ms; log scale)")
ax.set_ylabel("Cumulative proportion of trials")

for cutoff in [150, 200, 300, 3000, 5000, 10000]:
    ax.axvline(cutoff, linestyle="--", color="black", alpha=0.6)

plt.show()

# cut off at 5000 retains 99% of the data
# no. of trials before cut off 
before_cutoff = len(d_all)

# aggregate data for upcoming figures
d_all = d_all.loc[(d_all["rt"] >= 200) & (d_all["rt"] <= 5000)].copy()
dd_all = (d_all.groupby(['subject_id', 'session_num', 'session_type', 'phase', 'probe_condition'],
          as_index=False)[['acc', 'rt']].mean())

# no. trials removed 
after_cutoff = len(d_all)
trials_lost = before_cutoff - after_cutoff

# % of trials dropped after rt exclusion
dropped = (trials_lost / before_cutoff) * 100

# NOTE: aggregate data for upcoming figures 
pal = sns.color_palette('rocket', 6)
mid3 = pal[1:4]
back2 = pal[4:6]

# NOTE: Figure --- accuracy across all session types
fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(8, 8))

sns.pointplot(data=dd_all[dd_all['phase']=='train'], 
              x='session_num', 
              y='acc',
              hue='session_type', 
              errorbar=('se'), 
              palette=mid3,
              ax=ax[0, 0])

sns.pointplot(data=dd_all[dd_all['phase']=='test'], 
              x='session_num', 
              y='acc',
              hue='probe_condition', 
              errorbar=('se'), 
              dodge=0.25,
              palette=back2,
              ax=ax[0, 0])

[x.set_xticks(np.arange(0, dd_all['session_num'].max(), 1)) for x in ax.flatten()]
ax[0 ,0].set_title('Mean Accuracy Across Days per Session Type', fontsize=16)
ax[0, 0].set_xlabel('Day')
ax[0, 0].set_ylabel('Accuracy (Proportion Correct)')
ax[0, 0].legend(loc='upper left')
plt.show()

#plt.savefig('../figures/accuracy_across_days.png', dpi=300)
#plt.close()

# NOTE: Figure --- reaction time across all session types
fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(8, 8))

sns.pointplot(data=dd_all[dd_all['phase']=='train'],
              x='session_num', 
              y='rt', 
              hue='session_type',
              errorbar=('se'), 
              palette=mid3, 
              ax=ax[0, 0])

sns.pointplot(data=dd_all[dd_all['phase']=='test'], 
              x='session_num', 
              y='rt',
              hue='probe_condition', 
              errorbar=('se'), 
              dodge=0.25,
              palette=back2,
              ax=ax[0, 0])

[x.set_xticks(np.arange(0, dd_all['session_num'].max(), 1)) for x in ax.flatten()]
ax[0 ,0].set_title('Mean Reaction Times Across Days per Session Type', fontsize=16)
ax[0, 0].set_xlabel('Day')
ax[0, 0].set_ylabel('Reaction Time (ms)')
ax[0, 0].legend(loc='upper right')
plt.show()
#plt.savefig('../figures/rts_across_days.png', dpi=300)
#plt.close()

# NOTE: Figure -- accuracy across all lab days (blocks)
d_lab_all = d_all[d_all['session_type'] == 'Lab'].copy()
d_lab_all['block_cont'] = ((d_lab_all['session_num'] - 1) * 26) + d_lab_all['block'] + 1

fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(8,8))
sns.pointplot(data=d_lab_all, x='block_cont', y='acc', hue='probe_condition',
              errorbar='se', scale=0.75, ax=ax[0,0])
plt.tight_layout()
plt.show()

# NOTE: Stats -- anova across all days: does accuracy improve across days?
d_anova = d_all[~d_all['session_num'].isin(d_all[d_all['session_num']==22])]

res_anova = pg.rm_anova(data=d_anova,
                        dv='acc',
                        within='session_num',
                        subject='subject_id',
                        correction=True)

print('ANOVA \n', res_anova)

# NOTE: Figure -- calculating + plotting cost for accuracy and reaction time
# calibrated with block size of 25
test_start = (d_all['block'].where(d_all['phase'].eq('test'))
              .groupby([d_all['subject_id'], d_all['session_num']]).transform('min'))

keep = (d_all['phase'].eq('train') & d_all['block'].sub(test_start).isin([-4, -3, -2, -1]) |
        d_all['phase'].eq('test') & d_all['block'].sub(test_start).isin([0, 1, 2, 3]))

d_cost = d_all.loc[keep].copy()

dd = (d_cost.groupby(['subject_id', 'session_num', 'phase', 'probe_condition'])
      [['acc', 'rt']].mean().reset_index())

# accuracy
dd_wide_acc = (
  dd.pivot_table(
      index=['subject_id', 'session_num', 'probe_condition'],
      columns='phase',
      values='acc',
      aggfunc='mean'
  )
  .reset_index()
)

dd_wide_acc['diff_score'] = dd_wide_acc['train'] - dd_wide_acc['test']
dd_wide_acc['probe_condition'] = dd_wide_acc['probe_condition'].astype('category')
dd_wide_acc['subject_id'] = dd_wide_acc['subject_id'].astype('category')

# plot accuracy cost
fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(6, 6))
sns.pointplot(data=dd_wide_acc,
              x='session_num',
              y='diff_score',
              hue='probe_condition',
              errorbar='se',
              linestyle='none',
              palette=mid3,
              dodge=True
)
plt.show()

# reaction times
dd_wide_rt = (
  dd.pivot_table(
      index=['subject_id', 'session_num', 'probe_condition'],
      columns='phase',
      values='rt',
      aggfunc='mean'
  )
  .reset_index()
)

# making it test - train to make +ve values
dd_wide_rt['diff_score'] = dd_wide_rt['test'] - dd_wide_rt['train']
dd_wide_rt['probe_condition'] = dd_wide_rt['probe_condition'].astype('category')
dd_wide_rt['subject_id'] = dd_wide_rt['subject_id'].astype('category')

# plot reaction time cost
fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(6, 6))
sns.pointplot(data=dd_wide_rt,
              x='session_num',
              y='diff_score',
              hue='probe_condition',
              errorbar='se',
              linestyle='none',
              palette=mid3,
              dodge=True
)
plt.show()

# plot accuracy cost for each subject 
fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(10, 5))
sns.lineplot(data=dd_wide_acc[dd_wide_acc['probe_condition'] == 90],
             x = 'session_num',
             y = 'diff_score',
             hue = 'subject_id',
             ax=ax[0, 0]
)
sns.lineplot(data=dd_wide_acc[dd_wide_acc['probe_condition'] == 180],
             x = 'session_num',
             y = 'diff_score',
             hue = 'subject_id',
             ax=ax[0, 1]
)
sns.move_legend(ax[0, 0], 'upper left', bbox_to_anchor=(1, 1))
sns.move_legend(ax[0, 1], 'upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()
