import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

dir_data = '../at_home_data'
dir_data_lab = '../behavioural_data'

df_lab_rec = []
df_train_rec = []
df_dt_rec = []

sns.set_palette('rocket', 2)

for fd in os.listdir(dir_data_lab):
    if fd in exclude_subs:
        continue
    dir_data_lab_fd = os.path.join(dir_data_lab, fd)
    if os.path.isdir(dir_data_lab_fd):
        for fs in os.listdir(dir_data_lab_fd):
            f_full_path = os.path.join(dir_data_lab_fd, fs)
            if os.path.isfile(f_full_path) and fs.endswith('.csv'):

                # in session 4, ActiView had a syncing error and crached 30
                # trials in with participant 875, restarted experiment clean --
                # removing extra data file
                if fs not in ['sub_875_sess_004_part_001_date_2026_04_24_data (1).csv'
                              ]:

                    df = pd.read_csv(f_full_path)
                    df['f_name'] = fs
                    df_lab_rec.append(df)

# not reading in cp task
for fd in os.listdir(dir_data):
    if fd in exclude_subs:
        continue
    dir_data_fd = os.path.join(dir_data, fd)
    if os.path.isdir(dir_data_fd):
        for fs in os.listdir(dir_data_fd):
            f_full_path = os.path.join(dir_data_fd, fs)
            if os.path.isfile(f_full_path) and 'task_cp_' not in fs:
                
                df = pd.read_csv(f_full_path)
                df['f_name'] = fs

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

# NOTE: is this too much work? i don't want to just remove them because of my
# muck up *sad face* 
# in session 1, sub_875 completed 10 trian trials and 50 probe trials (part 1),
# then completed 540 train and 100 probe (part 2) -- adding 10 train trials from
# part 1 to part 2
f1 = 'sub_875_sess_001_part_001_date_2026_04_03_data (1).csv'
f2 = 'sub_875_sess_001_part_002_date_2026_04_03_data.csv'

p1_875 = d_lab[d_lab["f_name"] == f1]
p2_875 = d_lab[d_lab["f_name"] == f2]

p875 = pd.concat([p1_875[p1_875["phase"] == "train"].head(10), p2_875], ignore_index=True)

d_lab = d_lab[(d_lab["f_name"] != f1) & (d_lab["f_name"] != f2)]
d_lab = pd.concat([d_lab, p875], ignore_index=True)

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

d_dt = d_dt.sort_values(['subject_id', 'session_num', 'session_part',
                         'trial']).reset_index(drop=True)
d_dt['acc'] = (d_dt['cat'] == d_dt['resp']).astype(int)
d_dt['trial'] = d_dt.groupby(['subject_id', 'session_num']).cumcount()
d_dt['n_trials'] = d_dt.groupby(['subject_id', 'session_num'])['trial'].transform('count')
d_dt['block'] = d_dt.groupby(['subject_id', 'session_num'])['trial'].transform(lambda x: x // block_size)
d_dt['session_num'] = d_dt['session_num'].map({17: 22})
d_dt['session_type'] = 'Dual-Task'

# NOTE: create a numpy array of the intersection of subjects across all dataframes
all_subs = np.unique(np.concatenate([d_home.subject_id.unique(),
                                     d_dt.subject_id.unique(),
                                     d_lab.subject_id.unique()]))

subs_to_keep = np.intersect1d(all_subs, d_home.subject_id.unique())
subs_to_keep = np.intersect1d(subs_to_keep, d_dt.subject_id.unique())
subs_to_keep = np.intersect1d(subs_to_keep, d_lab.subject_id.unique())

# merge all dataframes inserting np.nan into columns that don't exist in a particular dataframe
d_all = pd.concat([d_home, d_dt, d_lab], ignore_index=True, sort=False)
d_all['session_num'] = d_all.groupby('subject_id')['session_num'].rank(method='dense').astype(int)

# NOTE: exclude subjects not in all three dataframes (i.e., who did not complete
# the task correctly)
d_all = d_all[d_all['subject_id'].isin(subs_to_keep)].reset_index(drop=True)

# compute average performance on lab days (train trials)
lab_train = (d_all['session_type'] == 'Lab') & (d_all['phase'] == 'train')
d_all['acc_lab_total'] = np.nan
d_all.loc[lab_train, 'acc_lab_total'] = (d_all.loc[lab_train].groupby(['subject_id', 'session_num'])['acc'].transform('mean')) 

# NOTE: exclude subjects with average accuracy < 75% on day 6 (lab day 2)
d_all = d_all[d_all.groupby('subject_id')['acc_lab_total']
                     .transform(lambda s: s[d_all.loc[s.index, 'session_num'].eq(6)].max())
                     .ge(0.75)
            ].reset_index(drop=True)

# NOTE: compute Stroop accuracy and exlcude subjects with accuracy < 80%
d_all['acc_stroop'] = np.nan
d_all.loc[d_all['fb_ns'].notna(), 'acc_stroop'] = (d_all['fb_ns'] == 'Correct').astype(int)
d_all['acc_stroop_mean'] = d_all.groupby('subject_id')['acc_stroop'].transform(lambda x: np.nanmean(x))
d_all = d_all[d_all.groupby('subject_id')['acc_stroop_mean'].transform('max').ge(0.8)
              ].reset_index(drop=True) 

# NOTE: aggregate data for upcoming figures -- make new acc column for plot
# excluding probe trials on lab days
d_all['acc_plot'] = d_all['acc']
lab_test = (d_all['session_type'] == 'Lab') & (d_all['phase'] != 'train')
d_all.loc[lab_test, 'acc_plot'] = np.nan

dd_all = d_all.groupby(['subject_id', 'session_num',
                        'session_type']).agg({'acc_plot': 'mean', 'rt': 'mean'}).reset_index()

# NOTE: Figure --- all session types
fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(8, 8))
sns.pointplot(data=dd_all, x='session_num', y='acc_plot', hue='session_type', errorbar=('se'), ax=ax[0, 0])
[x.set_xticks(np.arange(0, dd_all['session_num'].max(), 1)) for x in ax.flatten()]
ax[0 ,0].set_title('Mean Accuracy Across Days per Session Type', fontsize=16)
ax[0, 0].set_xlabel('Day')
ax[0, 0].set_ylabel('Accuracy (Proportion Correct)')
ax[0, 0].legend(title='Session Type', loc='lower right')
plt.show()
#plt.savefig('../figures/training_performance_days.png', dpi=300)
#plt.close()

# NOTE: Figure -- comparing last at home day and last lab day to dual-task day
d_dtf = dd_all[dd_all['session_num'].isin([20, 21, 22])].copy()

# change the day column to categorical for plotting with names "Last Training Day" and "Dual-Task Day"
d_dtf['session_num'] = d_dtf['session_num'].map({20: 'Last Training Day', 21: 'Lab Day', 22: 'Dual-Task Day'})

# plot point range plot comparing the last day of training and lab to dual-task day
fig, ax = plt.subplots(2, 1, squeeze=False, figsize=(5, 8))
sns.pointplot(data=d_dtf, x='session_num', y='acc_plot', errorbar=('se'), ax=ax[0, 0])
sns.pointplot(data=d_dtf, x='session_num', y='rt', errorbar=('se'), ax=ax[1, 0])
ax[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))
ax[0, 0].set_xlabel('')
ax[0, 0].set_ylabel('Accuracy (proportion correct)')
ax[1, 0].set_xlabel('')
ax[1, 0].set_ylabel('Reaction Time (ms)')
plt.tight_layout()
plt.show()
# plt.savefig('../figures/dual_task_performance.png', dpi=300)
# plt.close()

# plot point range plot comparing the last day of training and lab to dual-task day (acc only)
fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(6, 6))
sns.pointplot(data=d_dtf, x='session_num', y='acc_plot', errorbar=('se'), ax=ax[0, 0])
ax[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))
ax[0, 0].set_xlabel('')
ax[0, 0].set_ylabel('Accuracy (proportion correct)')
ax[0, 0].set_title('Mean Accuracy: Last Training and Lab Day vs. Dual Task', fontsize=12)
plt.tight_layout()
plt.show()
# plt.savefig('../figures/dual_task_performance_acc.png', dpi=300)
# plt.close()

# NOTE: calculating and plotting cost
d_cost = d_all.copy() 

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


fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(6, 6))
sns.pointplot(data=dd_wide,
              x = 'session_num',
              y = 'diff_score',
              hue = 'probe_condition',
              errorbar='se',
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
plt.show()
plt.savefig('90vs180.png')
