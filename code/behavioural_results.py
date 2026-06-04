import numpy as np
import scipy
import pingouin
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

                # in session 4, ActiView had a syncing error and crached 30
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

                # subject 943, sesssion 6 labelled as session 2 in .csv, unsure
                # why -- correcting that here
                if fs == 'sub_943_sess_006_part_001_date_2026_04_29_data.csv':
                    df['session_num'] = 6


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

# NOTE: add congruency column to d_all -- only for 90 condition
congruent = (
    ((d_all['y'] > d_all['x']) & (d_all['y'] < (-d_all['x'] + 100))) |
    ((d_all['y'] < d_all['x']) & (d_all['y'] > (-d_all['x'] + 100)))
)

incongruent = (
    ((d_all['y'] > d_all['x']) & (d_all['y'] > (-d_all['x'] + 100))) |
    ((d_all['y'] < d_all['x']) & (d_all['y'] < (-d_all['x'] + 100)))
)

d_all['congruency'] = np.select([(d_all['probe_condition'] == 90) & congruent,
                                 (d_all['probe_condition'] == 90) & incongruent],
                                ['congruent', 'incongruent'],
                                default=pd.NA)

# # NOTE: dbm fits
# models = [
#     nll_unix,
#     nll_unix,
#     nll_uniy,
#     nll_uniy,
#     nll_glc,
#     nll_glc,
# ]
# 
# side = [0, 1, 0, 1, 0, 1, 0, 1, 2, 3]
# k = [2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
# n = block_size
# 
# model_names = [
#     'nll_unix_0',
#     'nll_unix_1',
#     'nll_uniy_0',
#     'nll_uniy_1',
#     'nll_glc_0',
#     'nll_glc_1',
# ]
# 
# d_lab_dbm = d_all[d_all['session_type'] == 'Lab'].copy()
# d_lab_train_dbm = d_lab_dbm[d_lab_dbm['phase'] == 'train'].copy()
# d_lab_test_dbm = d_lab_dbm[d_lab_dbm['phase'] == 'test'].copy()
# 
# # make sure output dir exists
# os.makedirs('../dbm_fits', exist_ok=True)
# 
# # save one combined DBM fit file for both phases
# dbm_path = '../dbm_fits/dbm_results_lab_train_test.csv'
# 
# # fitting 2 dbms per participant -- one on train, the other on test
# if not os.path.exists(dbm_path):
#     # fit train phase
#     dbm_train = (
#         d_lab_train_dbm
#         .groupby(['subject_id', 'session_num'])
#         .apply(fit_dbm, models, side, k, block_size, model_names)
#         .reset_index()
#     )
# 
#     # fit test phase
#     dbm_test = (
#         d_lab_test_dbm
#         .groupby(['subject_id', 'session_num'])
#         .apply(fit_dbm, models, side, k, block_size, model_names)
#         .reset_index()
#     )
# 
#     dbm_train['phase_dbm'] = 'train'
#     dbm_test['phase_dbm'] = 'test'
# 
#     # combine long-format rows: one row per model fit per subject/session/phase
#     dbm = pd.concat([dbm_train, dbm_test], ignore_index=True)
#     dbm.to_csv(dbm_path, index=False)
# 
# else:
#     dbm = pd.read_csv(dbm_path)
#     dbm = dbm[['subject_id', 'session_num', 'phase_dbm', 'model', 'bic', 'p']]
# 
# def assign_best_model(x):
#     model = x['model'].to_numpy()
#     bic = x['bic'].to_numpy()
#     best_model = np.unique(model[bic == bic.min()])[0]
#     x['best_model'] = best_model
#     return x
# 
# dbm = dbm.groupby(['subject_id', 'session_num', 'phase_dbm']).apply(assign_best_model, include_groups=False).reset_index()
# dbm = dbm[dbm['model'] == dbm['best_model']]
# dbm = dbm[['subject_id', 'session_num', 'phase_dbm', 'bic', 'best_model']].drop_duplicates().reset_index(drop=True)
# dbm['best_model_class'] = dbm['best_model'].str.split('_').str[1]
# dbm.loc[dbm['best_model_class'] != 'glc', 'best_model_class'] = 'rule-based'
# dbm.loc[dbm['best_model_class'] == 'glc', 'best_model_class'] = 'procedural'
# dbm['best_model_class'] = dbm['best_model_class'].astype('category') 
# 
# # print proportion of best model classes across all subjects and days
# dbm.groupby('session_num')['best_model_class'].value_counts(normalize=True)
# 
# # NOTE: plot bic across days for each model class
# fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(14, 6), sharey=True)
# 
# for i, phase_name in enumerate(['train', 'test']):
#     dplot = dbm[dbm['phase_dbm'] == phase_name].copy()
#     sns.pointplot(
#         data=dplot,
#         x='session_num',
#         y='bic',
#         hue='best_model_class',
#         errorbar=('se'),
#         ax=ax[0, i]
#     )
#     ax[0, i].set_title(f'{phase_name.capitalize()} Phase')
#     ax[0, i].set_xlabel('Session')
#     ax[0, i].set_ylabel('BIC' if i == 0 else '')
#     ax[0, i].legend(title='Model Class')
# 
# plt.tight_layout()
# plt.show()
# # plt.savefig('../figures/dbm_bic_performance_lab_train_test.png', dpi=300)

# NOTE: aggregate data for upcoming figures -- make new acc column for plot
# created new column for plotting accuracy for all days (excluding probe trials)
d_all['acc_plot'] = d_all['acc']
lab_test = (d_all['session_type'] == 'Lab') & (d_all['phase'] != 'train')
d_all.loc[lab_test, 'acc_plot'] = np.nan

# created new column for plotting reaction times for all days (excluding probe trials)
d_all['rt_plot'] = d_all['rt']
lab_test = (d_all['session_type'] == 'Lab') & (d_all['phase'] != 'train')
d_all.loc[lab_test, 'rt_plot'] = np.nan

dd_all = d_all.groupby(['subject_id', 'session_num',
                        'session_type']).agg({'acc_plot': 'mean', 'rt_plot': 'mean'}).reset_index()

pal = sns.color_palette('rocket', 6)
mid3 = pal[1:4]

# NOTE: Figure --- accuracy across all session types
fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(8, 8))
sns.pointplot(data=dd_all, x='session_num', y='acc_plot', hue='session_type',
              errorbar=('se'), palette=mid3, ax=ax[0, 0])
[x.set_xticks(np.arange(0, dd_all['session_num'].max(), 1)) for x in ax.flatten()]
ax[0 ,0].set_title('Mean Accuracy Across Days per Session Type', fontsize=16)
ax[0, 0].set_xlabel('Day')
ax[0, 0].set_ylabel('Accuracy (Proportion Correct)')
ax[0, 0].legend(title='Session Type', loc='lower right')
plt.show()
#plt.savefig('../figures/accuracy_across_days.png', dpi=300)
#plt.close()

# NOTE: Figure --- reaction time across all session types
fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(8, 8))
sns.pointplot(data=dd_all, x='session_num', y='rt_plot', hue='session_type',
              errorbar=('se'), palette=mid3, ax=ax[0, 0])
[x.set_xticks(np.arange(0, dd_all['session_num'].max(), 1)) for x in ax.flatten()]
ax[0 ,0].set_title('Mean Reaction Times Across Days per Session Type', fontsize=16)
ax[0, 0].set_xlabel('Day')
ax[0, 0].set_ylabel('Reaction Time (ms)')
ax[0, 0].legend(title='Session Type', loc='lower left')
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

# NOTE: Figure -- comparing last at home day and last lab day to dual-task day
# using dd_all from above 
d_dtf = dd_all[dd_all['session_num'].isin([20, 21, 22])].copy()

# change the day column to categorical for plotting with names "Last Training
# Day" and "Dual-Task Day"
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

# NOTE: Stats -- anova across all days: does accuracy improve across days?
d_anova = d_all[~d_all['session_num'].isin(d_all[d_all['session_num']==22])]

res_anova = pg.rm_anova(data=d_anova,
                        dv='acc_plot',
                        within='session_num',
                        subject='subject_id',
                        correction=True)

# NOTE: Stats -- dual-task: is there a difference in accuracy?
res = pg.ttest(x=d_dtf[d_dtf['session_num'] == 'Last Training Day']['acc_plot'],
               y=d_dtf[d_dtf['session_num'] == 'Dual-Task Day']['acc_plot'],
               alternative='two-sided',
               paired=True)

res = pg.ttest(x=d_dtf[d_dtf['session_num'] == 'Lab']['acc_plot'],
               y=d_dtf[d_dtf['session_num'] == 'Dual-Task Day']['acc_plot'],
               alternative='two-sided',
               paired=True)

# NOTE: Figure -- calculating + plotting cost for accuracy and reaction time
d_cost = d_all.copy() 

drop_subs = [134, 213, 268, 358, 482]
d_cost = d_cost[~((d_cost['session_num'] == 1) & (d_cost['subject_id'].isin(drop_subs)))]

# dropping non-learners
drop_subs_exc = [2, 189, 639]
d_cost = d_cost[~((d_cost['subject_id'].isin(drop_subs_exc)))]

d = d_cost[d_cost['block'] > 17] # equating number of train and test blocks for fair compare
dd = d.groupby(['subject_id', 'session_num', 'phase',
                           'probe_condition'])[['acc', 'rt']].mean().reset_index()

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
              dodge=True
)
plt.show()

# plot accuracy cost for each subject 
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

# NOTE: Figure -- calculating + plotting conguency accurancy and reaction times (90 only)
d_congruency = d_all.copy() 

drop_subs = [134, 213, 268, 358, 482]
d_congruency = d_congruency[~((d_congruency['session_num'] == 1) & (d_congruency['subject_id'].isin(drop_subs)))]

# dropping non-learners
drop_subs_exc = [2, 189, 639]
d_congruency = d_congruency[~((d_congruency['subject_id'].isin(drop_subs_exc)))]

d = d_congruency[d_congruency['block'] > 17] # equating number of train and test blocks for fair compare
dd = d.groupby(['subject_id', 'session_num', 'phase', 'congruency',
                           'probe_condition'])[['acc', 'rt']].mean().reset_index()

# accuracy
dd_wide_acc = (
  dd.pivot_table(
      index=['subject_id', 'session_num', 'probe_condition', 'congruency'],
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
sns.pointplot(data=dd,
              x='session_num',
              y='acc',
              hue='congruency',
              errorbar='se',
              linestyle='phase',
              dodge=True
)
plt.show()

sns.catplot(
    data=dd,
    kind="point",
    x="session_num",
    y="acc",
    hue="congruency",     # congruent vs incongruent
    col="phase",          # train vs test (separate panels)
    errorbar=("se"),
    dodge=True,
    height=4,
    aspect=1.2
)
plt.show()

# reaction times
dd_wide_rt = (
  dd.pivot_table(
      index=['subject_id', 'session_num', 'probe_condition', 'congruency'],
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
              markers='congruency'
              errorbar='se',
              linestyle='none',
              dodge=True
)
plt.show()

sns.catplot(
    data=dd,
    kind="point",
    x="session_num",
    y="rt",
    hue="congruency",     # congruent vs incongruent
    col="phase",          # train vs test (separate panels)
    errorbar=("se"),
    dodge=True,
    height=4,
    aspect=1.2
)
plt.show()
