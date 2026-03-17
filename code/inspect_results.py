from imports import *
from util_func_dbm import *

if __name__ == '__main__':

    # NOTE: Init figure style
    sns.set_palette("rocket", n_colors=4)
    plt.rc('axes', labelsize=14)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)

    dir_data = "../data"
    dir_data_lab_beh = "../data_lab_behave"

    df_train_rec = []  # training days
    df_dt_rec = []  # dual task days
    df_lab_rec = []  # lab days

    for fd in os.listdir(dir_data):
        dir_data_fd = os.path.join(dir_data, fd)
        if os.path.isdir(dir_data_fd):
            for fs in os.listdir(dir_data_fd):
                f_full_path = os.path.join(dir_data_fd, fs)
                if os.path.isfile(f_full_path):

                    # NOTE: Day Exclusion List
                    if fs not in []:  

                        df = pd.read_csv(f_full_path)
                        df['f_name'] = fs

                        day = df['day'].unique()

                        # training days
                        if ~np.isin(day, [22, 23, 24]):
                            df_train_rec.append(df)

                        # dual-task days
                        if day == 22:
                            df_dt_rec.append(df)

    for fd in os.listdir(dir_data_lab_beh):
        f_df = os.path.join(dir_data_lab_beh, fd)
        if os.path.isfile(f_df) and fd != '.DS_Store':
            df = pd.read_csv(f_df)
            df_lab_rec.append(df)

    block_size = 25

    d = pd.concat(df_train_rec, ignore_index=True)
    d.sort_values(by=['subject', 'day', 'trial'], inplace=True)
    d['acc'] = (d['cat'] == d['resp']).astype(int)
    d['day'] = d.groupby('subject')['day'].rank(method='dense').astype(int)
    d['trial'] = d.groupby(['subject']).cumcount()
    d['n_trials'] = d.groupby(['subject', 'day'])['trial'].transform('count')
    d['block'] = d.groupby(['subject', 'day'
                            ])['trial'].transform(lambda x: x // block_size)
    d['session_type'] = 'Training at home'

    d_dt = pd.concat(df_dt_rec, ignore_index=True)
    d_dt.sort_values(by=['subject', 'day', 'trial'], inplace=True)
    d_dt['acc'] = (d_dt['cat'] == d_dt['resp']).astype(int)
    d_dt['day'] = d_dt.groupby('subject')['day'].rank(
        method='dense').astype(int)
    d_dt['day'] = d_dt['day'].map({1: 22})
    d_dt['trial'] = d_dt.groupby(['subject']).cumcount()
    d_dt['n_trials'] = d_dt.groupby(['subject',
                                     'day'])['trial'].transform('count')
    d_dt['session_type'] = 'Dual-Task at home'

    d_lab = pd.concat(df_lab_rec, ignore_index=True)
    d_lab['acc'] = (d_lab['cat'] == d_lab['resp']).astype(int)
    d_lab['day'] = d_lab.groupby('subject')['day'].rank(
        method='dense').astype(int)
    d_lab['day'] = d_lab['day'].map({1: 0.5, 2: 4.5, 3: 8.5, 4: 12.5, 5: 21})
    d_lab['trial'] = d_lab.groupby(['subject']).cumcount()
    d_lab['n_trials'] = d_lab.groupby(['subject',
                                       'day'])['trial'].transform('count')
    d_lab['block'] = d_lab.groupby(
        ['subject', 'day'])['trial'].transform(lambda x: x // block_size)
    d_lab['session_type'] = 'Training in the Lab'

    # NOTE: create a numpy array of the intersection of subjects across all dataframes
    all_subs = np.unique(
        np.concatenate([
            d.subject.unique(),
            d_dt.subject.unique(),
            d_lab.subject.unique()
        ]))

    subs_to_keep = np.intersect1d(all_subs, d.subject.unique())
    subs_to_keep = np.intersect1d(subs_to_keep, d_dt.subject.unique())
    subs_to_keep = np.intersect1d(subs_to_keep, d_lab.subject.unique())

    # merge all dataframes inserting np.nan into columns that don't exist in a particular dataframe
    d_all = pd.concat([d, d_dt,  d_lab], ignore_index=True, sort=False)
    d_all['day'] = d_all.groupby('subject')['day'].rank(
        method='dense').astype(int)

    # exclude subjects not in all three dataframes
    d_all = d_all[d_all['subject'].isin(subs_to_keep)].reset_index(drop=True)

    # NOTE: compute Stroop accuracy and exlcude subjects with accuracy < 80%
    d_all['acc_stroop'] = np.nan
    d_all.loc[d_all['ns_correct_side'].notna(), 'acc_stroop'] = (
        d_all['ns_correct_side'] == d_all['ns_resp']).astype(int)
    d_all['acc_stroop_mean'] = d_all.groupby(
        'subject')['acc_stroop'].transform(lambda x: np.nanmean(x))
    d_all = d_all[d_all['acc_stroop_mean'] >= 0.8].reset_index(drop=True)

    # NOTE: primary exclusion criteria for remaining subjects will be deciion bound fits
    #       Fit DBM here
    models = [
        nll_unix,
        nll_unix,
        nll_uniy,
        nll_uniy,
        nll_glc,
        nll_glc,
    ]
    side = [0, 1, 0, 1, 0, 1, 0, 1, 2, 3]
    k = [2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
    n = block_size
    model_names = [
        "nll_unix_0",
        "nll_unix_1",
        "nll_uniy_0",
        "nll_uniy_1",
        "nll_glc_0",
        "nll_glc_1",
    ]

    if not os.path.exists("../dbm_fits/dbm_results.csv"):
        dbm = (d.groupby(["subject", "day"]).apply(fit_dbm, models, side, k, n,
                                                   model_names).reset_index())
        dbm.to_csv("../dbm_fits/dbm_results.csv")
    else:
        dbm = pd.read_csv("../dbm_fits/dbm_results.csv")
        dbm = dbm[["subject", "day", "model", "bic", "p"]]

    def assign_best_model(x):
        model = x["model"].to_numpy()
        bic = x["bic"].to_numpy()
        best_model = np.unique(model[bic == bic.min()])[0]
        x["best_model"] = best_model
        return x

    dbm = dbm.groupby(["subject",
                       "day"]).apply(assign_best_model,
                                     include_groups=False).reset_index()
    dbm = dbm[dbm["model"] == dbm["best_model"]]
    dbm = dbm[["subject", "day", "bic", "best_model"]]
    dbm = dbm.drop_duplicates().reset_index(drop=True)
    dbm["best_model_class"] = dbm["best_model"].str.split("_").str[1]
    dbm.loc[dbm["best_model_class"] != "glc",
            "best_model_class"] = "rule-based"
    dbm.loc[dbm["best_model_class"] == "glc",
            "best_model_class"] = "procedural"
    dbm["best_model_class"] = dbm["best_model_class"].astype("category")
    dbm = dbm.reset_index(drop=True)

    # print proportion of best model classes across all subjects and days
    dbm.groupby('day')['best_model_class'].value_counts(normalize=True)

    # NOTE: plot bic across days for each model class
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10, 7))
    sns.pointplot(data=dbm,
                  x='day',
                  y='bic',
                  hue='best_model_class',
                  errorbar=('se'),
                  ax=ax[0, 0])
    ax[0, 0].set_xlabel('Day')
    ax[0, 0].set_ylabel('BIC')
    plt.tight_layout()
    plt.savefig('../figures/dbm_bic_performance.png', dpi=300)

    # NOTE: aggregate data for upcoming figures
    d_all = d_all[d_all['rt'] <= 3000]
    dd_all = d_all.groupby(['subject', 'day', 'session_type']).agg({
        'acc': 'mean',
        'rt': 'mean'
    }).reset_index()

    # NOTE: Figure --- all session types
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(8, 5))
    sns.pointplot(data=dd_all,
                  x='day',
                  y='acc',
                  hue='session_type',
                  errorbar=('se'),
                  ax=ax[0, 0])
    [
        x.set_xticks(np.arange(0, dd_all['day'].max() + 2, 1))
        for x in ax.flatten()
    ]
    ax[0, 0].set_xlabel('Day', fontsize=16)
    ax[0, 0].set_ylabel('Proportion correct', fontsize=16)
    ax[0, 0].legend(title='')
    plt.savefig('../figures/training_performance_days.png', dpi=300)
    plt.close()

    # NOTE: Figure --- all session types RT
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(8, 5))
    sns.pointplot(data=dd_all,
                  x='day',
                  y='rt',
                  hue='session_type',
                  errorbar=('se'),
                  ax=ax[0, 0])
    [
        x.set_xticks(np.arange(0, dd_all['day'].max() + 2, 1))
        for x in ax.flatten()
    ]
    ax[0, 0].set_xlabel('Day', fontsize=16)
    ax[0, 0].set_ylabel('Reaction Time', fontsize=16)
    ax[0, 0].legend(title='')
    plt.savefig('../figures/training_performance_days_rt.png', dpi=300)
    plt.close()

    # NOTE: dual-task figures
    # prepare a data frame comparing last day of training to dual-task day
    d_dtf = dd_all[dd_all['day'].isin([20, 22])].copy()

    # change the day column to categorical for plotting with names "Last Training Day" and "Dual-Task Day"
    d_dtf['day'] = d_dtf['day'].map({
        20: 'Last Training Day',
        22: 'Dual-Task Day'
    })

    # plot point range plot comparing the last day of training to dual-task day
    fig, ax = plt.subplots(2, 1, squeeze=False, figsize=(5, 8))
    sns.pointplot(data=d_dtf, x='day', y='acc', errorbar=('se'), ax=ax[0, 0])
    sns.pointplot(data=d_dtf, x='day', y='rt', errorbar=('se'), ax=ax[1, 0])
    ax[0, 0].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))
    ax[0, 0].set_xlabel('')
    ax[0, 0].set_ylabel('Accuracy (proportion correct)')
    ax[1, 0].set_xlabel('')
    ax[1, 0].set_ylabel('Reaction Time (ms)')
    plt.tight_layout()
    plt.savefig('../figures/dual_task_performance.png', dpi=300)
    plt.close()

    # NOTE: dual-task stats
    res = pg.ttest(x=d_dtf[d_dtf['day'] == 'Last Training Day']['acc'],
                   y=d_dtf[d_dtf['day'] == 'Dual-Task Day']['acc'],
                   alternative='greater',
                   paired=True)

