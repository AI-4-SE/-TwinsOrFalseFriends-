#!/bin/env python3
import os
import sys
import scipy
import numpy as np
import pandas as pd
# pd.options.mode.chained_assignment = None  # default='warn'
import sklearn.cluster
import sklearn.mixture
import seaborn as sns
import matplotlib.pyplot as plt


paper_systems = ['brotli', 'jump3r', 'x264', 'kanzi', 'HSQLDB', '7z']


def plot_corr(df):
    sns.scatterplot(data=df, x='performance', y='energy')
    plt.show()


def window_sections(df, window_pct=0.05, stepsize=0.5, mode='data', n_windows=30):
    # mode: [data|window]
    correlation_frames = []
    if mode == 'window':
        min_perf = min(df.performance)
        max_perf = max(df.performance)
        value_range = max_perf - min_perf

        window_width = value_range * window_pct
        current_perf = min_perf

        while current_perf < max_perf - (window_width * stepsize) - 1:
            window_df = df[(df['performance'] >= current_perf) & (df['performance'] <= current_perf + window_width)]

            if len(window_df) > 2:
                corr, p_value = scipy.stats.pearsonr(window_df['performance'].to_numpy(),
                                                     window_df['energy'].to_numpy())
                correlation_frames.append([corr])

            current_perf = current_perf + window_width * stepsize
    elif mode == 'data':
        df.sort_values(by=['performance'])
        current_window = 0.0
        total_len = len(df)
        stepsize = total_len / n_windows
        while current_window < n_windows:
            window_df = df.iloc[round(stepsize * current_window): round(stepsize * current_window + stepsize)]
            corr, p_value = scipy.stats.pearsonr(window_df['performance'].to_numpy(),
                                                 window_df['energy'].to_numpy())
            correlation_frames.append([corr])
            current_window += 1

    return correlation_frames


def corr_cluster(df, identifier='cluster'):
    correlation_cluster = []

    for key, tmp in df.groupby([identifier]):
        if len(tmp) < 2:
            continue

        corr, p_value = scipy.stats.pearsonr(tmp['performance'], tmp['energy'])
        correlation_cluster.append([corr])
    return correlation_cluster


def calc_lin_err(x, y, slope, intercept):
    return abs(y - (intercept + (slope * x)))


def cluster_dbscan(df, stepsize=0.01):
    x = df['performance'].to_numpy()
    eps = stepsize * (x.max() - x.min())
    cluster = sklearn.cluster.DBSCAN(eps=eps, min_samples=len(df) / 200).fit(x.reshape(-1, 1))
    df['DBSCAN'] = cluster.labels_
    print('number clusters DBSCAN', len(df.DBSCAN.unique()))
    return df


def cluster_gmm(df):
    x = df['performance'].to_numpy()
    y = df['energy'].to_numpy()
    # data = np.stack([x,y]).T
    data = x.reshape(-1, 1)
    model = sklearn.mixture.BayesianGaussianMixture(n_components=20, max_iter=1000, init_params='random').fit(data)
    y_pred = model.predict(data)
    df['GMM'] = y_pred
    print('number clusters GMM', len(df.GMM.unique()))
    return df


def window_correlation(df):
    corr_bb, p_value = scipy.stats.pearsonr(df['performance'].to_numpy(),
                                            df['energy'].to_numpy())

    fig2, axes2 = plt.subplots(ncols=2, nrows=1, constrained_layout=True, figsize=(12, 3))

    correlation_frames = window_sections(df, mode='window')
    sns.histplot(pd.DataFrame(data=correlation_frames, columns=['Pearson']), bins=20, kde=False, ax=axes2[0])
    axes2[0].axvline(corr_bb, ls='--', color='r')
    axes2[0].set_xlabel('Correlation of Window')
    axes2[0].set_title('Window Correlation')

    correlation_frames = window_sections(df, mode='data')
    sns.histplot(pd.DataFrame(data=correlation_frames, columns=['Pearson']), bins=20, kde=False, ax=axes2[1])
    axes2[1].axvline(corr_bb, ls='--', color='r')
    axes2[1].set_xlabel('Correlation of Dataslice')
    axes2[1].set_title('Dataslice Correlation')

    plt.show()


def correlation_error(df):
    min_perf = min(df.performance)
    max_perf = max(df.performance)

    print('Min perf:', min_perf, ' - Max perf:', max_perf)

    x = df['performance'].to_numpy()
    y = df['energy'].to_numpy()

    corr_bb, p_value = scipy.stats.pearsonr(x, y)
    print('Datapoints', len(df))
    print('Overall Correlation:', corr_bb)
    print()

    slope, intercept, r, p, stderr = scipy.stats.linregress(x, y)

    fig1, axes = plt.subplots(ncols=3, nrows=1, constrained_layout=True, figsize=(12, 3))

    sns.scatterplot(data=df, x='performance', y='energy', ax=axes[0], marker='+')
    axes[0].plot(x, intercept + slope * x, label="Regression Line")
    axes[0].set_title('Linear Regression')

    df['Lin Err'] = df.apply(lambda row: calc_lin_err(row['performance'], row['energy'], slope, intercept), axis=1)
    sns.histplot(pd.DataFrame(data=df, columns=['Lin Err']), bins=20, kde=False, ax=axes[1])
    axes[1].set_xlabel('Error Abs')
    axes[1].set_title('Linear Regression Absolute Error')

    df['Lin Err Norm'] = df['Lin Err'] / df['energy'] * 100
    sns.histplot(pd.DataFrame(data=df, columns=['Lin Err Norm']), bins=20, kde=False, ax=axes[2])
    axes[2].set_xlabel('Error in %')
    axes[2].set_title('Linear Regression Relative Error')
    print('Largest Error', max(list(df['Lin Err'])))
    print('Largest Relative Error', max(list(df['Lin Err Norm'])), 'Mean Energy', df['energy'].mean())
    print('Mean Relative Error', df['Lin Err Norm'].mean())

    plt.show()
    return df


def correlation_error_joint(df):
    x = df['performance'].to_numpy()
    y = df['energy'].to_numpy()
    slope, intercept, r, p, stderr = scipy.stats.linregress(x, y)
    g1 = sns.jointplot(data=df, x='performance', y='energy', marker='+', height=4)
    g1.ax_joint.plot(x, intercept + slope * x, label="Regression Line")

    g2 = sns.jointplot(data=df, x='performance', y='energy', marker='+', height=4)
    g2.ax_joint.plot(x, intercept + slope * x, label="Regression Line")
    g2.ax_joint.set_xscale('log')
    g2.ax_joint.set_yscale('log')

    plt.show()
    return


def clustering(df):
    min_perf = min(df.performance)
    max_perf = max(df.performance)
    corr_bb, p_value = scipy.stats.pearsonr(df['performance'].to_numpy(),
                                            df['energy'].to_numpy())

    fig2, axes2 = plt.subplots(ncols=4, nrows=1, constrained_layout=True, figsize=(12, 5))

    df = cluster_dbscan(df)
    sns.scatterplot(data=df, x='performance', y='energy', hue='DBSCAN', ax=axes2[0])
    axes2[0].set_xlabel('Performance [s]')
    axes2[0].set_ylabel('Energy Consumption [J]')
    axes2[0].set_title('DBSCAN')

    correlation_cluster = corr_cluster(df, 'DBSCAN')
    sns.histplot(pd.DataFrame(data=correlation_cluster, columns=['cluster']), bins=20, kde=False, ax=axes2[1])
    axes2[1].axvline(corr_bb, ls='--', color='r')
    axes2[1].set_title('DBSCAN Correlation')

    df = cluster_gmm(df)
    sns.scatterplot(data=df, x='performance', y='energy', hue='GMM', ax=axes2[2])
    axes2[2].set_xlabel('Performance [s]')
    axes2[2].set_ylabel('Energy Consumption [J]')
    axes2[2].set_title('GMM')

    correlation_cluster = corr_cluster(df, 'GMM')
    sns.histplot(pd.DataFrame(data=correlation_cluster, columns=['cluster']), bins=20, kde=False, ax=axes2[3])
    axes2[3].axvline(corr_bb, ls='--', color='r')
    axes2[3].set_title('GMM Correlation')

    plt.show()


def correlation_results(df, system):
    results = []
    min_perf = min(df.performance)
    max_perf = max(df.performance)

    # Linear Correlation
    corr_bb, p_value = scipy.stats.pearsonr(df['performance'].to_numpy(),
                                            df['energy'].to_numpy())

    # Linear Regression
    x = df['performance'].to_numpy()
    y = df['energy'].to_numpy()
    slope, intercept, r, p, stderr = scipy.stats.linregress(x, y)

    df['Lin Err'] = df.apply(lambda row: calc_lin_err(row['performance'], row['energy'], slope, intercept), axis=1)
    df['Lin Err ape'] = df['Lin Err'] / df['energy'] * 100
    mape = df['Lin Err ape'].mean()
    std = df['Lin Err ape'].std()

    # Correlation Windows
    # Thresolds: (-1) : (-0.7) : (-0.3) : 0.3 : 0.7 : 1
    # Strong N : Moderate N : Weak/None : Moderate P : Strong P
    bins = [-1, -0.7, -0.3, 0.3, 0.7, 1]

    correlation_frames = window_sections(df, window_pct=0.1, stepsize=0.5, mode='window')
    tmp_df = pd.DataFrame(data=correlation_frames, columns=['c'])

    results.append([system, corr_bb, mape, std] + list(tmp_df['c'].value_counts(bins=bins, sort=False)))
    return pd.DataFrame(results, columns=['System', 'Pearson', 'MAPE', 'StD', 'SN', 'MN', 'WN', 'MP', 'SP'])


def window_lin_reg_plots(systems, measurements_file):
    df_dump_paper_plots = []
    corr_dfs = []
    for system in systems:
        df = pd.read_csv(os.path.join(system, measurements_file), delimiter=';')
        system_name = system.rsplit('/', 1)[1]

        print()
        print('~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+')
        print(system_name)

        if 'revision' in list(df):
            rev = df.iloc[-1, :]['revision']
            df = df[df['revision'] == rev]
            df = df.drop(['revision'], axis=1)

        elif 'workload' in list(df):
            if system == '/home/mweber/git/performance-energy-correlation-code/data/jump3r':
                df = df[df['workload'] == 'dual-channel.wav']
            elif system == '/home/mweber/git/performance-energy-correlation-code/data/kanzi':
                df = df[df['workload'] == 'v5.12.tar']
            else:
                for key, tmp in df.groupby(['workload']):
                    print('Workload', key)

        correlation_error(df)
        window_correlation(df)
        corr_dfs.append(correlation_results(df, system_name))

        # correct axes scale because inconsistencies in data
        if system_name in paper_systems:
            df['system'] = system_name
            if system_name != 'x264':
                df['energy'] = df['energy'] / 1000
            if system_name == '7z':
                df['performance'] = df['performance'] / 1000
            if system_name == 'HSQLDB':
                df['energy'] = df['energy'] * 1000

            df_dump_paper_plots.append(df)
    return df_dump_paper_plots, corr_dfs


def correlation_table(corr_dfs):
    correlation_df = pd.concat(corr_dfs).round(3)
    correlation_df = correlation_df[['System', 'Pearson', 'SN', 'MN', 'WN', 'MP', 'SP', 'MAPE', 'StD']]
    correlation_df.sort_values(by='System', inplace=True, key=lambda col: col.str.lower())
    correlation_df = correlation_df.reset_index(drop=True)

    print(correlation_df.to_latex())
    return correlation_df


def correlation_overview_image(df_dump_paper_plots, output_path):
    # Fig 2 in Paper
    min_window = -1
    max_window = 1
    min_error = 0
    max_error = 0

    # palette = [sns.color_palette("husl")[3]]
    custom_pal = [sns.color_palette('tab10')[2]]
    sns.set_palette(custom_pal)

    for df in df_dump_paper_plots:
        if max_error < df['Lin Err Norm'].max():
            max_error = df['Lin Err Norm'].max()

    bin_width = 32
    # Hack:
    max_error = 85
    bin_width_window = (abs(min_window - max_window)) / bin_width
    bin_width_error = max_error / bin_width

    bins_window = np.arange(min_window, max_window + bin_width_window, bin_width_window)
    bins_error = np.arange(min_error, max_error + bin_width_error, bin_width_error)

    size = 15
    plt.rc('font', size=size)
    fig, axes = plt.subplots(ncols=3, nrows=6, constrained_layout=True, figsize=(10, 11))

    curr_row = 0
    for i, df in enumerate(df_dump_paper_plots):

        y = df['energy'].to_numpy()
        x = df['performance'].to_numpy()

        corr_bb, p_value = scipy.stats.pearsonr(x, y)
        slope, intercept, r, p, stderr = scipy.stats.linregress(x, y)

        ax0 = axes[curr_row][0]
        sns.scatterplot(data=df, x='performance', y='energy', ax=ax0, palette=custom_pal, alpha=.45)
        ax0.plot(x, intercept + slope * x, color=sns.color_palette('tab10')[7], label="Regression Line")
        ax0.set_ylabel('Energy [kJ]')
        ax0.set_xlabel('')
        if curr_row == 5: ax0.set_xlabel('Runtime performance [s]')
        if curr_row == 0: ax0.set_title('Linear regression')

        ax1 = axes[curr_row][1]
        ax1.set_xlim(xmin=-1.1, xmax=1.1)
        correlation_frames = window_sections(df, mode='window')
        sns.histplot(pd.DataFrame(data=correlation_frames, columns=['Pearson']), bins=bins_window, kde=False, ax=ax1,
                     palette=custom_pal, alpha=.45)
        ax1.axvline(corr_bb, ls='--', color='r')
        if curr_row == 5: ax1.set_xlabel('Correlation of slices')
        ax1.set_ylabel('Slices')
        ax1.get_legend().remove()
        if curr_row == 0:
            ax1.set_title('Slice correlation')

        ax2 = axes[curr_row][2]
        ax2.set_xlim(xmin=0, xmax=80)
        sns.histplot(pd.DataFrame(data=df, columns=['Lin Err Norm']), bins=bins_error, kde=False, ax=ax2,
                     palette=custom_pal, alpha=.45)
        if curr_row == 5: ax2.set_xlabel('Error in %')
        ax2.set_ylabel('Configurations')
        if curr_row == 0: ax2.set_title('Prediction error')
        ax2.get_legend().remove()

        ax3 = ax2.twinx()
        ax3.axes.yaxis.set_ticks([])
        plt.ylabel('\n' + paper_systems[i], fontsize=18)

        curr_row += 1

    plt.savefig(output_path+'examplesystems.pdf')
    plt.savefig(output_path+'examplesystems.png', dpi=200)


def print_usage() -> None:
    """
    Prints the usage of the python script.
    """
    print("Usage: generate_plots.py <InputPath> <OutputPath>")
    print("InputPath\t The path to the directory containing all relevant information of the case studies.")
    print("OutputPath\t The path to the directory where all plots should be exported to.")


def main() -> None:

    if len(sys.argv) == 3:
        # Read in the path to the case study data
        input_path = sys.argv[1]

        # Read in the output path of the plots
        output_path = sys.argv[2]
    else:
        input_path = './data/'
        output_path = './output/'

    systems = [os.path.join(input_path, system) for system in os.listdir(input_path)]
    measurements_file = 'measurements.csv'

    df_dump_paper_plots, corr_dfs = window_lin_reg_plots(systems, measurements_file)

    correlation_df = correlation_table(corr_dfs)
    correlation_overview_image(df_dump_paper_plots, output_path)


if __name__ == "__main__":
    main()
