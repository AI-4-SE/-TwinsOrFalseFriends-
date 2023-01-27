#!/bin/env python3
import sys
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from ast import literal_eval
import seaborn as sns
import numpy as np
import subprocess
from datetime import datetime
import time
import os


def plot_ml_2column(methods_for_proxy_analysis, output_path):
    size = 13
    plt.rc('font', size=size)

    methods_for_proxy_analysis = methods_for_proxy_analysis.rename(columns={"corr ratio": "Energy Factor"})

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 7), constrained_layout=True, sharex=False)
    axes[2, 0].set_xlabel('System transfer factor', fontsize=20)
    axes[2, 1].set_xlabel('Option transfer factors', fontsize=20)

    sns.rugplot(data=methods_for_proxy_analysis[methods_for_proxy_analysis['method'] == 'find_best_match'],
                x='Energy Factor',
                ax=axes[0, 0], height=0.3, hue='ram', alpha=0.5, palette="tab10", lw=0.8, expand_margins=False)
    sns.rugplot(data=methods_for_proxy_analysis[methods_for_proxy_analysis['method'] == 'BZ2_compressBlock'],
                x='Energy Factor',
                ax=axes[1, 0], height=0.3, hue='cores', alpha=0.5, palette="tab10", lw=0.8, expand_margins=False)
    sns.rugplot(data=methods_for_proxy_analysis[methods_for_proxy_analysis['method'] == 'primary_hash'],
                x='Energy Factor',
                ax=axes[2, 0], height=0.3, hue='level', alpha=0.5, palette="tab10", lw=0.8, expand_margins=False)

    sns.kdeplot(data=methods_for_proxy_analysis[methods_for_proxy_analysis['method'] == 'find_best_match'],
                x='Energy Factor', ax=axes[0, 0], c=sns.color_palette('tab10')[7], bw_adjust=1, cut=27)
    sns.kdeplot(data=methods_for_proxy_analysis[methods_for_proxy_analysis['method'] == 'BZ2_compressBlock'],
                x='Energy Factor', ax=axes[1, 0], c=sns.color_palette('tab10')[7], bw_adjust=0.5)
    sns.kdeplot(data=methods_for_proxy_analysis[methods_for_proxy_analysis['method'] == 'primary_hash'],
                x='Energy Factor', ax=axes[2, 0], c=sns.color_palette('tab10')[7], bw_adjust=1, cut=20)

    axes[0, 0].get_legend().remove()
    axes[1, 0].get_legend().remove()
    axes[2, 0].get_legend().remove()

    axes[0, 0].set_yticks([])
    axes[1, 0].set_yticks([])
    axes[2, 0].set_yticks([])

    axes[0, 0].set_ylabel('')
    axes[1, 0].set_ylabel('')
    axes[2, 0].set_ylabel('')
    axes[0, 0].set_xlabel('')
    axes[1, 0].set_xlabel('')

    g1 = sns.kdeplot(data=methods_for_proxy_analysis[methods_for_proxy_analysis['method'] == 'find_best_match'],
                     x='Energy Factor', palette="tab10", hue='ram', ax=axes[0, 1])
    g2 = sns.kdeplot(data=methods_for_proxy_analysis[methods_for_proxy_analysis['method'] == 'BZ2_compressBlock'],
                     x='Energy Factor', palette="tab10", hue='cores', ax=axes[1, 1])
    g3 = sns.kdeplot(data=methods_for_proxy_analysis[methods_for_proxy_analysis['method'] == 'primary_hash'],
                     x='Energy Factor', palette="tab10", hue='level', ax=axes[2, 1])

    axes[0, 1].set_yticks([])
    axes[1, 1].set_yticks([])
    axes[2, 1].set_yticks([])

    axes[0, 1].set_xlabel('')
    axes[1, 1].set_xlabel('')

    axes[0, 1].yaxis.set_label_position("right")
    axes[0, 1].yaxis.tick_right()
    axes[0, 1].set_ylabel('find_best\n_match', labelpad=8, fontsize=20)

    axes[1, 1].yaxis.set_label_position("right")
    axes[1, 1].yaxis.tick_right()
    axes[1, 1].set_ylabel('BZ2\n_compressBlock', labelpad=8, fontsize=20)

    axes[2, 1].yaxis.set_label_position("right")
    axes[2, 1].yaxis.tick_right()
    axes[2, 1].set_ylabel('primary\n_hash', labelpad=8, fontsize=20)

    g1.legend_.set_title("Ram")
    g2.legend_.set_title("Cores")
    g3.legend_.set_title("Level")

    for i in [0, 1]:
        for j in [0, 1, 2]:
            axes[j, i].set_xlim(1.3, 1.9)

    dump_lines = axes[2, 1].lines
    axes[2, 1].legend(reversed(dump_lines), reversed(['9', '8', '7', '6', '5', '4', '3', '2', '1']), ncol=2)
    g3.legend_.set_title("Level")

    plt.savefig(output_path+'rq3_overview_small.pdf')
    plt.savefig(output_path+'rq3_overview_small.png', dpi=200)
    plt.show()
    return


def extract_cfg(row_idx, lines):
    # lrzip_binary_options = ['--encrypt=secret', '-U', '-T']# 8
    # one_of_compression = ['--lzma', '-b', '-g', '-l', '-n', '-z']# 8 * 6 = 48
    # one_of_level = ['-L 1', '-L 2', '-L 3', '-L 4', '-L 5', '-L 6', '-L 7', '-L 8', '-L 9']# 48 * 9 = 432
    # one_of_cores = ['-p 1', '-p 2', '-p 3', '-p 4']# 432 * 4 = 1728
    # one_of_ram = ['-m 2', '-m 6', '-m 18']# 1728 * 3 = 5184

    # no gzip in 200 configurations

    curr_line = lines[int(row_idx) - 1]
    curr_params = curr_line.split()

    encrypt = 1 if '--encrypt=secret' in curr_line else 0
    wind_size_unl = 1 if '-U' in curr_line else 0
    threshold = 1 if '-T' in curr_line else 0

    comp = ''
    if '--lzma' in curr_line:
        comp = 'lzma'
    elif '-b' in curr_line:
        comp = 'bzip2'
    elif '-g' in curr_line:
        comp = 'gzip'
    elif '-l' in curr_line:
        comp = 'lzo'
    elif '-n' in curr_line:
        comp = 'no_compress'
    else:
        comp = 'zpaq'

    level = curr_params[curr_params.index('-L') + 1]
    cores = curr_params[curr_params.index('-p') + 1]
    ram = curr_params[curr_params.index('-m') + 1]

    return [int(cores), int(ram), int(level), encrypt, wind_size_unl, threshold, comp]


def get_method_share(row, total_time):
    return row['time']*100/total_time


def ml_analysis(input_path, output_path):
    in_path = input_path+'lrzip/measurements_ml.csv'
    df = pd.read_csv(in_path, index_col=0)
    df_lrzip = df[['taskID', 'method', 't_delta', 'energy']].groupby(['taskID', 'method'])[
        't_delta', 'energy'].sum().reset_index()

    cfg_file = input_path+'lrzip/lrzip_prof_configurations_rnd200.cfg'

    influential_methods_arr = []
    all_method_mean_perfs = []
    skipped_methods = 0

    dfs_coll = []
    lines = [line.strip() for line in open(cfg_file)]
    for index, row in df_lrzip.iterrows():
        out = extract_cfg(row['taskID'], lines)
        dfs_coll.append([row.taskID, row.method, row.t_delta, row.energy] + out)
    df_lrzip_opt = pd.DataFrame(dfs_coll,
                                columns=['taskID', 'method', 'time', 'energy', 'cores', 'ram', 'level', 'encrypt',
                                         'wind_size_unl', 'threshold', 'compression'])

    out_arr = []
    for key, tmp_grp in df_lrzip_opt.groupby(['taskID']):
        cfg_perf = np.sum(tmp_grp['time'])
        num_methods = len(tmp_grp)
        # print(key)
        tmp_grp = tmp_grp.reset_index(drop=True)
        tmp_grp['time frac cfg'] = tmp_grp.apply(get_method_share, total_time=cfg_perf, axis=1)
        out_arr.append(tmp_grp)

    df_lrzip_opt = pd.concat(out_arr)

    for method, method_df in df_lrzip_opt.groupby('method'):
        methods_mean_perf = method_df['time frac cfg'].mean()
        all_method_mean_perfs.append(methods_mean_perf)

        # filter value in percent
        filter_mean_value = 0.1
        if methods_mean_perf < filter_mean_value:
            # print(method, methods_mean_perf, methods_mean_perf < filter_mean_value)
            skipped_methods += 1
            continue
        elif len(method_df) < 20:
            skipped_methods += 1
            continue

        print(method, methods_mean_perf, len(method_df), methods_mean_perf < filter_mean_value)
        fig, axes = plt.subplots(nrows=1, ncols=7, figsize=(25, 5))

        # no gzip
        # 'cores', 'ram', 'level', 'encrypt', 'wind_size_unl', 'threshold', 'lzma', 'bzip2', 'gzip', 'lzo',
        #    'no_compress', 'zpaq'

        sns.scatterplot(data=method_df, x='time', y='energy', palette="tab10", hue='cores', ax=axes[0])
        sns.scatterplot(data=method_df, x='time', y='energy', palette="tab10", hue='ram', ax=axes[1])
        sns.scatterplot(data=method_df, x='time', y='energy', palette="tab10", hue='level', ax=axes[2])
        sns.scatterplot(data=method_df, x='time', y='energy', palette="tab10", hue='encrypt', ax=axes[3])
        sns.scatterplot(data=method_df, x='time', y='energy', palette="tab10", hue='wind_size_unl', ax=axes[4])
        sns.scatterplot(data=method_df, x='time', y='energy', palette="tab10", hue='threshold', ax=axes[5])
        sns.scatterplot(data=method_df, x='time', y='energy', palette="tab10", hue='compression', ax=axes[6])

        fig.tight_layout()

        if not os.path.exists(output_path+'RQ3/lrzip'):
            os.makedirs(output_path+'RQ3/lrzip')

        plt.savefig(output_path+'RQ3/lrzip/' + method + '.pdf')
        plt.show()

        method_df['method'] = method
        influential_methods_arr.append(method_df)

    influential_methods = pd.concat(influential_methods_arr)

    influential_methods['corr ratio'] = influential_methods['energy'] / influential_methods['time']
    methods_for_proxy_analysis = influential_methods.reset_index(drop=True)
    return methods_for_proxy_analysis


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

    df = ml_analysis(input_path, output_path)
    plot_ml_2column(df, output_path)


if __name__ == "__main__":
    main()
