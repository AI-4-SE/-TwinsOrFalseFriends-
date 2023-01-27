#!/bin/env python3
import os
import gc
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

rq1_error = {'brotli': 9.362309649338012,
             'jump3r': 10.07376088427494,
             'MongoDB': 1.6989472174721387,
             'LLVM': 0.7970528135693291,
             'x264': 10.360177309928984,
             'kanzi': 70.43173695926725,
             'PostgreSQL': 1.2976722566347876,
             'nginx': 8.285394035654088,
             'exastencils': 5.730121735854243,
             '7z': 14.793548148470265,
             'VP8': 22.056542234809573,
             'Apache': 6.312060052327612,
             'lrzip': 31.095766481514023,
             'HSQLDB': 3.0880223980905885
             }

# Correlation Windows
# Strong N : Moderate N : Weak/None : Moderate P : Strong P
bins = [-1, -0.7, -0.3, 0.3, 0.7, 1]


def option_wise_correlation_table(input_path):
    in_path = input_path+'./../RQ2/'
    file_corr = 'README_post.csv'
    file_window_corr = 'README_local_error.csv'
    file_stat = 'README_post_mean_std.csv'

    df_error = pd.read_csv(os.path.join(in_path, file_window_corr))
    print(list(df_error))

    df_corr = pd.read_csv(os.path.join(in_path, file_corr))

    # Correlation Part
    correlation_res_mean = []
    for study, df in df_corr.groupby(['CaseStudy']):

        mask = df['WindowMin'] == '-'
        df_windows = df[~mask]
        df_all = df[mask]

        opt_whole_corr = []
        for option, sub_df in df_all.groupby(['Group']):
            opt_whole_corr.append(sub_df['Pearson-Correlation'].mean())

        opt_whole_corr_df = pd.DataFrame(opt_whole_corr, columns=['corr'])

        opt_window_corr = []
        for option, sub_df in df_windows.groupby(['Group']):
            opt_window_corr.append(sub_df['Pearson-Correlation'].tolist())
        opt_window_corr_out = []
        for l in opt_window_corr:
            opt_window_corr_out = opt_window_corr_out + l
        opt_window_corr_df = pd.DataFrame(opt_window_corr_out, columns=['corr'])

        correlation_res_mean.append([study, opt_whole_corr_df['corr'].mean()] +
                                    list(opt_window_corr_df['corr'].value_counts(bins=bins, sort=False)))

    # Regression Part
    min_error_diff = 5
    lin_model_res = []

    for study, df in df_error.groupby(['CaseStudy']):
        regression_vals = []
        # print(study)
        count_err_diff = 0
        for i, (option, sub_df) in enumerate(df.groupby(['Group'])):
            if abs(sub_df['LocalRegressionError'].mean() - rq1_error.get(study)) > min_error_diff:
                count_err_diff += 1
            regression_vals.append(sub_df['LocalRegressionError'].mean())
        # print(regression_vals)
        # print()
        lin_model_res.append([study, min(regression_vals), max(regression_vals),
                              np.mean(regression_vals),
                              count_err_diff])

    # 'System', 'Pearson', 'SN', 'MN', 'WN', 'MP', 'SP', 'MAPE', 'StD'
    rq2_df = pd.merge(pd.DataFrame(correlation_res_mean, columns=['System', 'Pearson', 'SN', 'MN', 'WN', 'MP', 'SP']),
                      pd.DataFrame(lin_model_res, columns=['System', 'Best', 'Worst', 'Mean', 'FW IO']), on='System')
    rq2_df = rq2_df.rename(columns={"FW IO": "CO", "Best": "MAPE"})

    rq2_df = rq2_df.sort_values(by=['System'], key=lambda col: col.str.lower())
    rq2_df = rq2_df.round(2)
    rq2_df = rq2_df[['System', 'Pearson', 'SN', 'MN', 'WN', 'MP', 'SP', 'MAPE', 'CO']]
    rq2_df['CI'] = [3, 0, 0, 0, 3, 0, 2, 0, 3, 3, 0, 0, 3, 5]
    print(rq2_df.to_latex(index=False))
    return df_error


def options_influence_on_corr(input_path, output_path, df_error):
    results = []
    corr_res_cfg = []
    systems_to_show = ['7z', 'x264', 'LLVM', 'PostgreSQL', 'HSQLDB']
    systems_fig_4 = ['kanzi', '7z']
    systems_fig_4_arr = []

    path_7z = input_path+'7z/measurements.csv'
    df_7z = pd.read_csv(path_7z, sep=';')
    df_7z_arr = []
    for index, row in df_7z.iterrows():
        # 'jobs_8', 'jobs_1', 'jobs_4'
        # print(row['LZMA'],row['LZMA2'],row['PPMd'],row['BZip2'],row['Deflate'])
        if row['LZMA']:
            df_7z_arr.append(['7z', 'LZMA', row['energy'], row['performance']])
        if row['LZMA2']:
            df_7z_arr.append(['7z', 'LZMA2', row['energy'], row['performance']])
        if row['PPMd']:
            df_7z_arr.append(['7z', 'PPMd', row['energy'], row['performance']])
        if row['BZip2']:
            df_7z_arr.append(['7z', 'BZip2', row['energy'], row['performance']])
        if row['Deflate']:
            df_7z_arr.append(['7z', 'Deflate', row['energy'], row['performance']])

    df_7z = pd.DataFrame(data=df_7z_arr, columns=['CaseStudy', 'CompressionMethod', 'energy', 'performance'])
    df_7z['energy'] = df_7z['energy'] / 1000
    df_7z['performance'] = df_7z['performance'] / 1000

    path_kanzi = input_path+'kanzi/measurements.csv'
    df_kanzi = pd.read_csv(path_kanzi, sep=';')
    df_kanzi_arr = []
    for index, row in df_kanzi.iterrows():
        # 'jobs_8', 'jobs_1', 'jobs_4'
        if row['jobs_8']:
            df_kanzi_arr.append(['kanzi', 8, row['energy'], row['performance']])
        if row['jobs_1']:
            df_kanzi_arr.append(['kanzi', 1, row['energy'], row['performance']])
        if row['jobs_4']:
            df_kanzi_arr.append(['kanzi', 4, row['energy'], row['performance']])

    df_kanzi = pd.DataFrame(data=df_kanzi_arr, columns=['CaseStudy', 'Jobs', 'energy', 'performance'])
    df_kanzi['energy'] = df_kanzi['energy'] / 1000

    for study, df in df_error.groupby(['CaseStudy']):

        for option, sub_df in df.groupby(['Group']):
            color_column = 'Option' if len(sub_df.Option.unique()) > 1 else 'Value'
            for value, val_df in sub_df.groupby([color_column]):
                results.append([study, option, value, len(sub_df), val_df['LocalRegressionError'].mean(),
                                val_df['LocalRegressionError'].std(), val_df['LocalRegressionError'].mean(),
                                val_df['LocalRegressionError'].std()])

        print(study)
        if study in systems_fig_4:
            df['CaseStudy'] = study
            systems_fig_4_arr.append(df)

        if study not in systems_to_show:
            # continue
            pass

        rows = len(df.Group.unique())
        fig1, axes = plt.subplots(ncols=2, nrows=rows, constrained_layout=True, sharex='col',
                                  figsize=(12, rows * 3))

        if not os.path.exists(output_path+'RQ2/' + study + '/LinRegError'):
            os.makedirs(output_path+'RQ2/' + study + '/LinRegError')

        if not os.path.exists(output_path+'RQ2/' + study + '/OptionWiseLinearError'):
            os.makedirs(output_path+'RQ2/' + study + '/OptionWiseLinearError')

        for i, (option, sub_df) in enumerate(df.groupby(['Group'])):
            axes[i][0].set_title(option)

            color_column = 'Option' if len(sub_df.Option.unique()) > 1 else 'Value'

            sns.histplot(data=sub_df, x='LocalRegressionError', bins=20, hue=color_column, kde=False, ax=axes[i][0])
            axes[i][0].axvline(rq1_error.get(study), ls='--', color='r')
            axes[i][0].axvline(sub_df['LocalRegressionError'].mean(), ls='-', color='g')

            sns.kdeplot(data=sub_df, x='LocalRegressionError', hue=color_column, ax=axes[i][1])
            axes[i][1].axvline(rq1_error.get(study), ls='--', color='r')
            axes[i][1].axvline(sub_df['LocalRegressionError'].mean(), ls='-', color='g')

            del sub_df
            gc.collect()

        plt.savefig(output_path+'RQ2/' + study + '/LinRegError/all_options.pdf')
        plt.savefig(output_path+'RQ2/' + study + '/LinRegError/all_options.png')
        plt.show()
        plt.close()

        ids = {'Id': ['jobs_1', 'jobs_4', 'jobs_8'],
               'City': ['1', '4', '8']}
        ids = dict(zip(ids['Id'], ids['City']))

        for i, (option, sub_df) in enumerate(df.groupby(['Group'])):
            color_column = 'Option' if len(sub_df.Option.unique()) > 1 else 'Value'

            if option == 'jobs':
                newOption = 'Jobs'
            elif option == 'CompressionMethod':
                newOption = 'Compression method'
            else:
                newOption = option

            sub_df[newOption] = sub_df[color_column].replace(ids, regex=True)

            size = 20
            plt.rc('font', size=size)

            plt.figure(figsize=(12, 6))
            g = sns.kdeplot(data=sub_df, x='LocalRegressionError', hue=newOption)
            g.set(xlabel='Regression error')
            plt.axvline(rq1_error.get(study), ls='--', color='r')
            plt.savefig(output_path+'RQ2/' + study +
                        '/OptionWiseLinearError/' + study + '_' + option + '.pdf', bbox_inches='tight')
            plt.show()
            plt.close()

        del df
        gc.collect()
        # break
    df_systems_fig_4 = pd.concat(systems_fig_4_arr)

    dist_data_7z = df_systems_fig_4[
        (df_systems_fig_4['CaseStudy'] == '7z') & (df_systems_fig_4['Group'] == 'CompressionMethod')]

    dist_data_kanzi = df_systems_fig_4[
        (df_systems_fig_4['CaseStudy'] == 'kanzi') & (df_systems_fig_4['Group'] == 'jobs')]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), constrained_layout=True, sharex=False,
                             gridspec_kw={'width_ratios': [3, 2]})

    custom_pal = sns.color_palette('Accent')

    sns.scatterplot(data=df_kanzi, x='performance', y='energy', palette='tab10', hue='Jobs', alpha=.45, s=140,
                    style='Jobs', ax=axes[0, 0])
    sns.scatterplot(data=df_7z, x='performance', y='energy', palette='tab10', hue='CompressionMethod', alpha=.45, s=140,
                    style='CompressionMethod', ax=axes[1, 0])

    g1 = sns.kdeplot(data=dist_data_kanzi, x='LocalRegressionError', palette='tab10', hue='Option', ax=axes[0, 1])
    g2 = sns.kdeplot(data=dist_data_7z, x='LocalRegressionError', palette='tab10', hue='Option', ax=axes[1, 1])

    axes[0, 1].axvline(70.43, ls='--', color='r')
    axes[1, 1].axvline(14.79, ls='--', color='r')

    axes[0, 0].get_legend().remove()
    axes[1, 0].get_legend().remove()

    axes[0, 0].set_xlabel('')
    axes[0, 1].set_xlabel('')

    axes[1, 0].set_xlabel('Performance [s]')
    axes[1, 1].set_xlabel('Regression error')

    axes[0, 0].set_ylabel('Energy consumption [kJ]')
    axes[1, 0].set_ylabel('Energy consumption [kJ]')

    axes[0, 1].set_yticks([])
    axes[1, 1].set_yticks([])

    axes[0, 1].yaxis.set_label_position("right")
    axes[0, 1].yaxis.tick_right()
    axes[0, 1].set_ylabel('\nkanzi', fontsize=22)

    axes[1, 1].yaxis.set_label_position("right")
    axes[1, 1].yaxis.tick_right()
    axes[1, 1].set_ylabel('\n7z', fontsize=22)

    g1.legend_.set_title("Jobs")
    g2.legend_.set_title("Compression method")

    sns.despine()

    plt.savefig(output_path+'rq2_example.pdf')
    plt.savefig(output_path+'rq2_example.png', dpi=200)
    plt.show()


def print_usage() -> None:
    """
    Prints the usage of the python script.
    """
    print("Usage: generate_plots.py <InputPath> <OutputPath>")
    print("InputPath\t The path to the directory containing all relevant information of the case studies.")
    print("OutputPath\t The path to the directory where all plots should be exported to.")


def main() -> None:

    if len(sys.argv) != 3:
        print_usage()
        exit(0)

    # Read in the path to the case study data
    input_path = sys.argv[1]

    # Read in the output path of the plots
    output_path = sys.argv[2]

    df_error = option_wise_correlation_table(input_path)
    options_influence_on_corr(input_path, output_path, df_error)


if __name__ == "__main__":
    main()
