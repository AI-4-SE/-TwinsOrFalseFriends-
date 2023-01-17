import os
import typing
from typing import List, TextIO

import scipy.stats
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from research_question import ResearchQuestion
from case_study import CaseStudy
from scripts.analysis.feature import Feature


class ResearchQuestion2(ResearchQuestion):
    name = "RQ2"

    def initialize_for_metrics(self, path: str):
        super().create_directory(path)
        super().initialize_tables(os.path.join(path, "README_local_error"),
                                  ["CaseStudy", "WindowMin", "WindowMax", "Option", "Value",
                                   "LocalRegressionErrorAbs", "LocalRegressionError",
                                   "GlobalRegressionErrorAbs", "GlobalRegressionError",
                                   "Group"], only_csv=True)
        super().initialize_tables(os.path.join(path, "README_post"), ["CaseStudy", "WindowMin", "WindowMax", "Option",
                                                                      "Value", "Configurations", "Pearson-Correlation",
                                                                      "Group"])
        super().initialize_tables(os.path.join(path, "README_post_mean_std"), ["CaseStudy", "WindowMin", "WindowMax",
                                                                               "Option", "Value", "Configurations",
                                                                               "MeanPerf", "MeanEnergy", "StdPerf",
                                                                               "StdEnergy", "Group"])

    def prepare(self, case_study: CaseStudy, input_path: str) -> None:
        pass

    def evaluate_metrics(self, case_study: CaseStudy, path: str, input_path: str) -> None:
        self.compute_correlation_and_mean(case_study, path)
        self.compute_local_error(case_study, path)

    def compute_local_error(self, case_study: CaseStudy, path: str) -> None:
        local_error_path = os.path.join(path, "README_local_error")
        for feature in case_study.features.values():
            if feature.strictly_mandatory is not None and feature.strictly_mandatory and \
                    (not feature.alternative_parent or feature.alternative_parent and feature.mandatory) \
                    and feature.binary:
                continue
            # for every binary non-mandatory feature, search the configurations where this feature is deselected
            if feature.binary:
                group_identifier = feature.name
                values = ['0', '1']
                # Distinguish when observing alternative groups
                # If the parent of the alternative group is optional, only investigate the value 0
                if feature.alternative_parent:
                    values = ['0']
                # For the children of an alternative group, only investigate the value 1
                elif len(feature.alternatives) > 0:
                    values = ['1']
                    group_identifier = feature.parent

                self.perform_local_error_analysis(case_study, feature, local_error_path, values, group_identifier)

            # for every numeric feature, compute the error for each numeric feature value
            else:
                numeric_feature_values = list(case_study.configurations[feature.name].unique())
                self.perform_local_error_analysis(case_study, feature, local_error_path, numeric_feature_values, feature.name)

    @staticmethod
    def perform_local_error_analysis(case_study: CaseStudy, feature: Feature, local_error_path: str,
                                     values: List[str], group_identifier: str) -> None:
        for value in values:
            all_configurations = case_study.configurations.loc[case_study.configurations[feature.name] == value]

            global_regression_error_all = ResearchQuestion.comp_regression_error(case_study, all_configurations,
                                                                                 all_configurations.copy())
            global_regression_error_all.reset_index(inplace=True)
            global_regression_error_all_out = global_regression_error_all['Lin Err Norm']
            global_regression_error_all_out_abs = global_regression_error_all['Lin Err']

            with open(f"{local_error_path}.csv", 'a') as output_file:
                for i in range(0, len(all_configurations)):
                    column_values = [case_study.name, "-", "-",
                                     feature.name, str(value),
                                     "{:.2f}".format(global_regression_error_all_out_abs[i]),
                                     "{:.2f}".format(global_regression_error_all_out[i]),
                                     "-",
                                     "-",
                                     group_identifier]
                    output_file.write(",".join(column_values))
                    output_file.write("\n")

            continue
            for window, perf_min, perf_max in case_study.windows:
                window_configurations = window.loc[window[feature.name] == value]
                if len(window_configurations) < 3:
                    continue
                # Perform global regression
                global_regression_error = ResearchQuestion.comp_regression_error(case_study,
                                                                                 window_configurations.copy(),
                                                                                 all_configurations)
                global_regression_error.reset_index(inplace=True)
                global_regression_error_series = global_regression_error['Lin Err Norm']
                global_regression_error_series_abs = global_regression_error['Lin Err']

                # Perform local regression
                local_regression_error = ResearchQuestion.comp_regression_error(case_study,
                                                                                window_configurations,
                                                                                window_configurations.copy())
                local_regression_error.reset_index(inplace=True)
                local_regression_error_series = local_regression_error['Lin Err Norm']
                local_regression_error_series_abs = local_regression_error['Lin Err']

                # Print error
                with open(f"{local_error_path}.csv", 'a') as output_file:
                    for i in range(0, len(window_configurations)):
                        column_values = [case_study.name, "{:.1f}".format(perf_min), "{:.1f}".format(perf_max),
                                         feature.name, str(value),
                                         "{:.2f}".format(local_regression_error_series_abs[i]),
                                         "{:.2f}".format(local_regression_error_series[i]),
                                         "{:.2f}".format(global_regression_error_series_abs[i]),
                                         "{:.2f}".format(global_regression_error_series[i]),
                                         group_identifier]
                        output_file.write(",".join(column_values))
                        output_file.write("\n")

    def compute_correlation_and_mean(self, case_study: CaseStudy, path: str) -> None:
        pearson_path = os.path.join(path, "README_post")
        mean_std_path = os.path.join(path, "README_post_mean_std")

        for feature in case_study.features.values():
            if feature.strictly_mandatory is not None and feature.strictly_mandatory and \
                    (not feature.alternative_parent or feature.alternative_parent and feature.mandatory) \
                    and feature.binary:
                continue
            # for every binary non-mandatory feature, search the configurations where this feature is deselected
            if feature.binary:
                group_identifier = feature.name
                values = ['0', '1']
                # Distinguish when observing alternative groups
                # If the parent of the alternative group is optional, only investigate the value 0
                if feature.alternative_parent:

                    values = ['0']
                # For the children of an alternative group, only investigate the value 1
                elif len(feature.alternatives) > 0:
                    values = ['1']
                    group_identifier = feature.parent

                self.perform_pearson_correlation(case_study, case_study.configurations, feature,
                                                 pearson_path, values, group_identifier)
                self.compute_mean_standard_dev(case_study, case_study.configurations, feature,
                                               mean_std_path, values, group_identifier)

                for window, perf_min, perf_max in case_study.windows:
                    current_configurations = window

                    self.perform_pearson_correlation(case_study, current_configurations, feature,
                                                     pearson_path, values, group_identifier, perf_min, perf_max)
                    self.compute_mean_standard_dev(case_study, current_configurations, feature,
                                                   mean_std_path, values, group_identifier, perf_min, perf_max)
            # for every numeric feature, perform one Pearson Correlation for each value of the numeric feature
            else:
                numeric_feature_values = list(case_study.configurations[feature.name].unique())
                self.perform_pearson_correlation(case_study, case_study.configurations, feature, pearson_path,
                                                 numeric_feature_values, feature.name)
                self.compute_mean_standard_dev(case_study, case_study.configurations, feature, mean_std_path,
                                               numeric_feature_values, feature.name)
                for window, perf_min, perf_max in case_study.windows:
                    current_configurations = window
                    numeric_feature_values = list(current_configurations[feature.name].unique())
                    self.perform_pearson_correlation(case_study, current_configurations, feature,
                                                     pearson_path, numeric_feature_values, feature.name,
                                                     perf_min, perf_max)
                    self.compute_mean_standard_dev(case_study, current_configurations, feature,
                                                   mean_std_path, numeric_feature_values, feature.name,
                                                   perf_min, perf_max)

    @staticmethod
    def compute_mean_standard_dev(case_study: CaseStudy, current_configurations: pd.DataFrame, feature: Feature,
                                  output_path: str, values: List[typing.Any], group_identifier: str,
                                  perf_min: float = -1, perf_max: float = -1):
        perf_min_str = "{:.1f}".format(perf_min)
        perf_max_str = "{:.1f}".format(perf_max)
        if perf_min == -1 and perf_max == -1:
            perf_min_str = "-"
            perf_max_str = "-"
        for value in values:
            configurations_selected = current_configurations.loc[
                current_configurations[feature.name] == value]
            if len(configurations_selected) == 0:
                continue
            mean_perf = configurations_selected[case_study.Performance].mean()
            std_perf = configurations_selected[case_study.Performance].std()
            mean_energy = configurations_selected[case_study.Energy].mean()
            std_energy = configurations_selected[case_study.Energy].std()
            # Print the result in both files
            ResearchQuestion.write_to_tables(output_path, [case_study.name, perf_min_str,
                                                           perf_max_str, feature.name, str(value),
                                                           str(len(configurations_selected)), str(mean_perf),
                                                           str(mean_energy), str(std_perf), str(std_energy),
                                                           group_identifier])

    @staticmethod
    def perform_pearson_correlation(case_study: CaseStudy, current_configurations: pd.DataFrame, feature: Feature,
                                    output_path: str, values: List[typing.Any], group_identifier: str,
                                    perf_min: float = -1, perf_max: float = -1):
        perf_min_str = "{:.2f}".format(perf_min)
        perf_max_str = "{:.2f}".format(perf_max)
        if perf_min == -1 and perf_max == -1:
            perf_min_str = "-"
            perf_max_str = "-"

        for value in values:
            configurations_selected = current_configurations.loc[
                current_configurations[feature.name] == value]
            if len(configurations_selected) < 2:
                continue
            corr, _ = scipy.stats.pearsonr(
                configurations_selected[case_study.Performance].to_numpy(),
                configurations_selected[case_study.Energy].to_numpy())
            # Print the result
            ResearchQuestion.write_to_tables(output_path, [case_study.name, perf_min_str,
                                                           perf_max_str, feature.name, str(value),
                                                           str(len(configurations_selected)), "{:.3f}".format(corr),
                                                           group_identifier])

    def generate_plots(self, case_study: CaseStudy, path: str, input_path: str) -> None:
        self.generate_window_plot(case_study, path, input_path, True)

    @staticmethod
    def generate_window_plot(case_study: CaseStudy, path: str, input_path: str, for_paper: bool = True):
        """Generate a scatter plot for each (1) optional feature, (2) alternative group, and (3) numeric feature."""
        path = os.path.join(path, 'scatterplot')
        ResearchQuestion.create_directory(path)
        current_configurations = case_study.configurations
        for feature in case_study.features.values():
            if feature.strictly_mandatory is not None and feature.strictly_mandatory and \
                    not feature.alternative_parent and feature.binary:
                continue
            column_name = feature.name
            # Exclude the alternative children since they are already included in the plot of the alternative parent
            if len(feature.alternatives) > 0:
                continue
            # Preprocess the data by including an additional
            elif feature.alternative_parent:
                column_name = feature.name + "_group"
                current_configurations = current_configurations.copy()
                current_configurations[column_name] = [feature.name] * len(current_configurations)
                for child in feature.children:
                    current_configurations.loc[current_configurations[child] == '1', column_name] = child
                    if case_study.name == "HSQLDB" and child == "crypt_blowfish":
                        current_configurations.loc[(current_configurations["memory_tables"] == '1') & (current_configurations[child] == '1'), column_name] = "Blowfish (* MemoryTables)"
                        current_configurations.loc[(current_configurations["cached_tables"] == '1') & (current_configurations[child] == '1'), column_name] = "Blowfish (* CachedTables)"
                        #print(child)

            if for_paper:
                fig = plt.figure(figsize=(12, 5))
                size = 22
                plt.rc('font', size=size)
            else:
                fig = plt.figure(figsize=(16, 8))
            ax = fig.add_subplot(1, 1, 1)

            if not for_paper:
                # Draw the windows
                dashed_line = False
                for window, perf_min, perf_max in case_study.windows:
                    linestyle = '-'
                    if dashed_line:
                        dashed_line = False
                        linestyle = '--'
                    else:
                        dashed_line = True
                    plt.axvline(perf_min, linestyle=linestyle)
                    plt.axvline(perf_max, linestyle=linestyle)

            if case_study.name == "x264" and column_name == 'core_group':
                g = sns.scatterplot(data=current_configurations,
                            style=column_name,
                            hue=column_name,
                            alpha=.45,
                            s=140,
                            style_order=['core1', 'core2', 'core3', 'core4'],
                            hue_order=['core1', 'core2', 'core3', 'core4'],
                            x=case_study.Performance,
                            y=case_study.Energy)
            elif case_study.name == "kanzi" and column_name == 'jobs_group':
                g = sns.scatterplot(data=current_configurations,
                            style=column_name,
                            hue=column_name,
                            alpha=.45,
                            s=140,
                            style_order=['jobs_1', 'jobs_4', 'jobs_8'],
                            hue_order=['jobs_1', 'jobs_4', 'jobs_8'],
                            x=case_study.Performance,
                            y=case_study.Energy)
            else:
                g = sns.scatterplot(data=current_configurations,
                            style=column_name,
                            hue=column_name,
                            alpha=.45,
                            s=140,
                            x=case_study.Performance,
                            y=case_study.Energy)
            sns.despine()

            if case_study.name == "7z" and column_name == "CompressionMethod_group":
                g.legend_.set_title("Compression method")
            if case_study.name == "kanzi" and column_name == "jobs_group":
                g.legend_.set_title("Jobs")
                labels = ['1', '4', '8']
                for t, l in zip(g.legend_.texts, labels):
                    t.set_text(l)
            if case_study.name == "HSQLDB" and column_name == "encryption_group":
                g.legend_.set_title("Encryption (* Tables)")
                #print(g.legend_.texts)
                labels = ['No encryption', 'AES', 'Blowfish * MemoryTables', 'Blowfish * CachedTables']
                for t, l in zip(g.legend_.texts, labels):
                    t.set_text(l)
            if case_study.name == "x264" and column_name == 'core_group':
                g.legend_.set_title("Cores")
                labels = ['1', '2', '3', '4']
                for t, l in zip(g.legend_.texts, labels):
                    t.set_text(l)

            ax.set_xlabel('Performance [s]')
            ax.set_ylabel('Energy consumption [kJ]')
            fig.tight_layout()
            #print(os.path.join(path, f"scatterplot_{case_study.name}_{column_name}.pdf"))
            fig.savefig(os.path.join(path, f"scatterplot_{case_study.name}_{column_name}.pdf"))
            fig.savefig(os.path.join(path, f"scatterplot_{case_study.name}_{column_name}.png"), dpi=200)
            plt.close(fig)

    def finish(self, path: str) -> None:
        pass
