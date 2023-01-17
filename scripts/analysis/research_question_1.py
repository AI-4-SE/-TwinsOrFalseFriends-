import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import seaborn as sns
import sklearn.cluster

from case_study import CaseStudy
from research_question import ResearchQuestion


class ResearchQuestion1(ResearchQuestion):
    name = "RQ1"

    def __init__(self):
        self.clustered_configurations = dict()
        self.regression_error = None

    def initialize_for_metrics(self, path: str):
        super().create_directory(path)

    def prepare(self, case_study: CaseStudy, input_path: str) -> None:
        self.clustered_configurations = self.analyze_windows(case_study)
        self.regression_error = self.comp_regression_error(case_study, case_study.configurations,
                                                           case_study.configurations)

    @staticmethod
    def analyze_windows(case_study: CaseStudy) -> List[float]:
        windows = case_study.windows
        windows_correlation = []
        for window, _, _ in windows:
            corr, p_value = scipy.stats.pearsonr(window[case_study.Performance].to_numpy(),
                                                 window[case_study.Energy].to_numpy())
            windows_correlation.append(corr)
        clustered_configurations = windows_correlation
        return clustered_configurations

    def evaluate_metrics(self, case_study: CaseStudy, path: str, input_path: str) -> None:
        pass

    def generate_correlation_window_plot(self, case_study: CaseStudy, path: str, input_path: str) -> None:
        correlation_path = os.path.join(path, 'correlation_window')
        super().create_directory(correlation_path)
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(1, 1, 1)
        sns.histplot(pd.DataFrame(data=self.clustered_configurations, columns=['Pearson']),
                     bins=20, kde=False, ax=ax)
        ax.set_ylabel('# windows')
        ax.set_title('Window correlation')
        fig.tight_layout()
        fig.savefig(os.path.join(correlation_path, f"windows_{case_study.name}.pdf"))
        plt.close(fig)

    def generate_scatterplots(self, case_study: CaseStudy, path: str, input_path: str) -> None:
        scatter_path = os.path.join(path, 'scatterplot')
        super().create_directory(scatter_path)

        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(1, 1, 1)
        sns.regplot(data=case_study.configurations, x=case_study.Performance, y=case_study.Energy, ci=95)

        ax.set_xlabel('Performance [s]')
        ax.set_ylabel('Energy Consumption [kJ]')

        fig.tight_layout()
        fig.savefig(os.path.join(scatter_path, f"scatterplot_{case_study.name}.pdf"))
        plt.close(fig)

    def generate_reg_error_dist(self, case_study: CaseStudy, path: str, input_path: str) -> None:
        reg_error_path = os.path.join(path, 'regression_error')
        super().create_directory(reg_error_path)

        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(1, 1, 1)
        sns.histplot(pd.DataFrame(data=self.regression_error, columns=['Lin Err Norm']), bins=20, kde=False, ax=ax)
        ax.set_xlabel('Error in %')
        ax.set_title('Linear Regression Relative Error')

        fig.tight_layout()
        fig.savefig(os.path.join(reg_error_path, f"scatterplot_{case_study.name}.pdf"))
        plt.close(fig)

    def generate_plots(self, case_study: CaseStudy, path: str, input_path: str) -> None:
        self.generate_scatterplots(case_study, path, input_path)
        self.generate_correlation_window_plot(case_study, path, input_path)
        self.generate_reg_error_dist(case_study, path, input_path)

    def finish(self, path: str) -> None:
        pass
