from typing import Tuple, Any, List

import pandas as pd
import scipy.stats

from case_study import CaseStudy
import os
from abc import ABC, abstractmethod


class ResearchQuestion(ABC):
    name = ""

    @staticmethod
    def create_directory(path: str) -> None:
        """
        Creates the given directory if it does not exist already.
        :param path: the path to create
        """
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def comp_regression_error(case_study: CaseStudy, to_learn: pd.DataFrame, to_predict: pd.DataFrame) -> pd.DataFrame:
        slope, intercept = ResearchQuestion.perform_linear_regression(case_study, to_learn)
        to_predict['Lin Err'] = to_predict.apply(lambda row: ResearchQuestion.calc_lin_err(row[case_study.Performance],
                                                                           row[case_study.Energy], slope, intercept),
                                 axis=1)
        to_predict['Lin Err Norm'] = to_predict['Lin Err'] / to_predict['energy'] * 100
        return to_predict

    @staticmethod
    def perform_linear_regression(case_study: CaseStudy, df: pd.DataFrame) -> Tuple[Any, Any]:
        x = df[case_study.Performance].to_numpy()
        y = df[case_study.Energy].to_numpy()

        slope, intercept, _, _, _ = scipy.stats.linregress(x, y)
        return slope, intercept

    @staticmethod
    def calc_lin_err(x, y, slope, intercept) -> float:
        return abs(y - (intercept + (slope * x)))

    @staticmethod
    def initialize_tables(path: str, column_names: List[str], only_csv: bool = False) -> None:
        with open(f"{path}.csv", 'w') as output_file:
            output_file.write(",".join(column_names))
            output_file.write("\n")

        if only_csv:
            return
        with open(f"{path}.md", 'w') as output_file:
            output_file.write("| " + " | ".join(column_names) + " |")
            output_file.write("\n")
            output_file.write("|" + "  :-----: |" * len(column_names))
            output_file.write("\n")

    @staticmethod
    def write_to_tables(path: str, column_values: List[Any]) -> None:
        with open(f"{path}.csv", 'a') as output_file:
            output_file.write(",".join(column_values))
            output_file.write("\n")
        with open(f"{path}.md", 'a') as output_file:
            output_file.write("| " + " | ".join(column_values) + "|")
            output_file.write("\n")

    def get_name(self):
        return self.name

    @abstractmethod
    def initialize_for_metrics(self, path: str):
        pass

    @abstractmethod
    def prepare(self, case_study: CaseStudy, input_path: str) -> None:
        pass

    @abstractmethod
    def evaluate_metrics(self, case_study: CaseStudy, path: str, input_path: str) -> None:
        pass

    @abstractmethod
    def generate_plots(self, case_study: CaseStudy, path: str, input_path: str) -> None:
        pass

    @abstractmethod
    def finish(self, path: str) -> None:
        pass
