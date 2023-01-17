from feature import Feature
from typing import List
import pandas as pd
import xml.etree.ElementTree as ET
import os


class CaseStudy:
    Revision = "revision"
    Workload = "workload"
    Performance = "performance"
    Energy = "energy"

    Case_Studies_In_Milliseconds = ["7z", "exastencils", "LLVM", "lrzip", "MongoDB", "PostgreSQL", "VP8"]
    Case_Studies_In_KiloJoule = ["HSQLDB", "x264"]

    def __init__(self, name: str, feature_model_path: str, measurements_path: str, alternative_path: str) -> None:
        self.windows = []
        self.features = dict()
        self.configurations = None
        self.deviations = None
        self.name = name
        self.alternatives = []
        self.configurations_by_alternating_column = dict()
        self.read_feature_model(feature_model_path)
        self.read_measurements(measurements_path)

    @staticmethod
    def get_options(node) -> List:
        result = []
        for option_node in node:
            result.append(option_node.text)
        return result

    def read_feature_model(self, path: str) -> None:
        with open(path, 'r') as feature_model_file:
            root = ET.parse(feature_model_file).getroot()

            binary_options = root.find('binaryOptions')
            numeric_options = root.find('numericOptions')

            children_relation = dict()

            # Parse binary options
            for binary_option in binary_options:
                name = binary_option.find('name').text
                mandatory = binary_option.find('optional').text.lower() == 'false'

                parent = binary_option.find('parent').text
                if parent is None and name != "root":
                    parent = "root"
                elif parent is not None:
                    parent = parent.replace("\n", "")
                    parent = parent.replace(" ", "")
                    if parent == "":
                        if name != "root":
                            parent = "root"
                        else:
                            parent = None

                excluded_options = self.get_options(binary_option.find('excludedOptions'))
                implied_options = self.get_options(binary_option.find('impliedOptions'))
                self.features[name] = Feature(name, parent, excluded_options, implied_options, mandatory)
                if name != "root" and parent not in children_relation:
                    children_relation[parent] = []
                if name != "root":
                    children_relation[parent].append(name)

            # Parse numeric options
            for numeric_option in numeric_options:
                name = numeric_option.find('name').text
                parent = numeric_option.find('parent').text
                excluded_options = self.get_options(binary_option.find('excludedOptions'))
                implied_options = self.get_options(binary_option.find('impliedOptions'))
                self.features[name] = Feature(name, parent, excluded_options, implied_options)

            # Add children
            for feature_name in children_relation.keys():
                self.features[feature_name].children = children_relation[feature_name]

            # Add alternatives
            for feature_name in self.features.keys():
                if self.is_alternative_group(feature_name):
                    self.features[feature_name].alternative_parent = True
                    for child in self.features[feature_name].children:
                        other_children = self.features[feature_name].children.copy()
                        other_children.remove(child)
                        self.features[child].alternatives = other_children

            # Decide whether the features are strictly mandatory
            for feature_name in self.features.keys():
                self.features[feature_name].strictly_mandatory = self.is_strictly_mandatory(feature_name)

    def is_alternative_group(self, feature_name: str) -> bool:
        """
        This method checks whether the given feature is the parent of an alternative group or not. An alternative
        group contains only mandatory features and each of the features excludes all others.
        :param feature_name: the name of the parent node
        :return: <code>True</code> iff the given feature is an alternative group
        """
        if feature_name not in self.features.keys():
            print(f"Feature {feature_name} not in feature list!")
            exit(-1)
        feature: Feature = self.features[feature_name]
        if len(feature.children) <= 1:
            return False
        for child in feature.children:
            child_feature: Feature = self.features[child]
            if not child_feature.mandatory:
                return False
            for other_child in feature.children:
                if other_child == child:
                    continue
                if other_child not in child_feature.exclusions:
                    return False
        return True

    def is_strictly_mandatory(self, feature_name: str) -> bool:
        """
        This method decides whether the given feature is strictly mandatory or not.
        :param feature_name: the feature to check
        :return: <code>true</code> iff the feature and all its parents are mandatory and do not belong to an alternative
         group.
        """
        if feature_name not in self.features.keys():
            print(f"Feature {feature_name} not in feature list!")
            exit(-1)
        if feature_name == "root":
            return True
        feature: Feature = self.features[feature_name]
        # A numerical feature (mandatory = None) is also strictly mandatory
        if not feature.binary:
            return True
        if not feature.mandatory:
            return False
        if len(feature.alternatives) > 0:
            return False
        parent = self.features[feature.parent]
        while parent.name != "root":
            if not parent.mandatory or len(parent.exclusions) > 0:
                return False
            parent = self.features[parent.parent]
        return True

    def get_all_feature_names(self) -> List[str]:
        """
        This method returns a list containing all feature names
        :return a list containing all feature names
        """
        feature_names = []
        for feature in self.features:
            feature_names.append(self.features[feature].name)
        return feature_names

    def read_measurements(self, path: str) -> None:
        with open(path, 'r') as measurements_file:
            self.configurations = pd.read_csv(measurements_file, sep=';', lineterminator='\n', dtype=str)
        self.configurations[self.Performance] = pd.to_numeric(self.configurations[self.Performance])
        self.configurations[self.Energy] = pd.to_numeric(self.configurations[self.Energy])
        if self.name in self.Case_Studies_In_Milliseconds:
            self.configurations[self.Performance] = self.configurations[self.Performance] / 1000
        if self.name not in self.Case_Studies_In_KiloJoule:
            self.configurations[self.Energy] = self.configurations[self.Energy] / 1000
        # Remove revision and workload column
        if self.Revision in self.configurations.columns:
            self.configurations.drop(columns=[self.Revision])
        elif self.Workload in self.configurations.columns:
            self.configurations.drop(columns=[self.Workload])

        self.create_windows()

    def create_windows(self) -> None:
        window_pct = 0.1
        stepsize = 0.5

        # Analyze the performance data and declare different ranges for clusters where the configurations' performance
        # is higher than a given threshold.

        current_configurations: pd.DataFrame = self.configurations.sort_values(by=[self.Performance])
        # Find out the minimum and maximum value of the performance
        max_perf = current_configurations[self.Performance].max()
        min_perf = current_configurations[self.Performance].min()

        value_range = max_perf - min_perf
        window_width = value_range * window_pct
        current_perf = min_perf

        while current_perf < max_perf - window_width * stepsize - 1:
            window_df = current_configurations[(current_configurations[self.Performance] >= current_perf) &
                                               (current_configurations
                                                [self.Performance] <= current_perf + window_width)]

            if len(window_df) > 2:
                self.windows.append((window_df, current_perf, current_perf + window_width))
            current_perf = current_perf + window_width * stepsize
