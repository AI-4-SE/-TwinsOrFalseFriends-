#!/bin/env python3
import sys
import os
from typing import List, Type
from case_study import CaseStudy
from research_question import ResearchQuestion
from research_question_1 import ResearchQuestion1
from research_question_2 import ResearchQuestion2

FM = "FeatureModel.xml"
Measurements = "measurements.csv"
Alternative = "selected_alternative.txt"
Results_File = "README_POST.md"

Research_Questions: List[ResearchQuestion] = [
                                                    # ResearchQuestion1(),
                                                    ResearchQuestion2()
                                                   ]


def print_usage() -> None:
    """
    Prints the usage of the python script.
    """
    print("Usage: generate_plots.py <InputPath> <OutputPath>")
    print("InputPath\t The path to the directory containing all relevant information of the case studies.")
    print("OutputPath\t The path to the directory where all plots should be exported to.")


def list_directories(path: str) -> List:
    """
    Returns the subdirectories of the given path.
    :param path: the path to find the subdirectories from.
    :return: the subdirectories as list.
    """
    for root, dirs, files in os.walk(path):
        return list(filter(lambda x: not x.startswith("."), dirs))


def create_directory(path: str) -> None:
    """
    Creates the given directory if it does not exist already.
    :param path: the path to create
    """
    if not os.path.exists(path):
        os.makedirs(path)


def main() -> None:
    """
    The main method reads in the data of the case studies and evaluates the data with regard to the different
    research questions (1-4) of the study.
    """
    if len(sys.argv) != 3:
        print_usage()
        exit(0)

    # Read in the path to the case study data
    input_path = sys.argv[1]

    # Read in the output path of the plots
    output_path = sys.argv[2]

    # Be aware: The file structure is as follows: RQ -> Case Study [-> Strategy] -> <Plot>.pdf

    case_studies = sorted(list_directories(input_path), key=str.casefold)
    print("Progress:")
    i = -1

    for rq in Research_Questions:
        rq.initialize_for_metrics(os.path.join(output_path, rq.name))

    for case_study in case_studies:
        i += 1
        print(case_study + " (" + str(int((float(i) / len(case_studies)) * 100)) + "%)")
        if not os.path.exists(os.path.join(input_path, case_study, FM)):
            print("Skipping case study because feature model is missing.")
            continue
        # Read in one case study (i.e., its FM and measurements) after another (and wipe the data to save some RAM)
        cs = CaseStudy(case_study, os.path.join(input_path, case_study, FM),
                       os.path.join(input_path, case_study, Measurements),
                       os.path.join(input_path, case_study, Alternative))

        rq_count = 0
        for rq in Research_Questions:
            rq_count += 1
            print("\t" + rq.get_name() + "...", end="")
            sys.stdout.flush()
            rq.prepare(cs, input_path)
            rq.evaluate_metrics(cs, os.path.join(output_path, rq.get_name()), input_path)
            rq.generate_plots(cs, os.path.join(output_path, rq.get_name(), case_study), input_path)
            print("Finished!")

    for rq in Research_Questions:
        rq.finish(os.path.join(output_path, rq.name))


if __name__ == "__main__":
    main()
