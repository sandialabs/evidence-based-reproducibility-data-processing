"""
Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
"""

import pandas as pd
import numpy as np
import math
import scipy
import argparse

n_software_qualities = 8

software_quality_strings = [
    'Compatibility',
    'FunctionalSuitability',
    'Maintainability',
    'PerformanceEfficiency',
    'Portability',
    'Reliability',
    'Security',
    'Usability',
]


def filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out the NaN MaxDiff responses, find and note the unchosen
    software quality, add columns for the index of each software quality
    in our software_quality_strings list
    """

    def filter(df: pd.DataFrame, column: str) -> pd.DataFrame:
        return df[df[column].notnull()]

    df = filter(df, 'MaxDiff_AttributeChoice1')
    df = filter(df, 'MaxDiff_WorstChoice')

    return df


def find_middle_and_parse(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find and note the unchosen software quality, add columns for the
    index of each software quality in our software_quality_strings list
    """

    def find_middle_choice(row):
        qualities_in_q = [
            row["MaxDiff_AttributeChoice{}".format(i)] for i in range(1, 4)
        ]
        qualities_in_q.remove(row['MaxDiff_WorstChoice'])
        qualities_in_q.remove(row['MaxDiff_BestChoice'])
        return qualities_in_q[0]

    df["MaxDiff_MiddleChoice"] = df.apply(find_middle_choice, axis=1)

    # column names with software quality strings
    column_names = [column_name for column_name in df.columns][-6:]
    column_names_index = [column_name + "_index" for column_name in column_names]

    software_quality_string_indices = {
        software_quality_strings[i]: i for i in range(n_software_qualities)
    }

    for i in range(len(column_names)):
        column_name = column_names[i]
        column_name_index = column_names_index[i]
        assert all([i in software_quality_strings for i in df[column_name]])
        df[column_name_index] = [
            software_quality_string_indices[i] for i in df[column_name]
        ]

    return df


# Convert the utilities for the answers of a given MaxDiff question into
# a log-likelihood
# see page 14 of The MaxDiff System Technical Paper by Sawtooth Software
# https://sawtoothsoftware.com/resources/technical-papers/maxdiff-technical-paper
def maxdiff_utility_log_likelihood_best(
    most_useful_utility: float,
    unranked_useful_utility: float,
    least_useful_utility: float,
) -> float:
    return most_useful_utility - np.log(
        np.exp(most_useful_utility)
        + np.exp(unranked_useful_utility)
        + np.exp(least_useful_utility)
    )


def maxdiff_utility_log_likelihood_worst(
    most_useful_utility: float,
    unranked_useful_utility: float,
    least_useful_utility: float,
) -> float:
    return -least_useful_utility - np.log(
        np.exp(-most_useful_utility)
        + np.exp(-unranked_useful_utility)
        + np.exp(-least_useful_utility)
    )


def maxdiff_utility_log_likelihood(
    most_useful_utility: float,
    unranked_useful_utility: float,
    least_useful_utility: float,
) -> float:
    log_likelihood_best = maxdiff_utility_log_likelihood_best(
        most_useful_utility, unranked_useful_utility, least_useful_utility
    )
    log_likelihood_worst = maxdiff_utility_log_likelihood_worst(
        most_useful_utility, unranked_useful_utility, least_useful_utility
    )
    return log_likelihood_best + log_likelihood_worst


# likelihood for an answer being the best or worst should be 1/3 if all
# 3 options have the same utility
test_utility = np.random.random()
assert math.isclose(
    maxdiff_utility_log_likelihood_best(test_utility, test_utility, test_utility),
    np.log(1 / 3),
)
assert math.isclose(
    maxdiff_utility_log_likelihood_worst(test_utility, test_utility, test_utility),
    np.log(1 / 3),
)
assert math.isclose(
    maxdiff_utility_log_likelihood(test_utility, test_utility, test_utility),
    np.log(1 / 9),
)


def maxdiff_utility_log_likelihood_worst_given_best(
    unranked_useful_utility: float, least_useful_utility: float
) -> float:
    return -least_useful_utility - np.log(
        np.exp(-unranked_useful_utility) + np.exp(-least_useful_utility)
    )


def maxdiff_utility_log_likelihood_conditional(
    most_useful_utility: float,
    unranked_useful_utility: float,
    least_useful_utility: float,
) -> float:
    log_likelihood_best = maxdiff_utility_log_likelihood_best(
        most_useful_utility, unranked_useful_utility, least_useful_utility
    )
    log_likelihood_worst = maxdiff_utility_log_likelihood_worst_given_best(
        unranked_useful_utility, least_useful_utility
    )
    return log_likelihood_best + log_likelihood_worst


# likelihood for an answer being the worst after the best has been
# removed should be 1/2 if they are all the same utility
assert math.isclose(
    maxdiff_utility_log_likelihood_worst_given_best(test_utility, test_utility),
    np.log(1 / 2),
)
# likelihood for any given ordering should be 1/3! if they are all the
# same utility
assert math.isclose(
    maxdiff_utility_log_likelihood_conditional(
        test_utility, test_utility, test_utility
    ),
    np.log(1 / 6),
)


# this value makes the average population utilities match the overall
L2_factor = 2e-1


def full_maxdiff_utility_log_likelihood(utilities: np.array, df: pd.DataFrame) -> float:
    """
    Compute the MaxDiff lig likelihood for every answered question.
    An L2 norm is added to prevent utilities from blowing up and
    preventing a degeneracy that stems from the fact that increasing all
    utilities together does not change the loss, so their absolute value
    is arbitrary
    """
    result = 0.0
    n_survey_responses = len(df)
    for i in range(n_survey_responses):
        df_row = df.iloc[i]
        most_useful_utility = utilities[df_row["MaxDiff_BestChoice_index"]]
        unranked_useful_utility = utilities[df_row["MaxDiff_MiddleChoice_index"]]
        least_useful_utility = utilities[df_row["MaxDiff_WorstChoice_index"]]
        result -= maxdiff_utility_log_likelihood_conditional(
            most_useful_utility, unranked_useful_utility, least_useful_utility
        )
    L2_norm = np.dot(utilities, utilities)
    return result + (L2_factor * L2_norm)


def get_optimal_utilities(df: pd.DataFrame, initial_guess: np.array) -> np.array:
    """
    Find the optimal MaxDiff utilities based on the answers in the
    passed pd.DataFrame.
    Could be sped up by finding the expression for analytical answer

    Args:
        df (pd.DataFrame): Contains MaxDiff answers
        initial_guess (np.array): initial utility score guesses

    Returns:
        np.array: optimized utility scores
    """

    # defining our loss function using our preprocessed DataFrame
    def loss(utilities: np.array) -> float:
        return full_maxdiff_utility_log_likelihood(utilities, df)

    n_survey_responses = len(df)
    assert math.isclose(
        loss(np.zeros(n_software_qualities)), n_survey_responses * -np.log(1 / 6)
    )
    test_utilities = np.random.random() * np.ones(n_software_qualities)
    assert math.isclose(
        loss(test_utilities),
        n_survey_responses * -np.log(1 / 6)
        + (L2_factor * np.dot(test_utilities, test_utilities)),
    )

    # getting optimal utilities
    optimization_result = scipy.optimize.minimize(loss, initial_guess)
    software_quality_utilities = optimization_result.x - np.mean(optimization_result.x)

    return software_quality_utilities


def optional_calculations(df: pd.DataFrame, software_quality_utilities: np.array):
    # based on this model, the likelihood that the given software
    # quality would be chosen when presented with two other choices
    set_size_minus_one = 3 - 1
    sum_software_likelihood_unscaled = np.sum(np.exp(software_quality_utilities))

    # # simpler version where you assume the average unscaled likelihood of other choices is either e^0 or the mean unscaled software quality likelihood
    # theoretical_in_random_set_probability = 100 * np.exp(software_quality_utilities) / (np.exp(software_quality_utilities) + set_size_minus_one)
    # theoretical_in_random_set_probability = 100 * np.exp(software_quality_utilities) / (np.exp(software_quality_utilities) + set_size_minus_one * sum_software_likelihood_unscaled / n_software_qualities)

    # more complicated version where you use the actual average unscaled likelihood of other choices (excluding the one being looked at)
    theoretical_in_random_set_probability_advanced = 100 * np.array(
        [
            np.exp(software_quality_utilities[i])
            / (
                set_size_minus_one
                * (
                    sum_software_likelihood_unscaled
                    - np.exp(software_quality_utilities[i])
                )
                / (n_software_qualities - 1)
                + np.exp(software_quality_utilities[i])
            )
            for i in range(n_software_qualities)
        ]
    )

    # rescaling the in set probabilities to sum to 100
    scaled_theoretical_in_set_probability = (
        100
        * theoretical_in_random_set_probability_advanced
        / sum(theoretical_in_random_set_probability_advanced)
    )

    # percentage time that the given software quality was chosen when presented with two other choices
    empirical_in_random_set_probability = 100 * np.array(
        [
            sum(df["MaxDiff_BestChoice_index"] == i)
            / sum(
                (df["MaxDiff_AttributeChoice1_index"] == i)
                | (df["MaxDiff_AttributeChoice2_index"] == i)
                | (df["MaxDiff_AttributeChoice3_index"] == i)
            )
            for i in range(n_software_qualities)
        ]
    )

    # rescaling the in set probabilities to sum to 100
    scaled_empirical_in_random_set_probability = (
        empirical_in_random_set_probability
        * 100
        / sum(empirical_in_random_set_probability)
    )

    return (
        theoretical_in_random_set_probability_advanced,
        scaled_theoretical_in_set_probability,
        empirical_in_random_set_probability,
        scaled_empirical_in_random_set_probability,
    )


def save_1D_utility_DataFrame(
    software_quality_strings: list, software_quality_utilities: np.array, output_fn: str
):
    df = pd.DataFrame()
    df["QualityAttribute"] = software_quality_strings
    df["Utility"] = software_quality_utilities
    df.to_csv(output_fn)


if __name__ == "__main__":
    # parsing ARGS
    parser = argparse.ArgumentParser(
        description='Process MaxDiff results .csv and compute the software quality utilities'
    )
    parser.add_argument(
        'survey_csv_fn',
        metavar='survey_fn',
        type=str,
        nargs=1,
        help='The .csv with the TRANSFORMED survey results, maybe something like regularSurveyData_transformed.csv',
    )
    parser.add_argument(
        'maxdiff_csv_fn',
        metavar='maxdiff_fn',
        type=str,
        nargs=1,
        help='The .csv with the TRANSFORMED MaxDiff results, maybe something like maxDiffSurveyData_transformed.csv',
    )
    args = parser.parse_args()

    # reading in .csv's and preprocessing
    df_survey = pd.read_csv(args.survey_csv_fn[0])
    df_maxdiff = pd.read_csv(args.maxdiff_csv_fn[0])
    df_maxdiff = filter(df_maxdiff)
    df_maxdiff = find_middle_and_parse(df_maxdiff)

    # performing MaxDiff analysis on overall population
    print("performing MaxDiff analysis on overall population")
    software_quality_utilities = get_optimal_utilities(
        df_maxdiff, np.zeros(n_software_qualities)
    )
    save_1D_utility_DataFrame(
        software_quality_strings,
        software_quality_utilities,
        "data/software_quality_utilities.csv",
    )
    # optional_calculations(df_maxdiff, software_quality_utilities)

    # performing MaxDiff analysis on each person
    print("performing MaxDiff analysis on each person")
    respondents = np.unique(df_maxdiff["ResponseID"])

    def get_optimal_utilities_per_person(respondent: int) -> np.array:
        software_quality_utilities = get_optimal_utilities(
            df_maxdiff[df_maxdiff["ResponseID"] == respondent],
            np.zeros(n_software_qualities),
        )
        return software_quality_utilities

    software_quality_utilities_per_person = np.array(
        [get_optimal_utilities_per_person(respondent) for respondent in respondents]
    )
    df_each_person = pd.DataFrame(
        software_quality_utilities_per_person, columns=software_quality_strings
    )
    df_each_person["ResponseID"] = respondents
    df_each_person.to_csv("data/software_quality_utilities_per_person.csv", index=False)

    # performing MaxDiff analysis splitting by whether or not any given software quality was prioritized
    print(
        "performing MaxDiff analysis splitting by whether or not any given software quality was prioritized (takes a while)"
    )
    software_quality_prioritized_utilities = np.zeros(
        [n_software_qualities, n_software_qualities]
    )
    software_quality_not_prioritized_utilities = np.zeros(
        [n_software_qualities, n_software_qualities]
    )
    for i in range(len(software_quality_strings)):
        sq = software_quality_strings[i]
        df_survey_filter = df_survey["QualityPriority_" + sq] > 0
        df_filter = df_maxdiff["ResponseID"].isin(
            df_survey["ResponseID"][df_survey_filter]
        )

        software_quality_prioritized_utilities[i, :] = get_optimal_utilities(
            df_maxdiff[df_filter], np.zeros(n_software_qualities)
        )
        software_quality_not_prioritized_utilities[i, :] = get_optimal_utilities(
            df_maxdiff[~df_filter], np.zeros(n_software_qualities)
        )

    df_only_prioritized = pd.DataFrame(
        software_quality_prioritized_utilities, columns=software_quality_strings
    )
    df_only_prioritized["QualityAttributePriority"] = software_quality_strings
    df_not_prioritized = pd.DataFrame(
        software_quality_not_prioritized_utilities, columns=software_quality_strings
    )
    df_not_prioritized["QualityAttributePriority"] = [
        "no" + sq for sq in software_quality_strings
    ]
    df_prioritized = pd.concat([df_only_prioritized, df_not_prioritized])
    df_prioritized.to_csv(
        "data/software_quality_prioritized_utilities.csv", index=False
    )

    # performing MaxDiff analysis splitting by whether how much they care about reproducibility
    print(
        "performing MaxDiff analysis splitting by how much they care about reproducibility"
    )
    repro_opinion = (
        df_survey["Belief_HighQualitySoftwareIsMoreReproducible"]
        + df_survey["Belief_ReproducibilityIsMoreImportantThanSpeedOrFeatures"]
        + df_survey["Belief_WillingToInvestExtraInQuality"]
        + df_survey["Belief_WillingToSetUpComplexSolutions"]
        + df_survey["Belief_WillingToSacrificeQuality"]
    )
    nan_filter = [i for i in range(len(repro_opinion)) if ~np.isnan(repro_opinion[i])]
    repro_opinion_filtered = repro_opinion[nan_filter]
    opinion_threshold = 4
    low_repro_opinion_filter = df_maxdiff["ResponseID"].isin(
        df_survey["ResponseID"][nan_filter][repro_opinion_filtered <= opinion_threshold]
    )
    high_repro_opinion_filter = df_maxdiff["ResponseID"].isin(
        df_survey["ResponseID"][nan_filter][repro_opinion_filtered > opinion_threshold]
    )
    low_repro_opinion_utilities = get_optimal_utilities(
        df_maxdiff[low_repro_opinion_filter], np.zeros(n_software_qualities)
    )
    high_repro_opinion_utilities = get_optimal_utilities(
        df_maxdiff[high_repro_opinion_filter], np.zeros(n_software_qualities)
    )
    save_1D_utility_DataFrame(
        software_quality_strings,
        low_repro_opinion_utilities,
        "data/low_repro_opinion_utilities.csv",
    )
    save_1D_utility_DataFrame(
        software_quality_strings,
        high_repro_opinion_utilities,
        "data/high_repro_opinion_utilities.csv",
    )
