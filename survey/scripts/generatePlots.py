"""
Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
"""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import argparse
import plot_likert
import re
import numpy as np
import os

plot_size = 10
sns.set_theme(style="whitegrid")
likert_plot_color_scheme = [
    plot_likert.colors.TRANSPARENT,
    "#A9A9A9",
    "#0B1354",
    "#165BAA",
    "#F765A3",
    "#FFA4B6",
    "#F9D1D1",
]


def insert_spaces_before_uppercase(string: str):
    split_string = re.findall('[A-Z][^A-Z]*', string)
    if len(split_string) < 2:
        return string
    spaced_string = split_string[0]
    for i in range(1, len(split_string)):
        spaced_string += " " + split_string[i]
    return spaced_string


def proportion_bar_chart(
    df: pd.DataFrame, x_levels: list, df_col: str, x_label: str, output_filename: str
):
    df[df_col] = pd.Categorical(df[df_col], x_levels)
    plt.figure(figsize=(plot_size, plot_size))
    # plt.ylabel("Utility")
    plt.xlabel(x_label)
    ax = sns.histplot(df_survey, x=df_col, stat="proportion")
    props = [sum(df[df_col] == x) / len(df) for x in x_levels]
    for i, prop in enumerate(props):
        ax.text(x_levels[i], prop + 0.005, str(round(prop, ndigits=3)), ha="center")
    plt.savefig(output_filename, bbox_inches="tight")


def likert_plot(
    df: pd.DataFrame, column_names: list, plot_labels: list, output_filename: str
):
    df_survey = df[column_names]
    df_survey = df_survey.rename(
        columns={column_names[i]: plot_labels[i] for i in range(len(column_names))}
    )
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bar_scale = [
        'Strongly Disagree',
        'Disagree',
        'Neither Agree Nor Disagree',
        'Agree',
        'Strongly Agree',
    ]
    plot_likert.plot_likert(
        df_survey,
        bar_scale,
        ax=ax,
        colors=likert_plot_color_scheme,
        figsize=(0.5 * plot_size, plot_size),
        bar_labels=False,
        plot_percentage=True,
        align='edge',
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(True)

    # plt.show()
    plt.savefig(output_filename, bbox_inches="tight")


def box_and_whisker_plot(
    df_survey: pd.DataFrame,
    df_each_person: pd.DataFrame,
    df_prioritized: pd.DataFrame,
    software_quality_strings_sorted: list,
    software_quality_strings_sorted_spaced: list,
    output_fn: str,
    show_details: bool,
):
    f, ax = plt.subplots(figsize=(16, 9))
    ax.set(ylabel="Utility")

    sns.boxplot(data=df_each_person, whis=[0, 100], width=0.6, color="#B5B5B5")

    for i in range(len(software_quality_strings_sorted_spaced)):
        sq = software_quality_strings_sorted[i]
        sq_spaced = software_quality_strings_sorted_spaced[i]
        no_sq = "no" + sq
        df_survey_filter = (
            df_survey["QualityPriority_" + software_quality_strings_sorted[i]] > 0
        )

        if show_details:
            if i == 0:
                labels = ["Not prioritized", "Prioritized"]
            else:
                labels = ["", ""]

            df_each_person_filter = df_each_person.index.isin(
                df_survey["ResponseID"][df_survey_filter]
            )

            sns.stripplot(
                x=[sq_spaced for i in range(sum(df_each_person_filter))],
                y=df_each_person[sq_spaced][df_each_person_filter],
                size=4,
                color="#0C7BDC",  # blue, from https://davidmathlogic.com/colorblind/
                linewidth=0,
                alpha=0.4,
                label=labels[1],
            )
            sns.stripplot(
                x=[sq_spaced for i in range(sum(~df_each_person_filter))],
                y=df_each_person[sq_spaced][~df_each_person_filter],
                size=4,
                color="#FFC20A",  # orange, from https://davidmathlogic.com/colorblind/
                linewidth=0,
                alpha=0.4,
                label=labels[0],
            )
        else:
            prioritized_proportion = np.sum(df_survey_filter) / len(df_survey)
            assert prioritized_proportion <= 1

            sns.scatterplot(
                x=[sq_spaced],
                y=[df_prioritized[sq].loc[sq]],
                color="#0C7BDC",  # blue, from https://davidmathlogic.com/colorblind/
                label="",
                s=prioritized_proportion * 1000,
                linewidth=1,
                edgecolors='#686868',  # color of box plot border
                alpha=1,
                zorder=10,
            )
            sns.scatterplot(
                x=[sq_spaced],
                y=[df_prioritized[sq].loc[no_sq]],
                color="#FFC20A",  # orange, from https://davidmathlogic.com/colorblind/
                label="",
                s=(1 - prioritized_proportion) * 1000,
                linewidth=1,
                edgecolors='#686868',  # color of box plot border
                alpha=1,
                zorder=10,
            )
    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(25)
    plt.xticks(rotation=45)

    plt.savefig(output_fn, bbox_inches="tight")


if __name__ == "__main__":
    # parsing ARGS
    parser = argparse.ArgumentParser(description='Make plots for presentations')
    parser.add_argument(
        'survey_csv_fn',
        metavar='survey_fn',
        type=str,
        nargs=1,
        help=(
            'The .csv with the NON-TRANSFORMED survey results, maybe '
            + 'something like regularSurveyData.csv'
        ),
    )
    parser.add_argument(
        'utils_fn',
        metavar='utils_fn',
        type=str,
        nargs=1,
        help=(
            'The .csv with the overall MaxDiff utility estimates. '
            + 'Probably software_quality_utilities.csv, an output of '
            + 'analyzeMaxDiffUtilities.py'
        ),
    )
    parser.add_argument(
        'utils_each_person_fn',
        metavar='utils_each_person_fn',
        type=str,
        nargs=1,
        help=(
            'The .csv with the MaxDiff utility estimates for each'
            + ' person. Probably '
            + 'software_quality_utilities_per_person.csv, an output'
            + ' of analyzeMaxDiffUtilities.py'
        ),
    )
    parser.add_argument(
        'utils_prioritized_fn',
        metavar='utils_prioritized_fn',
        type=str,
        nargs=1,
        help=(
            "The .csv with the MaxDiff utility estimates based on"
            + " people who did, or didn't, care about any given"
            + "software quality. Probably "
            + "software_quality_prioritized_utilities.csv, an "
            + "output of analyzeMaxDiffUtilities.py"
        ),
    )

    args = parser.parse_args()

    os.makedirs("figures", exist_ok=True)

    # reading in .csv and preprocessing
    df_survey = pd.read_csv(args.survey_csv_fn[0])

    # Years of experience plot
    year_levels = [
        "Less than a year",
        "1-5 years",
        "6-10 years",
        "11-15 years",
        "16-20 years",
        "20+ years",
    ]
    proportion_bar_chart(
        df_survey,
        year_levels,
        "YearsOfExperience",
        "Years of experience",
        'figures/ldrd_years_of_exp.png',
    )

    # Level of education plot
    # data cleaning
    df_survey = df_survey.replace("2 master's degrees", "Master's degree")
    df_survey = df_survey.replace(
        "Pursuing Bachelors", "High school diploma or equivalent"
    )
    df_survey = df_survey.replace("Post Doc", 'Doctorate degree (Ph.D., MD, JD, etc.)')

    education_levels = [
        'High school',
        "Associate's",
        "Bachelor's",
        "Master's",
        'Doctorate',
    ]

    # making labels shorter
    df_survey = df_survey.replace(
        "Associate's degree (e.g., community college degree)", education_levels[1]
    )
    df_survey = df_survey.replace("Bachelor's degree", education_levels[2])
    df_survey = df_survey.replace(
        "Doctorate degree (Ph.D., MD, JD, etc.)", education_levels[4]
    )
    df_survey = df_survey.replace(
        "High school diploma or equivalent", education_levels[0]
    )
    df_survey = df_survey.replace("Master's degree", education_levels[3])

    proportion_bar_chart(
        df_survey,
        education_levels,
        "LevelOfEducation",
        "Level of education",
        'figures/ldrd_level_of_education.png',
    )

    # Software experience plot
    experience_levels = [
        "No experience",
        "Limited understanding",
        "Moderate level",
        "Significant amount",
        "Highly experienced",
    ]

    # making labels shorter
    df_survey = df_survey.replace(
        (
            "I am highly experienced and could mentor others in software "
            + "development."
        ),
        experience_levels[4],
    )
    df_survey = df_survey.replace(
        (
            "I have a limited understanding and would need assistance with "
            + "software development tasks."
        ),
        experience_levels[1],
    )
    df_survey = df_survey.replace(
        (
            "I have a moderate level of experience and can complete basic tasks"
            + " independently."
        ),
        experience_levels[2],
    )
    df_survey = df_survey.replace(
        (
            "I have a significant amount of experience and can handle complex "
            + "tasks."
        ),
        experience_levels[3],
    )
    df_survey = df_survey.replace(
        "I have no experience or knowledge in software development.",
        experience_levels[0],
    )

    # taking out nans
    df_no_nan = df_survey[~(df_survey['SoftwareDevelopmentExperience'].isna())]

    proportion_bar_chart(
        df_no_nan,
        experience_levels,
        "SoftwareDevelopmentExperience",
        "Software experience",
        'figures/ldrd_software_exp.png',
    )

    # Reproducibility beliefs plots
    belief_columns = [
        'Belief_ReproducibilityIsFundamental',
        'Belief_ReproducibilityEnablesProgress',
        'Belief_HighQualitySoftwareIsMoreReproducible',
        'Belief_ReproducibilityIsMoreImportantThanSpeedOrFeatures',
        'Belief_WillingToInvestExtraInQuality',
        'Belief_WillingToSetUpComplexSolutions',
        'Belief_WillingToSacrificeQuality',
    ]
    belief_labels = [
        'Reproducibility is fundamental',
        'Reproducibility enables scientific progress',
        'High quality software is more reproducible',
        'Reproducibility is more important than speed or features',
        'Would invest extra effort in software quality for reproducibility',
        'Would invest extra effort setting up complex solutions',
        'Would sacrifice some quality for reproducibility',
    ]
    likert_plot(
        df_survey,
        belief_columns,
        belief_labels,
        "figures/ldrd_reproducibility_beliefs.png",
    )

    # Perceived competencies plots
    competency_names = [
        'Belief_AwareOfBestPractices',
        'Belief_HasKnowledgeToolsAndResources',
        'Belief_CanTakeActionOnReproducibility',
        'Belief_ProjectHasEffectivePractices',
        'Belief_ReproducibilityNotAHindrance',
    ]
    competency_labels = [
        'Aware of reproducibility best practices',
        'Have knowledge, resources, and tools needed for reproducibility',
        'Can identify and act on opportunities to improve reproducibility',
        'Project has implemented effective reproducibility practices',
        'Reproducibility requirements do not hinder productivity',
    ]
    likert_plot(
        df_survey,
        competency_names,
        competency_labels,
        "figures/ldrd_perceived_competencies.png",
    )

    # Organizational support plots
    orgsupport_names = [
        'Belief_CommunityValuesReproducibility',
        'Belief_InstitutionValuesReproducibility',
        'Belief_StakeholdersValueReproducibility',
        'Belief_CanLearnFromPeers',
        'Belief_NeedForStandardGuidelines',
        'Belief_NeedForTrainingOpportunities',
    ]
    orgsupport_labels = [
        'Community values reproducibility',
        'Institution values reproducibility',
        'Stakeholders understand resources required to ensure reproducibility',
        'Collaborate with peers to learn about reproducibility best practices',
        'Guidelines for reproducibility should be standardized',
        'Should be training on reproducibility best practices',
    ]
    likert_plot(
        df_survey,
        orgsupport_names,
        orgsupport_labels,
        "figures/ldrd_organizational_support.png",
    )

    # Box and whisker plots
    df_total = pd.read_csv(args.utils_fn[0], index_col="QualityAttribute")
    df_each_person = pd.read_csv(args.utils_each_person_fn[0], index_col="ResponseID")
    df_prioritized = pd.read_csv(
        args.utils_prioritized_fn[0], index_col="QualityAttributePriority"
    )

    # reordering df_each_person so that the columns are sorted by
    # usefulness for reproducibility and have plot-worthy labels
    software_quality_strings_sorted = list(
        (df_total.sort_values("Utility", ascending=False)).index
    )
    df_each_person = df_each_person[software_quality_strings_sorted]
    software_quality_strings_sorted_spaced = [
        insert_spaces_before_uppercase(sq) for sq in software_quality_strings_sorted
    ]
    df_each_person = df_each_person.rename(
        columns={
            software_quality_strings_sorted[i]: (
                software_quality_strings_sorted_spaced[i]
            )
            for i in range(len(software_quality_strings_sorted))
        }
    )

    box_and_whisker_plot(
        df_survey,
        df_each_person,
        df_prioritized,
        software_quality_strings_sorted,
        software_quality_strings_sorted_spaced,
        "figures/Figure_2.png",
        False,
    )
    box_and_whisker_plot(
        df_survey,
        df_each_person,
        df_prioritized,
        software_quality_strings_sorted,
        software_quality_strings_sorted_spaced,
        "figures/ldrd_box_and_whisker.png",
        True,
    )
