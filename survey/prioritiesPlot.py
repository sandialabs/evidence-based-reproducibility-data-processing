"""
Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
"""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn.cluster import KMeans
import scipy
import argparse

plot_size = 10
sns.set_theme(style="whitegrid")


def insert_spaces_before_uppercase(string: str):
    split_string = re.findall('[A-Z][^A-Z]*', string)
    if len(split_string) < 2:
        return string
    spaced_string = split_string[0]
    for i in range(1, len(split_string)):
        spaced_string += " " + split_string[i]
    return spaced_string


def outputVariablesToFile(variables, fileName="surveyDataVariables.tex"):
    with open(fileName, 'w') as variablesFile:
        for variableName in variables:
            variablesFile.write(
                "\\newcommand \\{varName}{{{varContent}}}\n".format(
                    varName=variableName, varContent=variables[variableName]
                )
            )


def floats_to_nice_strings(arr: np.array):
    return np.array([str(round(i, ndigits=2)) for i in arr])


if __name__ == "__main__":
    # parsing ARGS
    parser = argparse.ArgumentParser(description='Make more plots for presentations')
    parser.add_argument(
        'regularSurveyDataTransformed',
        type=str,
        help='The regularSurveyData file produced by transformSurveyData.py',
    )
    args = parser.parse_args()

    # reading in .csv and preprocessing
    df_survey = pd.read_csv(args.regularSurveyDataTransformed)
    df_survey = df_survey[df_survey["Project_Maturity"].notnull()]

    software_quality_strings = [
        i.split("_")[1]
        for i in df_survey.columns
        if i.startswith("QualityPriority_") and not i.endswith("_OriginalRank")
    ]
    sort_inds = np.flip(
        np.argsort(
            [
                np.sum(df_survey["QualityPriority_" + sq])
                for sq in software_quality_strings
            ]
        )
    )
    software_quality_strings_sorted = [software_quality_strings[i] for i in sort_inds]
    software_quality_strings_sorted_spaced = [
        insert_spaces_before_uppercase(sq) for sq in software_quality_strings_sorted
    ]

    # strip plot
    f, ax = plt.subplots(figsize=(16, 9))

    for i in range(len(software_quality_strings_sorted_spaced)):
        sq = software_quality_strings_sorted[i]

        sns.stripplot(
            x=[sq for i in range(len(df_survey))],
            y=df_survey["QualityPriority_" + sq],
            size=8,
            linewidth=0,
            alpha=0.5,
            label="",
        )

    ax.set(ylabel="Normalized Priority")
    # ax.set(ylim=(0,.6))
    plt.savefig("figures/qualityPriorityStripPlot.png", bbox_inches="tight")
    plt.clf()
    plt.close()

    # violin plot
    f, ax = plt.subplots(figsize=(16, 9))
    violin_xs = []
    violin_ys = np.array([])
    for i in range(len(software_quality_strings_sorted_spaced)):
        violin_xs += [
            software_quality_strings_sorted_spaced[i] for j in range(len(df_survey))
        ]
        violin_ys = np.append(
            violin_ys,
            df_survey["QualityPriority_" + software_quality_strings_sorted[i]],
        )

    sns.violinplot(x=violin_xs, y=violin_ys, label="", cut=0, inner="stick")

    ax.set(ylabel="Normalized Priority")
    ax.set(ylim=(0, 0.6))
    plt.savefig("figures/qualityPriorityViolinPlot.png", bbox_inches="tight")
    plt.clf()
    plt.close()

    # violin plot
    df2 = pd.DataFrame()
    for i in range(len(software_quality_strings_sorted_spaced)):
        sq = software_quality_strings_sorted[i]
        df2["QualityPriority_" + sq] = df_survey["QualityPriority_" + sq]
    df2["Project_Maturity"] = (df_survey["Project_Maturity"] - 1) / 3
    df2 = df2[df2["Project_Maturity"].notnull()]
    df2 = df2[[any(df2.values[i] != np.zeros(9)) for i in range(len(df2))]]

    holder = df2.values[:, :-1]
    n_nz = len(df2)

    def optimalK(data, nrefs=300, maxClusters=10):
        """
        Calculates KMeans optimal K using Gap Statistic
        Params:
            data: ndarry of shape (n_samples, n_features)
            nrefs: number of sample reference datasets to create
            maxClusters: Maximum number of clusters to test for
        Returns: (gaps, optimalK)
        """
        gaps = np.zeros((len(range(1, maxClusters)),))
        gap_stds = np.zeros((len(range(1, maxClusters)),))
        resultsdf = pd.DataFrame({'clusterCount': [], 'gap': [], 'gap_std': []})
        # Holder for reference dispersion results
        for gap_index, k in enumerate(range(1, maxClusters)):
            # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
            refDisps = np.zeros(nrefs)
            for i in range(nrefs):

                # Create new random reference set
                randomReference = np.random.random_sample(size=data.shape)

                # Fit to it
                km = KMeans(k, n_init=10)
                km.fit(randomReference)
                refDisps[i] = km.inertia_

            km = KMeans(k)
            km.fit(data)

            origDisp = km.inertia_  # Calculate gap statistic
            gap = np.mean(np.log(refDisps) - np.log(origDisp))
            gap_std = np.std(np.log(refDisps) - np.log(origDisp))
            gaps[gap_index] = gap
            gap_stds[gap_index] = gap_std

            resultsdf = resultsdf._append(
                {'clusterCount': k, 'gap': gap, 'gap_std': gap_std}, ignore_index=True
            )
            if gap_index > 0 and gaps[gap_index - 1] > gap - gap_std:
                return (len(resultsdf) - 1, resultsdf)
        return (len(resultsdf) - 1, resultsdf)

    nc = 0
    while nc < 3:  # something is buggy here but don't want to fix it
        nc, resultsdf = optimalK(holder, maxClusters=50)

    plt.plot(
        resultsdf['clusterCount'],
        resultsdf['gap'],
        linestyle='-',
        marker='o',
        color='b',
        label="gap",
    )
    plt.plot(
        resultsdf['clusterCount'],
        resultsdf['gap'] - resultsdf['gap_std'],
        linestyle='--',
        marker='o',
        color='b',
        label="gap - std",
    )
    plt.xlabel('K')
    plt.ylabel('Gap Statistic')
    plt.title('Gap Statistic vs. K')
    plt.savefig("figures/gap_statistic.png", bbox_inches="tight")
    print("optimal number of classes: " + str(nc))

    k_means = KMeans(init="k-means++", n_clusters=nc, n_init=100)
    k_means.fit(holder)

    class_size_ordering = np.flip(
        np.argsort([np.sum(k_means.labels_ == i) for i in range(nc)])
    )
    sort_classes = {class_size_ordering[i]: i for i in range(nc)}
    labels = np.array([sort_classes[i] for i in k_means.labels_])
    cluster_centers = k_means.cluster_centers_[class_size_ordering]

    # orange-yellow, red-pink,  blue-purple from IBM palette
    # https://davidmathlogic.com/colorblind/#%23648FFF-%23785EF0-%23DC267F-%23FE6100-%23FFB000
    colors = ["#FFB000", "#DC267F", "#648FFF"]
    lss = ["solid", "dashed", "dotted"]

    fig, axs = plt.subplots(1, 1, figsize=(18, 6))
    for i in range(nc):
        holder2 = holder[labels == i, :]
        lo = [np.quantile(holder2[:, j], 0.16) for j in range(8)]
        hi = [np.quantile(holder2[:, j], 0.84) for j in range(8)]
        axs.fill_between(range(8), lo, hi, alpha=0.15, color=colors[i])
    plt.xticks(rotation=35)

    for i in range(nc):
        ax = axs
        ax.set_ylabel("Priorities")
        # ax.set_title("Profile " + chr(65+i))
        ax.set_ylim((-0.03, 0.37))
        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            item.set_fontsize(25)
        ax.plot(
            software_quality_strings_sorted_spaced,
            cluster_centers[i],
            alpha=1,
            ls=lss[i],
            c=colors[i],
            lw=4,
        )

    plt.savefig('figures/Figure_3.png', bbox_inches="tight")

    # correlation analysis
    print("number of people in each class")
    print([np.sum(labels == i) for i in range(nc)])
    numeric_cols = [
        col
        for col in df_survey.columns
        if np.issubdtype(df_survey[col].dtype, np.number)
    ]
    in_class = [np.array([int(i == j) for i in labels]) for j in range(nc)]

    respondentProfileVariables = {}

    def pad_str(string: str):
        val = string + (" " * (0 - len(string)))
        return val

    round_digits = 4

    def swerve_zero(string):
        if string == "0.0":
            return "<1e-" + str(round_digits)
        else:
            return string

    for i in range(len(in_class)):
        print()
        for col in numeric_cols:
            filter = df_survey[col].notnull()
            tmp = df_survey[df_survey[col].notnull()]
            statistic, pvalue = scipy.stats.pearsonr(tmp[col], in_class[i][filter])
            if pvalue <= 0.05:
                respondentProfileVariables[
                    ("profile" + chr(65 + i) + col + "Correlation").replace('_', '')
                ] = swerve_zero(str(round(statistic, ndigits=round_digits)))
                respondentProfileVariables[
                    ("profile" + chr(65 + i) + col + "Pvalue").replace('_', '')
                ] = swerve_zero(str(round(pvalue, ndigits=round_digits)))

                if pvalue < 0.05 / 18:
                    print(
                        pad_str("X " + col + " x profile " + chr(65 + i) + ": ")
                        + " Pearson's r:",
                        str(round(statistic, ndigits=round_digits)),
                        "p-value",
                        str(round(pvalue, ndigits=round_digits)),
                    )
                else:
                    print(
                        pad_str(col + " x profile " + chr(65 + i) + ": ")
                        + " Pearson's r:",
                        str(round(statistic, ndigits=round_digits)),
                        "p-value",
                        str(round(pvalue, ndigits=round_digits)),
                    )
    outputVariablesToFile(
        respondentProfileVariables, fileName="tex/respondentDataVariables.tex"
    )

    df3 = pd.DataFrame()
    for sq in software_quality_strings_sorted:
        df3["QualityPriority_" + sq] = df_survey["QualityPriority_" + sq]
    most_relevant_index = df3.idxmax(axis=1)
    most_relevant = (
        100
        * np.array(
            [
                np.sum(most_relevant_index == "QualityPriority_" + sq)
                for sq in software_quality_strings_sorted
            ]
        )
        / len(most_relevant_index)
    )
    relevant = (
        np.array(
            [
                np.sum(df3["QualityPriority_" + sq] > 0)
                for sq in software_quality_strings_sorted
            ]
        )
        / len(df3)
        * 100
    )

    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(16, 9))

    colors = list(mcolors.TABLEAU_COLORS.keys())[: len(relevant)]

    for i in range(len(relevant)):
        plt.scatter(
            relevant[i],
            most_relevant[i],
            sizes=[500 for i in range(len(relevant))],
            c=colors[i],
        )
    ax.set_ylabel("Most Important for Project(%)")
    ax.set_xlabel("Was Relevant to Project (%)")
    ax.set_xlim((0, 100))
    ax.set_ylim((0, 40))
    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(25)

    txt_xs = [r - 1 for r in relevant]
    txt_ys = [r + 0.5 for r in most_relevant]
    txt_ys[-2] -= 1  # compatibility
    txt_xs[-2] -= 0.3  # performance
    txt_ys[-4] -= 2.5  # performance
    txt_xs[-4] += 5  # performance

    for i, txt in enumerate(software_quality_strings_sorted_spaced):
        ax.annotate(txt, (txt_xs[i], txt_ys[i]), size=20, ha="right", c=colors[i])

    plt.vlines([50], 0, 40, color="black", linestyles="--")

    plt.savefig('figures/software_priority_relevance.png', bbox_inches="tight")

    most_relevant = floats_to_nice_strings(most_relevant)
    relevant = floats_to_nice_strings(relevant)

    priorityVariables = {}
    for i in range(len(software_quality_strings_sorted)):
        sq = software_quality_strings_sorted[i]
        priorityVariables[sq + "RelevantPercentage"] = relevant[i]
        priorityVariables[sq + "MostRelevantPercentage"] = most_relevant[i]

    outputVariablesToFile(priorityVariables, fileName="tex/priorityDataVariables.tex")
