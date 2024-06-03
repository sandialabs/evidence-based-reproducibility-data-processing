"""
Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
"""

import argparse
import pandas as pd
import numpy as np
import scipy.stats

# from matplotlib import pyplot as plt
import os

import warnings

warnings.simplefilter(action='ignore')


def correlationMiningTest(
    regularSurveyData,
    statisticThreshold,
    levelOfSignificanceThreshold,
    filterByIndices=None,
    bonferroniCorrectionFactor=None,
):
    def r_pvalues(df):
        df[df['CanThinkOfProject'] == "Yes, I can think of a project."]
        cols = pd.DataFrame(columns=df.columns)
        p = cols.transpose().join(cols, how='outer')
        for r in df.columns:
            for c in df.columns:
                tmp = df[df[r].notnull() & df[c].notnull()]
                try:
                    statistic, pvalue = scipy.stats.pearsonr(tmp[r], tmp[c])
                    p[r][c] = (statistic, pvalue)
                except Exception:
                    p[r][c] = (0.0, 9999.0)
        return p

    p = r_pvalues(regularSurveyData)
    for r in p.columns:
        for c in p.columns:
            statistic, pvalue = p[r][c]
            if r == c:
                continue
            if filterByIndices is not None and (
                r not in filterByIndices or c not in filterByIndices
            ):
                continue
            if (
                pvalue <= levelOfSignificanceThreshold
                and abs(statistic) >= statisticThreshold
            ):
                if (
                    bonferroniCorrectionFactor is not None
                    and pvalue <= statisticThreshold / bonferroniCorrectionFactor
                ):
                    print("!!! ", end='')
                print(r, c, "Pearson's r:", statistic, "p-value", pvalue)


def testWhetherMaxDiffChoicesMerelyReflectRespondentsQualityPriorities(
    mergedSurveyData,
):
    """
    Here we simulate picking answers to the max-diff portion based solely on their quality priorites,
    (that is, best choice is their top-ranked priority and worst choice is their lowest-ranked priority,
    and we see how those compare to their actual answers.
    """
    print(
        "\tTest: For the max-diff portion, did people merely answer the questions based on their projects' general quality priorities?"
    )
    print(
        "\t\tSimulating what choices respondents would have picked if they based their decision purely on their stated priorities..."
    )
    maxDiffChoicePriorityDict = {
        "Maintainability": "QualityPriority_Maintainability",
        "Portability": "QualityPriority_Portability",
        "Usability": "QualityPriority_Usability",
        "Reliability": "QualityPriority_Reliability",
        "Compatibility": "QualityPriority_Compatibility",
        "PerformanceEfficiency": "QualityPriority_PerformanceEfficiency",
        "FunctionalSuitability": "QualityPriority_FunctionalSuitability",
        "Security": "QualityPriority_Security",
    }
    mergedSurveyData["MaxDiff_BestChoice_Predicted"] = " "
    mergedSurveyData["MaxDiff_WorstChoice_Predicted"] = " "

    def pickMaxDiffBestAndWorstBasedOnQualityPriorities(row):
        maxDiffChoices = [
            "MaxDiff_AttributeChoice1",
            "MaxDiff_AttributeChoice2",
            "MaxDiff_AttributeChoice3",
        ]
        priorityValuesForChoices = [
            row[maxDiffChoicePriorityDict[row[choice]]] for choice in maxDiffChoices
        ]
        row["MaxDiff_BestChoice_Predicted"] = row[
            maxDiffChoices[np.argmax(priorityValuesForChoices)]
        ]
        row["MaxDiff_WorstChoice_Predicted"] = row[
            maxDiffChoices[np.argmin(priorityValuesForChoices)]
        ]
        return row

    for i, row in mergedSurveyData.iterrows():
        mergedSurveyData.loc[i] = pickMaxDiffBestAndWorstBasedOnQualityPriorities(row)

    numberOfCorrectBestChoicePredictions = len(
        mergedSurveyData.loc[
            (
                mergedSurveyData["MaxDiff_BestChoice"]
                == mergedSurveyData["MaxDiff_BestChoice_Predicted"]
            )
        ].index
    )
    numberOfCorrectWorstChoicePredictions = len(
        mergedSurveyData.loc[
            (
                mergedSurveyData["MaxDiff_WorstChoice"]
                == mergedSurveyData["MaxDiff_WorstChoice_Predicted"]
            )
        ].index
    )
    numberOfFullyCorrectChoicePredictions = len(
        mergedSurveyData.loc[
            (
                mergedSurveyData["MaxDiff_BestChoice"]
                == mergedSurveyData["MaxDiff_BestChoice_Predicted"]
            )
            & (
                mergedSurveyData["MaxDiff_WorstChoice"]
                == mergedSurveyData["MaxDiff_WorstChoice_Predicted"]
            )
        ].index
    )
    numberOfMaxDiffQuestionsAsked = len(mergedSurveyData.index)

    print(
        "\t\tOut of {numberOfQuestions} asked...".format(
            numberOfQuestions=numberOfMaxDiffQuestionsAsked
        )
    )
    print(
        "\t\t\tThe best choice was correctly predicted {bestCorrect}/{numberOfQuestions} times ({percent}\%, {percentDiff}\% more than random chance)".format(
            bestCorrect=numberOfCorrectBestChoicePredictions,
            numberOfQuestions=numberOfMaxDiffQuestionsAsked,
            percent=round(
                100
                * (
                    numberOfCorrectBestChoicePredictions / numberOfMaxDiffQuestionsAsked
                ),
                2,
            ),
            percentDiff=round(
                100
                * (numberOfCorrectBestChoicePredictions / numberOfMaxDiffQuestionsAsked)
                - (100 * (1 / 3)),
                2,
            ),
        )
    )
    print(
        "\t\t\tThe worst choice was correctly predicted {worstCorrect}/{numberOfQuestions} times ({percent}\%, {percentDiff}\% more than random chance)".format(
            worstCorrect=numberOfCorrectWorstChoicePredictions,
            numberOfQuestions=numberOfMaxDiffQuestionsAsked,
            percent=round(
                100
                * (
                    numberOfCorrectWorstChoicePredictions
                    / numberOfMaxDiffQuestionsAsked
                ),
                2,
            ),
            percentDiff=round(
                100
                * (
                    numberOfCorrectWorstChoicePredictions
                    / numberOfMaxDiffQuestionsAsked
                )
                - (100 * (1 / 3)),
                2,
            ),
        )
    )
    print(
        "\t\t\tOur model was correct on both {bothCorrect}/{numberOfQuestions} times ({percent}\%, {percentDiff}\% more than random chance)".format(
            bothCorrect=numberOfFullyCorrectChoicePredictions,
            numberOfQuestions=numberOfMaxDiffQuestionsAsked,
            percent=round(
                100
                * (
                    numberOfFullyCorrectChoicePredictions
                    / numberOfMaxDiffQuestionsAsked
                ),
                2,
            ),
            percentDiff=round(
                100
                * (
                    numberOfFullyCorrectChoicePredictions
                    / numberOfMaxDiffQuestionsAsked
                )
                - (100 * ((1 / 3) * (1 / 3))),
                2,
            ),
        )
    )

    def computeAccuracyForRespondent(groupedResponses):
        numberOfFullyCorrectChoicePredictions = len(
            groupedResponses.loc[
                (
                    groupedResponses["MaxDiff_BestChoice"]
                    == groupedResponses["MaxDiff_BestChoice_Predicted"]
                )
                & (
                    groupedResponses["MaxDiff_WorstChoice"]
                    == groupedResponses["MaxDiff_WorstChoice_Predicted"]
                )
            ].index
        )
        numberOfMaxDiffQuestionsAsked = len(groupedResponses.index)
        return 100 * (
            numberOfFullyCorrectChoicePredictions / numberOfMaxDiffQuestionsAsked
        )

    resultsPerRespondent = mergedSurveyData.groupby(
        ['SurveyBranch', 'ResponseID']
    ).apply(computeAccuracyForRespondent)

    print(
        "\t\t\tOn average, the median accuracy (getting a fully correct prediction) was {medianPercent}\% on a per person basis (average {averagePercent}\%)".format(
            medianPercent=resultsPerRespondent.median(),
            averagePercent=resultsPerRespondent.mean(),
        )
    )

    # fig, ax = plt.subplots()
    # n, bins, patches = ax.hist(resultsPerRespondent, 10,edgecolor='black')
    # fig.tight_layout()
    # ax.set_xlabel('Percent of questions (out of 10) predicted correctly based on quality priorities')
    # ax.set_ylabel('Percentage of respondents')
    # ax.set_title("For the max-diff portion, did people merely answer the questions based on their projects' general quality priorities?")
    # plt.show()


def estimateBenefitOfQualityPrioritiesBasedOnMaxDiffUtilities(
    regularSurveyData, qualityUtilitiesData
):
    qualityPriorityRankCategories_BordaNormalized = [
        "QualityPriority_Maintainability",
        "QualityPriority_Portability",
        "QualityPriority_Reliability",
        "QualityPriority_Usability",
        "QualityPriority_Compatibility",
        "QualityPriority_PerformanceEfficiency",
        "QualityPriority_FunctionalSuitability",
        "QualityPriority_Security",
    ]
    qualityPriorityRankCategories_OriginalRank = [
        "QualityPriority_Maintainability_OriginalRank",
        "QualityPriority_Portability_OriginalRank",
        "QualityPriority_Reliability_OriginalRank",
        "QualityPriority_Usability_OriginalRank",
        "QualityPriority_Compatibility_OriginalRank",
        "QualityPriority_PerformanceEfficiency_OriginalRank",
        "QualityPriority_FunctionalSuitability_OriginalRank",
        "QualityPriority_Security_OriginalRank",
    ]
    qualityAttributesInUtilitiesDataframe = (
        qualityUtilitiesData["QualityAttribute"].dropna().unique()
    )
    assert all(
        [
            quality in qualityAttributesInUtilitiesDataframe
            for quality in [
                "Maintainability",
                "Portability",
                "Reliability",
                "Compatibility",
                "PerformanceEfficiency",
                "FunctionalSuitability",
                "Security",
            ]
        ]
    )

    respondentsWhoCouldThinkOfAProject = regularSurveyData.loc[
        (regularSurveyData['CanThinkOfProject'] == "Yes, I can think of a project.")
    ]

    def mapQualityColumnToUtilityColumn(qualityAttributeColumnLabel):
        qualityUtilityColumnNames = [
            "Maintainability",
            "Portability",
            "Reliability",
            "Usability",
            "Compatibility",
            "PerformanceEfficiency",
            "FunctionalSuitability",
            "Security",
        ]
        for name in qualityUtilityColumnNames:
            if name in qualityAttributeColumnLabel:
                return name
        return None  # This shouldn't happen and will cause an error.

    def calculateUtilityTotal(row, qualityColumns, outputColumnName, scalingFactor=100):
        totalUtility = 0.0
        for qualityAttribute in qualityColumns:
            totalUtility += (
                scalingFactor
                * row[qualityAttribute]
                * qualityUtilitiesData.loc[
                    (
                        qualityUtilitiesData["QualityAttribute"]
                        == mapQualityColumnToUtilityColumn(qualityAttribute)
                    )
                ]["Utility"].item()
            )
        row[outputColumnName] = totalUtility
        return row

    def calculateSimpleTotal(row, qualityColumns, outputColumnName, scalingFactor=100):
        totalUtility = 0.0
        for qualityAttribute in qualityColumns:
            changeInUtility = (
                scalingFactor
                * row[qualityAttribute]
                * qualityUtilitiesData.loc[
                    (
                        qualityUtilitiesData["QualityAttribute"]
                        == mapQualityColumnToUtilityColumn(qualityAttribute)
                    )
                ]["Utility"].item()
            )
            if (
                mapQualityColumnToUtilityColumn(qualityAttribute)
                in ["Maintainability", "Reliability,Usability"]
                and changeInUtility > 0
            ):
                # totalUtility += changeInUtility
                totalUtility += 1
        row[outputColumnName] = totalUtility
        return row

    respondentsWhoCouldThinkOfAProject["Alignment_BordaNormalized"] = 0.0
    respondentsWhoCouldThinkOfAProject["Alignment_ExponentialDropoffNormalized"] = 0.0
    respondentsWhoCouldThinkOfAProject["Alignment_Simple"] = 0.0
    rankToDropoffDict = {
        "QualityPriority_Maintainability_OriginalRank": "QualityPriority_Maintainability_Dropoff",
        "QualityPriority_Portability_OriginalRank": "QualityPriority_Portability_Dropoff",
        "QualityPriority_Reliability_OriginalRank": "QualityPriority_Reliability_Dropoff",
        "QualityPriority_Usability_OriginalRank": "QualityPriority_Usability_Dropoff",
        "QualityPriority_Compatibility_OriginalRank": "QualityPriority_Compatibility_Dropoff",
        "QualityPriority_PerformanceEfficiency_OriginalRank": "QualityPriority_PerformanceEfficiency_Dropoff",
        "QualityPriority_FunctionalSuitability_OriginalRank": "QualityPriority_FunctionalSuitability_Dropoff",
        "QualityPriority_Security_OriginalRank": "QualityPriority_Security_Dropoff",
    }
    for columnName in list(rankToDropoffDict.values()):
        respondentsWhoCouldThinkOfAProject[columnName] = 0.0

    def exponentialDropoffScore(row):
        for quality in qualityPriorityRankCategories_OriginalRank:
            if pd.isna(row[quality]):
                row[rankToDropoffDict[quality]] = 0.0
            else:
                row[rankToDropoffDict[quality]] = np.exp(-1 * (row[quality] - 1))
        return row

    def normalizeRow(row, columnsToNormalize):
        if np.sum(row[columnsToNormalize]) > 0:
            return row[columnsToNormalize] / np.sum(row[columnsToNormalize])
        else:
            return row

    for i, row in respondentsWhoCouldThinkOfAProject.iterrows():
        respondentsWhoCouldThinkOfAProject.loc[i] = exponentialDropoffScore(row)
        respondentsWhoCouldThinkOfAProject.loc[i, list(rankToDropoffDict.values())] = (
            normalizeRow(row, list(rankToDropoffDict.values()))
        )

    for i, row in respondentsWhoCouldThinkOfAProject.iterrows():
        respondentsWhoCouldThinkOfAProject.loc[i] = calculateUtilityTotal(
            row,
            qualityPriorityRankCategories_BordaNormalized,
            "Alignment_BordaNormalized",
        )
        respondentsWhoCouldThinkOfAProject.loc[i] = calculateSimpleTotal(
            row, qualityPriorityRankCategories_BordaNormalized, "Alignment_Simple"
        )
        respondentsWhoCouldThinkOfAProject.loc[i] = calculateUtilityTotal(
            row,
            list(rankToDropoffDict.values()),
            "Alignment_ExponentialDropoffNormalized",
        )

    def scanForCorrelations(df, columnName):
        cols = pd.DataFrame(columns=df.columns)
        p = cols.transpose().join(cols, how='outer')
        for r in df.columns:
            tmp = df[df[r].notnull() & df[columnName].notnull()]
            try:
                statistic, pvalue = scipy.stats.pearsonr(tmp[r], tmp[columnName])
                p[r][columnName] = (statistic, pvalue)
            except Exception:
                p[r][columnName] = (0.0, 9999.0)
        print("Scanning for correlations for", columnName)
        for r in p.columns:
            if "QualityPriority" in r or "Alignment" in r:
                continue
            statistic, pvalue = p[r][columnName]
            if pvalue <= 0.05:  # and abs(statistic) >= 0.05:
                print("\t\t", r, "Pearson's r:", statistic, "p-value", pvalue)

    scanForCorrelations(respondentsWhoCouldThinkOfAProject, "Alignment_BordaNormalized")
    scanForCorrelations(respondentsWhoCouldThinkOfAProject, "Alignment_Simple")
    scanForCorrelations(
        respondentsWhoCouldThinkOfAProject, "Alignment_ExponentialDropoffNormalized"
    )

    respondentsWhoCouldThinkOfAProject["Alignment_BordaNormalized_AboveAverage"] = (
        respondentsWhoCouldThinkOfAProject["Alignment_BordaNormalized"]
        > respondentsWhoCouldThinkOfAProject["Alignment_BordaNormalized"].mean()
    )

    # fig, ax = plt.subplots()
    # n, bins, patches = ax.hist(
    #     respondentsWhoCouldThinkOfAProject["Alignment_BordaNormalized"],
    #     edgecolor='black',
    # )
    # bin_centers = 0.5 * (bins[:-1] + bins[1:])
    # col = bin_centers - min(bin_centers)
    # col /= max(col)
    # cm = plt.cm.get_cmap('magma')
    # for c, p in zip(col, patches):
    #     plt.setp(p, 'facecolor', cm(c))
    # # fig.tight_layout()
    # ax.set_xlabel(
    #     'Estimated Alignment Based on MaxDiff Utilities and Normalized Borda Quality Scores'
    # )
    # ax.set_ylabel('Percentage of Respondents')
    # ax.set_title("Software Quality and Reproducibility Goal Alignment")
    # plt.show()


def sumTogetherMaxDiffChoicesAndAppendToRegularSurveyData(
    regularSurveyData, maxDiffSurveyData
):
    maxDiffChoiceSumDict = {
        "Maintainability": "MaxDiffSum_Maintainability",
        "Portability": "MaxDiffSum_Portability",
        "Usability": "MaxDiffSum_Usability",
        "Reliability": "MaxDiffSum_Reliability",
        "Compatibility": "MaxDiffSum_Compatibility",
        "PerformanceEfficiency": "MaxDiffSum_PerformanceEfficiency",
        "FunctionalSuitability": "MaxDiffSum_FunctionalSuitability",
        "Security": "MaxDiffSum_Security",
    }

    def computeSumOfMaxDiffAnswers(groupedResponses):
        # Note: If the user was never presented with a question where a quality was present, then
        # we'll leave it as np.nan and drop that person when doing correlations.
        maxDiffSum = {
            # "SurveyBranch" : groupedResponses['SurveyBranch'].iloc[0],
            # "ResponseID" : groupedResponses['ResponseID'].iloc[0],
            "MaxDiffSum_Maintainability": np.nan,
            "MaxDiffSum_Portability": np.nan,
            "MaxDiffSum_Usability": np.nan,
            "MaxDiffSum_Reliability": np.nan,
            "MaxDiffSum_Compatibility": np.nan,
            "MaxDiffSum_PerformanceEfficiency": np.nan,
            "MaxDiffSum_FunctionalSuitability": np.nan,
            "MaxDiffSum_Security": np.nan,
        }
        maxDiffSum = pd.Series(data=maxDiffSum)
        for index, row in groupedResponses.iterrows():
            if pd.isna(row["MaxDiff_BestChoice"]) or pd.isna(
                row["MaxDiff_WorstChoice"]
            ):
                continue
            for choice in [
                "MaxDiff_AttributeChoice1",
                "MaxDiff_AttributeChoice2",
                "MaxDiff_AttributeChoice3",
            ]:
                if pd.isna(maxDiffSum[maxDiffChoiceSumDict[row[choice]]]):
                    maxDiffSum[maxDiffChoiceSumDict[row[choice]]] = 0
            maxDiffSum[maxDiffChoiceSumDict[row["MaxDiff_BestChoice"]]] += 1
            maxDiffSum[maxDiffChoiceSumDict[row["MaxDiff_WorstChoice"]]] -= 1
        return maxDiffSum

    summedResultsPerRespondent = maxDiffSurveyData.groupby(
        ['SurveyBranch', 'ResponseID']
    ).apply(computeSumOfMaxDiffAnswers)
    regularSurveyData = regularSurveyData.merge(
        summedResultsPerRespondent, on=['SurveyBranch', 'ResponseID'], how='left'
    )

    return regularSurveyData


def maxDiffChoiceExploration(
    regularSurveyData, maxDiffSurveyData, qualityUtilitiesData
):
    print("-----------------------------")
    print("Testing what factors might influence MaxDiff choices...")
    maxDiffChoicePriorityDict = {
        "Maintainability": "QualityPriority_Maintainability",
        "Portability": "QualityPriority_Portability",
        "Usability": "QualityPriority_Usability",
        "Reliability": "QualityPriority_Reliability",
        "Compatibility": "QualityPriority_Compatibility",
        "PerformanceEfficiency": "QualityPriority_PerformanceEfficiency",
        "FunctionalSuitability": "QualityPriority_FunctionalSuitability",
        "Security": "QualityPriority_Security",
    }
    maxDiffChoiceReproducibilityNeedDict = {
        "Maintainability": "ReproducibilityNeed_Maintainability",
        "Portability": "ReproducibilityNeed_Portability",
        "Usability": "ReproducibilityNeed_Usability",
        "Reliability": "ReproducibilityNeed_Reliability",
        "Compatibility": "ReproducibilityNeed_Compatibility",
        "PerformanceEfficiency": "ReproducibilityNeed_PerformanceEfficiency",
        "FunctionalSuitability": "ReproducibilityNeed_FunctionalSuitability",
        "Security": "ReproducibilityNeed_Security",
    }

    maxDiffSurveyData = maxDiffSurveyData.merge(
        regularSurveyData, on=['SurveyBranch', 'ResponseID'], how='left'
    )

    maxDiffSurveyData['BestChoice_QualityPriority'] = 0.0
    maxDiffSurveyData['WorstChoice_QualityPriority'] = 0.0
    maxDiffSurveyData['BestChoice_ReproducibilityNeed'] = 0.0
    maxDiffSurveyData['WorstChoice_ReproducibilityNeed'] = 0.0

    for choice in maxDiffChoicePriorityDict:
        maxDiffSurveyData['BestChoice_QualityPriority'] = np.where(
            maxDiffSurveyData["MaxDiff_BestChoice"] == choice,
            maxDiffSurveyData[maxDiffChoicePriorityDict[choice]],
            maxDiffSurveyData['BestChoice_QualityPriority'],
        )
        maxDiffSurveyData['WorstChoice_QualityPriority'] = np.where(
            maxDiffSurveyData["MaxDiff_WorstChoice"] == choice,
            maxDiffSurveyData[maxDiffChoicePriorityDict[choice]],
            maxDiffSurveyData['WorstChoice_QualityPriority'],
        )

        maxDiffSurveyData['BestChoice_ReproducibilityNeed'] = np.where(
            maxDiffSurveyData["MaxDiff_BestChoice"] == choice,
            maxDiffSurveyData[maxDiffChoiceReproducibilityNeedDict[choice]],
            maxDiffSurveyData['BestChoice_ReproducibilityNeed'],
        )
        maxDiffSurveyData['WorstChoice_ReproducibilityNeed'] = np.where(
            maxDiffSurveyData["MaxDiff_WorstChoice"] == choice,
            maxDiffSurveyData[maxDiffChoiceReproducibilityNeedDict[choice]],
            maxDiffSurveyData['WorstChoice_ReproducibilityNeed'],
        )

    r_priority, p_priority = scipy.stats.pearsonr(
        maxDiffSurveyData['BestChoice_QualityPriority'],
        maxDiffSurveyData['WorstChoice_QualityPriority'],
    )
    print(
        "\tPearson's r correlation between priority of best choice and priority of worst choice: r={rvalue},p={pvalue}".format(
            rvalue=r_priority, pvalue=p_priority
        )
    )

    r_need, p_need = scipy.stats.pearsonr(
        maxDiffSurveyData.loc[
            (maxDiffSurveyData['BestChoice_ReproducibilityNeed'].notnull())
            & (maxDiffSurveyData['WorstChoice_ReproducibilityNeed'].notnull())
        ]['BestChoice_ReproducibilityNeed'],
        maxDiffSurveyData['WorstChoice_ReproducibilityNeed'].loc[
            (maxDiffSurveyData['BestChoice_ReproducibilityNeed'].notnull())
            & (maxDiffSurveyData['WorstChoice_ReproducibilityNeed'].notnull())
        ],
    )
    print(
        "\tPearson's r correlation between stated reproducibility need of best choice and stated reproducibility need of worst choice: r={rvalue},p={pvalue}".format(
            rvalue=r_need, pvalue=p_need
        )
    )

    testWhetherMaxDiffChoicesMerelyReflectRespondentsQualityPriorities(
        maxDiffSurveyData
    )
    estimateBenefitOfQualityPrioritiesBasedOnMaxDiffUtilities(
        regularSurveyData, qualityUtilitiesData
    )

    print("-----------------------------")


def testWhetherProjectCharacteristicsInfluenceQualityPriorities(
    regularSurveyData, characteristicsToTest, alpha=0.05
):
    print("-----------------------------")
    print(
        "Testing whether project characteristics influence quality priorities among our survey respondents..."
    )

    def testInfluenceOfProjectCharacteristicOnQualityPriorities(
        regularSurveyData, characteristic, alpha, numberOfHypotheses
    ):
        """
        alpha: The threshold for statistical significance (usually 0.05).
        numberOfHypotheses: How many characteristics are we testing? We apply the Bonferroni correction by dividing the p-values by the number of characteristics we are testing.
        """
        respondentsWhoCouldThinkOfAProject = regularSurveyData.loc[
            (regularSurveyData['CanThinkOfProject'] == "Yes, I can think of a project.")
            & (regularSurveyData[characteristic].notnull())
        ]
        uniqueValuesForCharacteristic = (
            respondentsWhoCouldThinkOfAProject[characteristic].dropna().unique()
        )
        uniqueValuesForCharacteristic.sort()
        areAnyCharacteristicValuesStrings = any(
            [isinstance(v, str) for v in uniqueValuesForCharacteristic]
        )

        qualityPriorityRankCategories = [
            "QualityPriority_Maintainability",
            "QualityPriority_Portability",
            "QualityPriority_Reliability",
            "QualityPriority_Usability",
            "QualityPriority_Compatibility",
            "QualityPriority_PerformanceEfficiency",
            "QualityPriority_FunctionalSuitability",
            "QualityPriority_Security",
        ]
        # qualityPriorityRankCategories=["Belief_ReproducibilityIsFundamental","Belief_ReproducibilityEnablesProgress","Belief_WillingToInvestExtraInQuality","Belief_WillingToSetUpComplexSolutions",
        #    "Belief_WillingToSacrificeQuality","Belief_AwareOfBestPractices","Belief_HasKnowledgeToolsAndResources","Belief_CanTakeActionOnReproducibility",
        #    "Belief_ProjectHasEffectivePractices","Belief_ReproducibilityNotAHindrance","Belief_CommunityValuesReproducibility","Belief_InstitutionValuesReproducibility",
        #    "Belief_StakeholdersValueReproducibility","Belief_CanLearnFromPeers","Belief_NeedForStandardGuidelines","Belief_NeedForTrainingOpportunities"]

        for qualityPriority in qualityPriorityRankCategories:
            groupsToTest = [
                respondentsWhoCouldThinkOfAProject.loc[
                    (respondentsWhoCouldThinkOfAProject[characteristic] == value)
                ][qualityPriority]
                for value in uniqueValuesForCharacteristic
            ]

            h_statistic, p_value = scipy.stats.kruskal(*groupsToTest)
            if p_value <= alpha:
                print(
                    qualityPriority,
                    "\tKruskal-Wallis H-Test Outcome:",
                    "h={h},p={p}".format(h=h_statistic, p=p_value),
                )
                print(
                    "\tExplanation: According to the Kruskal-Wallis H-test, the median score of {priority} is *not* equal across projects grouped by {characteristic}.".format(
                        priority=qualityPriority, characteristic=characteristic
                    )
                )

                if p_value <= alpha / numberOfHypotheses:
                    print(
                        "\tThe p-value of the Kruskal-Wallis test IS statistically significant after applying the Bonferroni correction (p<={adjustedAlpha}).".format(
                            adjustedAlpha=alpha / numberOfHypotheses
                        )
                    )
                else:
                    print(
                        "\tThe p-value of the Kruskal-Wallis test IS *NOT* statistically significant after applying the Bonferroni correction (p<={adjustedAlpha}).".format(
                            adjustedAlpha=alpha / numberOfHypotheses
                        )
                    )

                if not areAnyCharacteristicValuesStrings:
                    r_characteristic, p_characteristic = scipy.stats.pearsonr(
                        respondentsWhoCouldThinkOfAProject[characteristic],
                        respondentsWhoCouldThinkOfAProject[qualityPriority],
                    )
                    print(
                        "\tPearson's r correlation between {priority} and {characteristic}: r={rvalue},p={pvalue}".format(
                            priority=qualityPriority,
                            characteristic=characteristic,
                            rvalue=r_characteristic,
                            pvalue=p_characteristic,
                        )
                    )

                for i in range(len(groupsToTest)):
                    group = groupsToTest[i]
                    characteristicValue = uniqueValuesForCharacteristic[i]
                    percentWhoRankedQuality = round(
                        100 * (len(group.loc[group > 0].index) / len(group.index)), 2
                    )
                    print(
                        "\t\tGroup ({characteristicValue}): {percentRanked}\% ranked this quality, median score among those who ranked it {medianScore}".format(
                            characteristicValue=characteristicValue,
                            percentRanked=percentWhoRankedQuality,
                            medianScore=group.loc[group > 0].median(),
                        )
                    )

    for characteristic in characteristicsToTest:
        testInfluenceOfProjectCharacteristicOnQualityPriorities(
            regularSurveyData,
            characteristic,
            alpha,
            numberOfHypotheses=len(characteristicsToTest),
        )

    print("-----------------------------")


def theoryOfPlannedBehaviorAnalysis(regularSurveyData):
    print("-----------------------------")
    print("Computing theory of planned behavior analysis...")

    def cronbachAlpha(data):
        """
        Cronbach's alpha is a reliability coefficient and a measure of internal consistency of tests and measures.
        """
        # Transform the data frame into a correlation matrix
        df_corr = data.corr()
        # Calculate N
        # The number of variables is equal to the number of columns in the dataframe
        N = data.shape[1]
        # Calculate r
        # For this, we'll loop through all the columns and append every
        # relevant correlation to an array called 'r_s'. Then, we'll
        # calculate the mean of 'r_s'.
        rs = np.array([])
        for i, col in enumerate(df_corr.columns):
            sum_ = df_corr[col][i + 1 :].values
            rs = np.append(sum_, rs)
        mean_r = np.mean(rs)

        # Use the formula to calculate Cronbach's Alpha
        cronbach_alpha = (N * mean_r) / (1 + (N - 1) * mean_r)
        return cronbach_alpha

    def combineMeasuresIntoConstruct(constructName, listOfSurveyQuestionsToCombine):
        cronbachAlphaScore = cronbachAlpha(
            regularSurveyData[listOfSurveyQuestionsToCombine]
        )
        print("\t", constructName, ":", cronbachAlphaScore)
        regularSurveyData[constructName] = regularSurveyData[
            listOfSurveyQuestionsToCombine
        ].sum(axis=1)

    combineMeasuresIntoConstruct(
        "construct_Attitude_ReproducibilityIsImportant",
        [
            "Belief_ReproducibilityIsFundamental",
            "Belief_ReproducibilityEnablesProgress",
        ],
    )
    combineMeasuresIntoConstruct(
        "construct_Attitude_QualityAndReproducibilityAreRelated",
        [
            "Belief_HighQualitySoftwareIsMoreReproducible",
            "Belief_ReproducibilityIsMoreImportantThanSpeedOrFeatures",
        ],
    )
    combineMeasuresIntoConstruct(
        "construct_PerceivedBehavioralControl_HasKnowledgeAndAwareness",
        ["Belief_AwareOfBestPractices", "Belief_HasKnowledgeToolsAndResources"],
    )
    combineMeasuresIntoConstruct(
        "construct_PerceivedBehavioralControl_CanTakeAction",
        [
            "Belief_CanTakeActionOnReproducibility",
            "Belief_ProjectHasEffectivePractices",
            "Belief_ReproducibilityNotAHindrance",
        ],
    )
    combineMeasuresIntoConstruct(
        "construct_SubjectiveNorm_FeelsSupported",
        [
            "Belief_CommunityValuesReproducibility",
            "Belief_InstitutionValuesReproducibility",
            "Belief_StakeholdersValueReproducibility",
            "Belief_CanLearnFromPeers",
        ],
    )

    combineMeasuresIntoConstruct(
        "construct_ReproducibilityNeedOverTime",
        [
            "ReproducibilityNeed_NeededForMonths",
            "ReproducibilityNeed_NeededForYears",
            "ReproducibilityNeed_NeededForDecades",
        ],
    )
    combineMeasuresIntoConstruct(
        "construct_Intention",
        [
            "Belief_WillingToInvestExtraInQuality",
            "Belief_WillingToSetUpComplexSolutions",
            "Belief_WillingToSacrificeQuality",
        ],
    )

    print("-----------------------------")


def generateDemographicVariables(regularSurveyData, maxDiffSurveyData):
    demographicVariables = {}

    # Survey recruitment information
    demographicVariables["surveyStartDate"] = "June 2023"
    demographicVariables["surveyEndDate"] = "September 2023"
    demographicVariables["surveyNumberOfCentersOutsideFourteenHundred"] = "eight"
    demographicVariables["surveyEstimatedCost"] = "\$3000"
    demographicVariables["surveyEstimatedResponseRate"] = "ESTIMATED RESPONSE RATE"
    demographicVariables["surveyTotalResponses"] = len(regularSurveyData.index)
    demographicVariables["surveyTotalNumberOfSandiaParticipants"] = len(
        regularSurveyData.loc[
            regularSurveyData['SurveyBranch'].isin(
                ["MainClosed", "MainOpen", "CaseStudy"]
            )
        ].index
    )
    demographicVariables["surveyNumberOfSandiaCaseStudyResponses"] = len(
        regularSurveyData.loc[regularSurveyData['SurveyBranch'] == "CaseStudy"].index
    )
    demographicVariables["surveyNumberOfSandiaNonCaseStudyResponses"] = (
        demographicVariables["surveyTotalNumberOfSandiaParticipants"]
        - demographicVariables["surveyNumberOfSandiaCaseStudyResponses"]
    )
    demographicVariables["surveyTotalNumberOfExternalResponses"] = len(
        regularSurveyData.loc[
            regularSurveyData['SurveyBranch'].isin(
                ["ACMREP", "Astro", "IDEAS", "USRSE"]
            )
        ].index
    )
    demographicVariables["surveyNumberOfACMREPResponses"] = len(
        regularSurveyData.loc[regularSurveyData['SurveyBranch'] == "ACMREP"].index
    )
    demographicVariables["surveyNumberOfUSRSEResponses"] = len(
        regularSurveyData.loc[regularSurveyData['SurveyBranch'] == "USRSE"].index
    )
    demographicVariables["surveyNumberOfAstroAndIDEASResponses"] = len(
        regularSurveyData.loc[
            regularSurveyData['SurveyBranch'].isin(["Astro", "IDEAS"])
        ].index
    )

    # Demographic information: Employment
    demographicVariables["surveyDemographicsPercentOfRespondentsFromSandia"] = round(
        100
        * demographicVariables["surveyTotalNumberOfSandiaParticipants"]
        / demographicVariables["surveyTotalResponses"],
        2,
    )
    demographicVariables[
        "surveyDemographicsPercentOfExternalRespondentsFromNationalLabs"
    ] = round(
        100
        * len(
            regularSurveyData.loc[
                (
                    regularSurveyData['SurveyBranch'].isin(
                        ["ACMREP", "Astro", "IDEAS", "USRSE"]
                    )
                )
                & (regularSurveyData['Institution'] == "University")
            ].index
        )
        / demographicVariables["surveyTotalNumberOfExternalResponses"],
        2,
    )

    demographicVariables[
        "surveyDemographicsPercentOfExternalRespondentsFromUniversities"
    ] = round(
        100
        * len(
            regularSurveyData.loc[
                (
                    regularSurveyData['SurveyBranch'].isin(
                        ["ACMREP", "Astro", "IDEAS", "USRSE"]
                    )
                )
                & (
                    regularSurveyData['Institution']
                    == "National Laboratory (or other government-affiliated institution)"
                )
            ].index
        )
        / demographicVariables["surveyTotalNumberOfExternalResponses"],
        2,
    )

    demographicVariables[
        "surveyDemographicsPercentOfExternalRespondentsFromPrivateCompanies"
    ] = round(
        100
        * len(
            regularSurveyData.loc[
                (
                    regularSurveyData['SurveyBranch'].isin(
                        ["ACMREP", "Astro", "IDEAS", "USRSE"]
                    )
                )
                & (regularSurveyData['Institution'] == "Private Company")
            ].index
        )
        / demographicVariables["surveyTotalNumberOfExternalResponses"],
        2,
    )

    demographicVariables[
        "surveyDemographicsPercentOfExternalRespondentsSelfEmployedOrOther"
    ] = round(
        (
            100.0
            - demographicVariables[
                "surveyDemographicsPercentOfExternalRespondentsFromNationalLabs"
            ]
            - demographicVariables[
                "surveyDemographicsPercentOfExternalRespondentsFromUniversities"
            ]
            - demographicVariables[
                "surveyDemographicsPercentOfExternalRespondentsFromPrivateCompanies"
            ]
        ),
        2,
    )

    employmentValueCounts = regularSurveyData["Institution"].value_counts()
    demographicVariables["surveyDemographicsPercentOfRespondentsFromNationalLabs"] = (
        round(
            100
            * employmentValueCounts.get(
                "National Laboratory (or other government-affiliated institution)", 0
            )
            / demographicVariables["surveyTotalResponses"],
            2,
        )
    )
    demographicVariables["surveyDemographicsPercentOfRespondentsFromUniversities"] = (
        round(
            100
            * employmentValueCounts.get("University", 0)
            / demographicVariables["surveyTotalResponses"],
            2,
        )
    )
    demographicVariables[
        "surveyDemographicsPercentOfRespondentsFromPrivateCompanies"
    ] = round(
        100
        * employmentValueCounts.get("Private Company", 0)
        / demographicVariables["surveyTotalResponses"],
        2,
    )
    demographicVariables[
        "surveyDemographicsPercentOfRespondentsSelfEmployedOrUnemployed"
    ] = round(
        100
        * employmentValueCounts.get("Unemployed or Self-Employed", 0)
        / demographicVariables["surveyTotalResponses"],
        2,
    )
    demographicVariables["surveyDemographicsPercentOfRespondentsOtherEmployment"] = (
        round(
            100
            - demographicVariables[
                "surveyDemographicsPercentOfRespondentsFromNationalLabs"
            ]
            - demographicVariables[
                "surveyDemographicsPercentOfRespondentsFromUniversities"
            ]
            - demographicVariables[
                "surveyDemographicsPercentOfRespondentsFromPrivateCompanies"
            ]
            - demographicVariables[
                "surveyDemographicsPercentOfRespondentsSelfEmployedOrUnemployed"
            ],
            2,
        )
    )

    # Demographic information: Job Role
    demographicVariables["surveyDemographicsPercentWhoDevelopSoftware"] = round(
        100
        * len(
            regularSurveyData.loc[
                regularSurveyData['Role_SoftwareDeveloper'] == 1
            ].index
        )
        / demographicVariables["surveyTotalResponses"],
        2,
    )
    demographicVariables["surveyDemographicsPercentWhoUseSoftware"] = round(
        100
        * len(regularSurveyData.loc[regularSurveyData['Role_SoftwareUser'] == 1].index)
        / demographicVariables["surveyTotalResponses"],
        2,
    )
    demographicVariables["surveyDemographicsPercentWhoBothDevelopAndUseSoftware"] = (
        round(
            100
            * len(
                regularSurveyData.loc[
                    (regularSurveyData['Role_SoftwareDeveloper'] == 1)
                    & (regularSurveyData['Role_SoftwareUser'] == 1)
                ].index
            )
            / demographicVariables["surveyTotalResponses"],
            2,
        )
    )
    demographicVariables["surveyDemographicsPercentWhoPerformResearch"] = round(
        100
        * len(regularSurveyData.loc[regularSurveyData['Role_Researcher'] == 1].index)
        / demographicVariables["surveyTotalResponses"],
        2,
    )
    demographicVariables["surveyDemographicsPercentWhoPerformEngineering"] = round(
        100
        * len(regularSurveyData.loc[regularSurveyData['Role_Engineer'] == 1].index)
        / demographicVariables["surveyTotalResponses"],
        2,
    )
    demographicVariables["surveyDemographicsPercentWhoPerformManagement"] = round(
        100
        * len(regularSurveyData.loc[regularSurveyData['Role_Manager'] == 1].index)
        / demographicVariables["surveyTotalResponses"],
        2,
    )

    # Demographic information: Education and Experience
    educationValueCounts = regularSurveyData["LevelOfEducation"].value_counts()
    demographicVariables["surveyDemographicsPercentWhoHaveHighSchool"] = round(
        100
        * educationValueCounts.get(0, 0)
        / demographicVariables["surveyTotalResponses"],
        2,
    )
    demographicVariables["surveyDemographicsPercentWhoHaveAssociates"] = round(
        100
        * educationValueCounts.get(1, 0)
        / demographicVariables["surveyTotalResponses"],
        2,
    )
    demographicVariables["surveyDemographicsPercentWhoHaveBachelors"] = round(
        100
        * educationValueCounts.get(2, 0)
        / demographicVariables["surveyTotalResponses"],
        2,
    )
    demographicVariables["surveyDemographicsPercentWhoHaveMasters"] = round(
        100
        * educationValueCounts.get(3, 0)
        / demographicVariables["surveyTotalResponses"],
        2,
    )
    demographicVariables["surveyDemographicsPercentWhoHavePhD"] = round(
        100
        * educationValueCounts.get(4, 0)
        / demographicVariables["surveyTotalResponses"],
        2,
    )
    demographicVariables["surveyDemographicsPercentWhoHaveOther"] = round(
        100
        - demographicVariables["surveyDemographicsPercentWhoHaveHighSchool"]
        - demographicVariables["surveyDemographicsPercentWhoHaveAssociates"]
        - demographicVariables["surveyDemographicsPercentWhoHaveBachelors"]
        - demographicVariables["surveyDemographicsPercentWhoHaveMasters"]
        - demographicVariables["surveyDemographicsPercentWhoHavePhD"],
        2,
    )

    experienceValueCounts = regularSurveyData["YearsOfExperience"].value_counts()
    demographicVariables["surveyDemographicsPercentWhoHaveLessThanAYearExperience"] = (
        round(
            100
            * experienceValueCounts.get(0, 0)
            / demographicVariables["surveyTotalResponses"],
            2,
        )
    )
    demographicVariables["surveyDemographicsPercentWhoHaveOneToFiveYearsExperience"] = (
        round(
            100
            * experienceValueCounts.get(1, 0)
            / demographicVariables["surveyTotalResponses"],
            2,
        )
    )
    demographicVariables["surveyDemographicsPercentWhoHaveSixToTenYearsExperience"] = (
        round(
            100
            * experienceValueCounts.get(2, 0)
            / demographicVariables["surveyTotalResponses"],
            2,
        )
    )
    demographicVariables[
        "surveyDemographicsPercentWhoHaveElevenToFifteenYearsExperience"
    ] = round(
        100
        * experienceValueCounts.get(3, 0)
        / demographicVariables["surveyTotalResponses"],
        2,
    )
    demographicVariables[
        "surveyDemographicsPercentWhoHaveSixteenToTwentyYearsExperience"
    ] = round(
        100
        * experienceValueCounts.get(4, 0)
        / demographicVariables["surveyTotalResponses"],
        2,
    )
    demographicVariables[
        "surveyDemographicsPercentWhoHaveTwentyPlusYearsExperience"
    ] = round(
        100
        * experienceValueCounts.get(5, 0)
        / demographicVariables["surveyTotalResponses"],
        2,
    )

    softwareDevelopmentExperienceCounts = regularSurveyData[
        "SoftwareDevelopmentExperience"
    ].value_counts()
    demographicVariables["surveyDemographicsPercentWhoHaveNoSWDExperience"] = round(
        100
        * softwareDevelopmentExperienceCounts.get(0, 0)
        / demographicVariables["surveyTotalResponses"],
        2,
    )
    demographicVariables["surveyDemographicsPercentWhoHaveLimitedSWDExperience"] = (
        round(
            100
            * softwareDevelopmentExperienceCounts.get(1, 0)
            / demographicVariables["surveyTotalResponses"],
            2,
        )
    )
    demographicVariables["surveyDemographicsPercentWhoHaveModerateSWDExperience"] = (
        round(
            100
            * softwareDevelopmentExperienceCounts.get(2, 0)
            / demographicVariables["surveyTotalResponses"],
            2,
        )
    )
    demographicVariables["surveyDemographicsPercentWhoHaveSignificantSWDExperience"] = (
        round(
            100
            * softwareDevelopmentExperienceCounts.get(3, 0)
            / demographicVariables["surveyTotalResponses"],
            2,
        )
    )
    demographicVariables["surveyDemographicsPercentWhoHaveHighSWDExperience"] = round(
        100
        * softwareDevelopmentExperienceCounts.get(4, 0)
        / demographicVariables["surveyTotalResponses"],
        2,
    )
    demographicVariables["surveyDemographicsPercentWhoHaveNotAskedSWDExperience"] = (
        round(
            100
            - demographicVariables["surveyDemographicsPercentWhoHaveNoSWDExperience"]
            - demographicVariables[
                "surveyDemographicsPercentWhoHaveLimitedSWDExperience"
            ]
            - demographicVariables[
                "surveyDemographicsPercentWhoHaveModerateSWDExperience"
            ]
            - demographicVariables[
                "surveyDemographicsPercentWhoHaveSignificantSWDExperience"
            ]
            - demographicVariables["surveyDemographicsPercentWhoHaveHighSWDExperience"],
            2,
        )
    )

    demographicVariables["surveyDemographicsPercentWithElevenPlusYearsExperience"] = (
        round(
            100
            * len(
                regularSurveyData.loc[regularSurveyData['YearsOfExperience'] >= 3].index
            )
            / demographicVariables["surveyTotalResponses"],
            2,
        )
    )
    yearsOfExperienceDict = {
        0: "Less than a year",
        1: "1-5 years",
        2: "6-10 years",
        3: "11-15 years",
        4: "16-20 years",
        5: "20+ years",
    }
    demographicVariables["surveyDemographicsMedianExperienceRange"] = (
        yearsOfExperienceDict[regularSurveyData['YearsOfExperience'].median()]
    )
    demographicVariables[
        "surveyDemographicsPercentWithSignificantOrMoreDevelopmentExpertise"
    ] = round(
        100
        * len(
            regularSurveyData.loc[
                regularSurveyData['SoftwareDevelopmentExperience'] >= 3
            ].index
        )
        / demographicVariables["surveyTotalResponses"],
        2,
    )

    return demographicVariables


def generateProjectInformationVariables(regularSurveyData, maxDiffSurveyData):
    projectInformationVariables = {}

    # Project Information: Maturity, Team Size, and FTE Count
    projectInformationVariables["surveyProjectInfoNumberCanThinkOfAProject"] = len(
        regularSurveyData.loc[
            regularSurveyData['CanThinkOfProject'] == "Yes, I can think of a project."
        ].index
    )
    projectInformationVariables["surveyProjectInfoPercentCanThinkOfAProject"] = round(
        100
        * len(
            regularSurveyData.loc[
                regularSurveyData['CanThinkOfProject']
                == "Yes, I can think of a project."
            ].index
        )
        / len(regularSurveyData.index),
        2,
    )
    projectInformationVariables["surveyProjectInfoPercentProductionized"] = round(
        100
        * len(regularSurveyData.loc[regularSurveyData['Project_Maturity'] >= 3].index)
        / projectInformationVariables["surveyProjectInfoNumberCanThinkOfAProject"],
        2,
    )
    projectInformationVariables["surveyProjectInfoAverageMaturity"] = round(
        regularSurveyData.loc[
            regularSurveyData['CanThinkOfProject'] == "Yes, I can think of a project."
        ]["Project_Maturity"].mean(),
        2,
    )
    projectInformationVariables["surveyProjectInfoMedianMaturity"] = round(
        regularSurveyData.loc[
            regularSurveyData['CanThinkOfProject'] == "Yes, I can think of a project."
        ]["Project_Maturity"].median(),
        2,
    )

    maturityValueCounts = regularSurveyData.loc[
        regularSurveyData['CanThinkOfProject'] == "Yes, I can think of a project."
    ]["Project_Maturity"].value_counts()
    projectInformationVariables[
        "surveyProjectInfoPercentEarlyResearchStagesMaturity"
    ] = round(
        100
        * maturityValueCounts.get(0, 0)
        / projectInformationVariables["surveyProjectInfoNumberCanThinkOfAProject"],
        2,
    )
    projectInformationVariables["surveyProjectInfoPercentVeryExploratoryMaturity"] = (
        round(
            100
            * maturityValueCounts.get(1, 0)
            / projectInformationVariables["surveyProjectInfoNumberCanThinkOfAProject"],
            2,
        )
    )
    projectInformationVariables[
        "surveyProjectInfoPercentSomewhatExploratoryMaturity"
    ] = round(
        100
        * maturityValueCounts.get(2, 0)
        / projectInformationVariables["surveyProjectInfoNumberCanThinkOfAProject"],
        2,
    )
    projectInformationVariables[
        "surveyProjectInfoPercentSomewhatProductionizedMaturity"
    ] = round(
        100
        * maturityValueCounts.get(3, 0)
        / projectInformationVariables["surveyProjectInfoNumberCanThinkOfAProject"],
        2,
    )
    projectInformationVariables["surveyProjectInfoPercentVeryProductizedMaturity"] = (
        round(
            100
            * maturityValueCounts.get(4, 0)
            / projectInformationVariables["surveyProjectInfoNumberCanThinkOfAProject"],
            2,
        )
    )
    projectInformationVariables["surveyProjectInfoPercentOtherMaturity"] = round(
        100
        - projectInformationVariables[
            "surveyProjectInfoPercentEarlyResearchStagesMaturity"
        ]
        - projectInformationVariables["surveyProjectInfoPercentVeryExploratoryMaturity"]
        - projectInformationVariables[
            "surveyProjectInfoPercentSomewhatExploratoryMaturity"
        ]
        - projectInformationVariables[
            "surveyProjectInfoPercentSomewhatProductionizedMaturity"
        ]
        - projectInformationVariables[
            "surveyProjectInfoPercentVeryProductizedMaturity"
        ],
        2,
    )

    projectFTECounts = regularSurveyData.loc[
        (regularSurveyData['CanThinkOfProject'] == "Yes, I can think of a project.")
    ]["Project_FTECount"].value_counts()
    projectInformationVariables["surveyProjectFTEPercentLessThanZeroPointFive"] = round(
        100
        * projectFTECounts.get(0, 0)
        / projectInformationVariables["surveyProjectInfoNumberCanThinkOfAProject"],
        2,
    )
    projectInformationVariables[
        "surveyProjectFTEPercentZeroPointFiveToOnePointFourNine"
    ] = round(
        100
        * projectFTECounts.get(1, 0)
        / projectInformationVariables["surveyProjectInfoNumberCanThinkOfAProject"],
        2,
    )
    projectInformationVariables[
        "surveyProjectFTEPercentOnePointFiveToTwoPointFourNine"
    ] = round(
        100
        * projectFTECounts.get(2, 0)
        / projectInformationVariables["surveyProjectInfoNumberCanThinkOfAProject"],
        2,
    )
    projectInformationVariables[
        "surveyProjectFTEPercentTwoPointFiveToThreePointFourNine"
    ] = round(
        100
        * projectFTECounts.get(3, 0)
        / projectInformationVariables["surveyProjectInfoNumberCanThinkOfAProject"],
        2,
    )
    projectInformationVariables[
        "surveyProjectFTEPercentThreePointFiveToFourPointFourNine"
    ] = round(
        100
        * projectFTECounts.get(4, 0)
        / projectInformationVariables["surveyProjectInfoNumberCanThinkOfAProject"],
        2,
    )
    projectInformationVariables["surveyProjectFTEPercentFourPointFiveOrMore"] = round(
        100
        * projectFTECounts.get(5, 0)
        / projectInformationVariables["surveyProjectInfoNumberCanThinkOfAProject"],
        2,
    )
    projectInformationVariables["surveyProjectFTEPercentDontKnow"] = round(
        100
        - projectInformationVariables["surveyProjectFTEPercentLessThanZeroPointFive"]
        - projectInformationVariables[
            "surveyProjectFTEPercentZeroPointFiveToOnePointFourNine"
        ]
        - projectInformationVariables[
            "surveyProjectFTEPercentOnePointFiveToTwoPointFourNine"
        ]
        - projectInformationVariables[
            "surveyProjectFTEPercentTwoPointFiveToThreePointFourNine"
        ]
        - projectInformationVariables["surveyProjectFTEPercentFourPointFiveOrMore"],
        2,
    )

    projectTeamSizeCounts = regularSurveyData.loc[
        (regularSurveyData['CanThinkOfProject'] == "Yes, I can think of a project.")
    ]["Project_TeamSize"].value_counts()
    projectInformationVariables["surveyProjectTeamSizePercentOnePerson"] = round(
        100
        * projectTeamSizeCounts.get(0, 0)
        / projectInformationVariables["surveyProjectInfoNumberCanThinkOfAProject"],
        2,
    )
    projectInformationVariables["surveyProjectTeamSizePercentTwoToThreePeople"] = round(
        100
        * projectTeamSizeCounts.get(1, 0)
        / projectInformationVariables["surveyProjectInfoNumberCanThinkOfAProject"],
        2,
    )
    projectInformationVariables["surveyProjectTeamSizePercentFourToFivePeople"] = round(
        100
        * projectTeamSizeCounts.get(2, 0)
        / projectInformationVariables["surveyProjectInfoNumberCanThinkOfAProject"],
        2,
    )
    projectInformationVariables["surveyProjectTeamSizePercentSixToTenPeople"] = round(
        100
        * projectTeamSizeCounts.get(3, 0)
        / projectInformationVariables["surveyProjectInfoNumberCanThinkOfAProject"],
        2,
    )
    projectInformationVariables["surveyProjectTeamSizePercentElevenToFifteenPeople"] = (
        round(
            100
            * projectTeamSizeCounts.get(4, 0)
            / projectInformationVariables["surveyProjectInfoNumberCanThinkOfAProject"],
            2,
        )
    )
    projectInformationVariables["surveyProjectTeamSizePercentSixteenToTwentyPeople"] = (
        round(
            100
            * projectTeamSizeCounts.get(5, 0)
            / projectInformationVariables["surveyProjectInfoNumberCanThinkOfAProject"],
            2,
        )
    )
    projectInformationVariables[
        "surveyProjectTeamSizePercentTwentyOneToTwentyFivePeople"
    ] = round(
        100
        * projectTeamSizeCounts.get(6, 0)
        / projectInformationVariables["surveyProjectInfoNumberCanThinkOfAProject"],
        2,
    )
    projectInformationVariables["surveyProjectTeamSizePercentTwentySixOrMorePeople"] = (
        round(
            100
            * projectTeamSizeCounts.get(7, 0)
            / projectInformationVariables["surveyProjectInfoNumberCanThinkOfAProject"],
            2,
        )
    )

    projectInformationVariables["surveyProjectPercentWithOneToFivePeople"] = round(
        100
        * len(regularSurveyData.loc[regularSurveyData['Project_TeamSize'] <= 2].index)
        / projectInformationVariables["surveyProjectInfoNumberCanThinkOfAProject"],
        2,
    )
    projectInformationVariables["surveyProjectPercentWithSixToFifteenPeople"] = round(
        100
        * len(
            regularSurveyData.loc[
                (regularSurveyData['Project_TeamSize'] >= 3)
                & (regularSurveyData['Project_Maturity'] < 5)
            ].index
        )
        / projectInformationVariables["surveyProjectInfoNumberCanThinkOfAProject"],
        2,
    )
    projectInformationVariables["surveyProjectPercentWithSixteenOrMorePeople"] = round(
        100
        * len(regularSurveyData.loc[regularSurveyData['Project_TeamSize'] >= 5].index)
        / projectInformationVariables["surveyProjectInfoNumberCanThinkOfAProject"],
        2,
    )
    projectInformationVariables["surveyProjectPercentWithAtLeastOnePointFiveFTEs"] = (
        round(
            100
            * len(
                regularSurveyData.loc[regularSurveyData['Project_FTECount'] >= 2].index
            )
            / projectInformationVariables["surveyProjectInfoNumberCanThinkOfAProject"],
            2,
        )
    )
    (
        projectInformationVariables[
            "surveyProjectCorrelationBetweenTeamSizeAndFTECountRValue"
        ],
        projectInformationVariables[
            "surveyProjectCorrelationBetweenTeamSizeAndFTECountPValue"
        ],
    ) = scipy.stats.pearsonr(
        regularSurveyData[
            (regularSurveyData['CanThinkOfProject'] == "Yes, I can think of a project.")
            & (regularSurveyData['Project_TeamSize'].notnull())
            & (regularSurveyData['Project_FTECount'].notnull())
        ]["Project_TeamSize"],
        regularSurveyData[
            (regularSurveyData['CanThinkOfProject'] == "Yes, I can think of a project.")
            & (regularSurveyData['Project_TeamSize'].notnull())
            & (regularSurveyData['Project_FTECount'].notnull())
        ]["Project_FTECount"],
    )
    projectInformationVariables[
        "surveyProjectCorrelationBetweenTeamSizeAndFTECountRValue"
    ] = round(
        projectInformationVariables[
            "surveyProjectCorrelationBetweenTeamSizeAndFTECountRValue"
        ],
        3,
    )

    # Project Information: Team Software Engineering Training
    projectInformationVariables[
        "surveyProjectPercentOfProjectsWithNoSoftwareEngineeringTraining"
    ] = round(
        100
        * len(
            regularSurveyData.loc[
                regularSurveyData['Project_TeamSoftwareTraining'] == 0
            ].index
        )
        / projectInformationVariables["surveyProjectInfoNumberCanThinkOfAProject"],
        2,
    )
    projectInformationVariables[
        "surveyProjectPercentOfProjectsWithQuarterOrLessSoftwareEngineeringTraining"
    ] = round(
        100
        * len(
            regularSurveyData.loc[
                regularSurveyData['Project_TeamSoftwareTraining'] == 0.25
            ].index
        )
        / projectInformationVariables["surveyProjectInfoNumberCanThinkOfAProject"],
        2,
    )
    projectInformationVariables[
        "surveyProjectPercentOfProjectsWithHalfSoftwareEngineeringTraining"
    ] = round(
        100
        * len(
            regularSurveyData.loc[
                regularSurveyData['Project_TeamSoftwareTraining'] == 0.5
            ].index
        )
        / projectInformationVariables["surveyProjectInfoNumberCanThinkOfAProject"],
        2,
    )
    projectInformationVariables[
        "surveyProjectPercentOfProjectsWithThreeQuartersSoftwareEngineeringTraining"
    ] = round(
        100
        * len(
            regularSurveyData.loc[
                regularSurveyData['Project_TeamSoftwareTraining'] == 0.75
            ].index
        )
        / projectInformationVariables["surveyProjectInfoNumberCanThinkOfAProject"],
        2,
    )
    projectInformationVariables[
        "surveyProjectPercentOfProjectsWithAllSoftwareEngineeringTraining"
    ] = round(
        100
        * len(
            regularSurveyData.loc[
                regularSurveyData['Project_TeamSoftwareTraining'] == 1
            ].index
        )
        / projectInformationVariables["surveyProjectInfoNumberCanThinkOfAProject"],
        2,
    )
    projectInformationVariables[
        "surveyProjectPercentOfProjectsWithDontKnowSoftwareEngineeringTraining"
    ] = round(
        100
        - projectInformationVariables[
            "surveyProjectPercentOfProjectsWithNoSoftwareEngineeringTraining"
        ]
        - projectInformationVariables[
            "surveyProjectPercentOfProjectsWithQuarterOrLessSoftwareEngineeringTraining"
        ]
        - projectInformationVariables[
            "surveyProjectPercentOfProjectsWithHalfSoftwareEngineeringTraining"
        ]
        - projectInformationVariables[
            "surveyProjectPercentOfProjectsWithThreeQuartersSoftwareEngineeringTraining"
        ]
        - projectInformationVariables[
            "surveyProjectPercentOfProjectsWithAllSoftwareEngineeringTraining"
        ],
        2,
    )

    return projectInformationVariables


def outputBeliefsSectionTable(beliefsVariables):
    beliefsAboutReproducibilityAndSoftwareQuality = [
        "Belief_ReproducibilityIsFundamental",
        "Belief_ReproducibilityEnablesProgress",
        "Belief_HighQualitySoftwareIsMoreReproducible",
        "Belief_ReproducibilityIsMoreImportantThanSpeedOrFeatures",
    ]
    beliefsIntentToAct = [
        "Belief_WillingToInvestExtraInQuality",
        "Belief_WillingToSetUpComplexSolutions",
        "Belief_WillingToSacrificeQuality",
    ]
    beliefsPerceivedBehavioralControl = [
        "Belief_AwareOfBestPractices",
        "Belief_HasKnowledgeToolsAndResources",
        "Belief_CanTakeActionOnReproducibility",
        "Belief_ProjectHasEffectivePractices",
        "Belief_ReproducibilityNotAHindrance",
    ]
    beliefsSubjectiveNorms = [
        "Belief_CommunityValuesReproducibility",
        "Belief_InstitutionValuesReproducibility",
        "Belief_StakeholdersValueReproducibility",
        "Belief_CanLearnFromPeers",
    ]
    beliefRecommendations = [
        "Belief_NeedForStandardGuidelines",
        "Belief_NeedForTrainingOpportunities",
    ]

    def outputRow(
        tableFile,
        category,
        questionText,
        agreePercent,
        mean,
        median,
        distribution,
        rowShouldBeGray,
    ):
        if rowShouldBeGray:
            cellColor = "\cellcolor[HTML]{EFEFEF}"
        else:
            cellColor = ""
        tableFile.write(
            "{category} & {cellColor} \small{{{questionText}}} & {cellColor} {agreePercent}\\% & {cellColor} {mean}\; ({median})\; {distribution} \\\\\n\n".format(
                category=category,
                cellColor=cellColor,
                questionText=questionText,
                agreePercent=agreePercent,
                mean=mean,
                median=median,
                distribution=distribution,
            )
        )

    def outputQuestionBlock(
        tableFile,
        categoryName,
        beliefLabels,
        startingRowIndex,
        isFinalQuestionBlock=False,
    ):
        currentRowIndex = startingRowIndex
        for i in range(len(beliefLabels)):
            if currentRowIndex % 2 == 1:
                rowShouldBeGray = True
            else:
                rowShouldBeGray = False
            if i == 0:
                category = "\multirow{{{numberOfRows}}}{{2cm}}{{\cellcolor[HTML]{{FFFFFF}} {categoryName}}}".format(
                    numberOfRows=len(beliefLabels), categoryName=categoryName
                )
            else:
                category = ""
            beliefLabel = beliefLabels[i]
            beliefQuestionCommandEntry = beliefLabel.replace("Belief_", "")
            questionText = beliefsVariables[
                "surveyBelief{belief}QuestionText".format(
                    belief=beliefQuestionCommandEntry
                )
            ]
            agreePercent = "\surveyBelief{belief}PercentAgreeOrStronglyAgree".format(
                belief=beliefQuestionCommandEntry
            )
            mean = "\surveyBelief{belief}Mean".format(belief=beliefQuestionCommandEntry)
            median = "\surveyBelief{belief}Median".format(
                belief=beliefQuestionCommandEntry
            )
            distribution = "\surveyBelief{belief}Sparkline".format(
                belief=beliefQuestionCommandEntry
            )

            outputRow(
                tableFile,
                category,
                questionText,
                agreePercent,
                mean,
                median,
                distribution,
                rowShouldBeGray,
            )
            currentRowIndex += 1
        if not isFinalQuestionBlock:
            tableFile.write("\midrule\n")
        return currentRowIndex

    with open("tex/beliefsTable.tex", 'w') as tableFile:
        tableFile.write("\\begin{table}[]\n")
        caption = "Summary statistics for responses to the beliefs section of the survey among all respondents. 5-point Likert scale scales converted to range of values [-2,2]."
        tableFile.write("\caption{{{captionText}}}\n".format(captionText=caption))
        tableFile.write(
            "\label{tab:beliefsTable}\n\\begin{tabular}{@{}p{2cm}p{6cm}cc@{}}\n\\toprule\nCategory & Question & Agree(\\%) & Mean (Median) Distribution \\\\ \midrule\n"
        )
        currentRowIndex = 0
        currentRowIndex = outputQuestionBlock(
            tableFile,
            "Attitudes about Reproducibility and Software Quality",
            beliefsAboutReproducibilityAndSoftwareQuality,
            currentRowIndex,
        )
        currentRowIndex = outputQuestionBlock(
            tableFile, "Intent to Act", beliefsIntentToAct, currentRowIndex
        )
        currentRowIndex = outputQuestionBlock(
            tableFile,
            "Perceived Behavioral Control",
            beliefsPerceivedBehavioralControl,
            currentRowIndex,
        )
        currentRowIndex = outputQuestionBlock(
            tableFile, "Subjective Norms", beliefsSubjectiveNorms, currentRowIndex
        )
        currentRowIndex = outputQuestionBlock(
            tableFile,
            "Recommendations",
            beliefRecommendations,
            currentRowIndex,
            isFinalQuestionBlock=True,
        )
        tableFile.write("\\bottomrule\n\end{tabular}\n\end{table}\n")


def generateBeliefsSectionVariables(regularSurveyData):
    beliefsVariables = {}
    beliefQuestions = [
        "Belief_ReproducibilityIsFundamental",
        "Belief_ReproducibilityEnablesProgress",
        "Belief_HighQualitySoftwareIsMoreReproducible",
        "Belief_ReproducibilityIsMoreImportantThanSpeedOrFeatures",
        "Belief_WillingToInvestExtraInQuality",
        "Belief_WillingToSetUpComplexSolutions",
        "Belief_WillingToSacrificeQuality",
        "Belief_AwareOfBestPractices",
        "Belief_HasKnowledgeToolsAndResources",
        "Belief_CanTakeActionOnReproducibility",
        "Belief_ProjectHasEffectivePractices",
        "Belief_ReproducibilityNotAHindrance",
        "Belief_CommunityValuesReproducibility",
        "Belief_InstitutionValuesReproducibility",
        "Belief_StakeholdersValueReproducibility",
        "Belief_CanLearnFromPeers",
        "Belief_NeedForStandardGuidelines",
        "Belief_NeedForTrainingOpportunities",
    ]
    beliefQuestionTexts = {
        "Belief_ReproducibilityIsFundamental": "Science and engineering software needs to be reproducible to be useful.",
        "Belief_ReproducibilityEnablesProgress": "Reproducibility enables scientific progress by improving collaboration and the ability to build upon previous findings.",
        "Belief_HighQualitySoftwareIsMoreReproducible": "High-quality software is more likely to yield reproducible results than low-quality software.",
        "Belief_ReproducibilityIsMoreImportantThanSpeedOrFeatures": "It's more important for software to be reproducible than fast or feature-rich.",
        "Belief_WillingToInvestExtraInQuality": "I’m willing to invest extra effort in ensuring software quality (like usability or maintainability) if it means the software is more reproducible.",
        "Belief_WillingToSetUpComplexSolutions": "I'm willing to invest extra effort in setting up complex solutions (like containers) if it means the software is more reproducible.",
        "Belief_WillingToSacrificeQuality": "I'm willing to sacrifice some aspects of software quality (like performance or portability) if it means the software is more reproducible.",
        "Belief_AwareOfBestPractices": "I am aware of best practices for ensuring reproducibility of my software results.",
        "Belief_HasKnowledgeToolsAndResources": "I have the knowledge, tools, and resources necessary to ensure reproducibility of my software results.",
        "Belief_CanTakeActionOnReproducibility": "I can quickly identify and take action on opportunities for improvement in my code base that would increase reproducibility.",
        "Belief_ProjectHasEffectivePractices": "My project has implemented effective practices for ensuring reproducibility.",
        "Belief_ReproducibilityNotAHindrance": "On my project, reproducibility considerations do not hinder my team's productivity.",
        "Belief_CommunityValuesReproducibility": "Broadly speaking, the communities I am part of value reproducibility.",
        "Belief_InstitutionValuesReproducibility": "My institution values and prioritizes reproducibility of software results.",
        "Belief_StakeholdersValueReproducibility": "My stakeholders (customers, users, etc.)  have a good understanding of the time, cost, and effort required to ensure reproducibility of software results.",
        "Belief_CanLearnFromPeers": "I regularly collaborate with my peers to share and learn about reproducibility best practices.",
        "Belief_NeedForStandardGuidelines": "Computational guidelines and best practices for reproducibility should be standardized across different science and engineering disciplines.",
        "Belief_NeedForTrainingOpportunities": "There should be training and professional development opportunities to help people learn more about best practices for reproducibility.",
    }

    for beliefLabel in beliefQuestions:
        beliefQuestion = beliefLabel.replace("Belief_", "")
        peopleWhoAnsweredThisQuestion = regularSurveyData[[beliefLabel]].dropna()
        beliefsVariables["surveyBelief{belief}Mean".format(belief=beliefQuestion)] = (
            round(np.mean(peopleWhoAnsweredThisQuestion[beliefLabel]), 2)
        )
        beliefsVariables["surveyBelief{belief}Median".format(belief=beliefQuestion)] = (
            round(np.median(peopleWhoAnsweredThisQuestion[beliefLabel]), 2)
        )
        beliefsVariables[
            "surveyBelief{belief}QuestionText".format(belief=beliefQuestion)
        ] = beliefQuestionTexts[beliefLabel]

        beliefQuestionValueCounts = peopleWhoAnsweredThisQuestion[
            beliefLabel
        ].value_counts()
        beliefsVariables[
            "surveyBelief{belief}PercentStronglyDisagree".format(belief=beliefQuestion)
        ] = round(
            100
            * beliefQuestionValueCounts.get(-2, 0)
            / len(peopleWhoAnsweredThisQuestion),
            2,
        )
        beliefsVariables[
            "surveyBelief{belief}PercentDisagree".format(belief=beliefQuestion)
        ] = round(
            100
            * beliefQuestionValueCounts.get(-1, 0)
            / len(peopleWhoAnsweredThisQuestion),
            2,
        )
        beliefsVariables[
            "surveyBelief{belief}PercentNeitherAgreeNorDisagree".format(
                belief=beliefQuestion
            )
        ] = round(
            100
            * beliefQuestionValueCounts.get(0, 0)
            / len(peopleWhoAnsweredThisQuestion),
            2,
        )
        beliefsVariables[
            "surveyBelief{belief}PercentAgree".format(belief=beliefQuestion)
        ] = round(
            100
            * beliefQuestionValueCounts.get(1, 0)
            / len(peopleWhoAnsweredThisQuestion),
            2,
        )
        beliefsVariables[
            "surveyBelief{belief}PercentStronglyAgree".format(belief=beliefQuestion)
        ] = round(
            100
            * beliefQuestionValueCounts.get(2, 0)
            / len(peopleWhoAnsweredThisQuestion),
            2,
        )
        beliefsVariables[
            "surveyBelief{belief}PercentAgreeOrStronglyAgree".format(
                belief=beliefQuestion
            )
        ] = round(
            100
            * (
                beliefQuestionValueCounts.get(1, 0)
                + beliefQuestionValueCounts.get(2, 0)
            )
            / len(peopleWhoAnsweredThisQuestion),
            2,
        )

        beliefsVariables[
            "surveyBelief{belief}FractionStronglyDisagree".format(belief=beliefQuestion)
        ] = round(
            beliefQuestionValueCounts.get(-2, 0) / len(peopleWhoAnsweredThisQuestion), 2
        )
        beliefsVariables[
            "surveyBelief{belief}FractionDisagree".format(belief=beliefQuestion)
        ] = round(
            beliefQuestionValueCounts.get(-1, 0) / len(peopleWhoAnsweredThisQuestion), 2
        )
        beliefsVariables[
            "surveyBelief{belief}FractionNeitherAgreeNorDisagree".format(
                belief=beliefQuestion
            )
        ] = round(
            beliefQuestionValueCounts.get(0, 0) / len(peopleWhoAnsweredThisQuestion), 2
        )
        beliefsVariables[
            "surveyBelief{belief}FractionAgree".format(belief=beliefQuestion)
        ] = round(
            beliefQuestionValueCounts.get(1, 0) / len(peopleWhoAnsweredThisQuestion), 2
        )
        beliefsVariables[
            "surveyBelief{belief}FractionStronglyAgree".format(belief=beliefQuestion)
        ] = round(
            beliefQuestionValueCounts.get(2, 0) / len(peopleWhoAnsweredThisQuestion), 2
        )

        beliefsVariables[
            "surveyBelief{belief}Sparkline".format(belief=beliefQuestion)
        ] = "\\barchartfive{{\{StronglyDisagree}}}{{\{Disagree}}}{{\{NeitherAgreeNorDisagree}}}{{\{Agree}}}{{\{StronglyAgree}}}".format(
            StronglyDisagree="surveyBelief{belief}FractionStronglyDisagree".format(
                belief=beliefQuestion
            ),
            Disagree="surveyBelief{belief}FractionDisagree".format(
                belief=beliefQuestion
            ),
            NeitherAgreeNorDisagree="surveyBelief{belief}FractionNeitherAgreeNorDisagree".format(
                belief=beliefQuestion
            ),
            Agree="surveyBelief{belief}FractionAgree".format(belief=beliefQuestion),
            StronglyAgree="surveyBelief{belief}FractionStronglyAgree".format(
                belief=beliefQuestion
            ),
        )
    outputBeliefsSectionTable(beliefsVariables)
    return beliefsVariables


def generateReproducibilityNeedVariables(regularSurveyData):
    reproducibilityNeedVariables = {}
    reproducibilityNeedOverTime = [
        "ReproducibilityNeed_NeededForMonths",
        "ReproducibilityNeed_NeededForYears",
        "ReproducibilityNeed_NeededForDecades",
    ]
    reproducibilityNeedVsQuality = [
        'ReproducibilityNeed_SelfUse',
        'ReproducibilityNeed_Maintainability',
        'ReproducibilityNeed_Reliability',
        'ReproducibilityNeed_Usability',
        'ReproducibilityNeed_FunctionalSuitability',
        'ReproducibilityNeed_Portability',
        'ReproducibilityNeed_Compatibility',
        'ReproducibilityNeed_PerformanceEfficiency',
        'ReproducibilityNeed_Security',
    ]

    importanceScoreDict = {
        0: "Not Important",
        1: "Somewhat",
        2: "Moderately",
        3: "Important",
        4: "Very",
    }

    respondentsWhoCouldThinkOfAProject = regularSurveyData.loc[
        (regularSurveyData['CanThinkOfProject'] == "Yes, I can think of a project.")
    ]
    for reproducibilityNeedLabel in (
        reproducibilityNeedVsQuality + reproducibilityNeedOverTime
    ):
        category = reproducibilityNeedLabel.replace("ReproducibilityNeed_", "")
        reproducibilityNeedVariables[
            "surveyReproducibilityNeed{category}Mean".format(category=category)
        ] = round(
            np.mean(respondentsWhoCouldThinkOfAProject[reproducibilityNeedLabel]), 3
        )
        reproducibilityNeedVariables[
            "surveyReproducibilityNeed{category}Median".format(category=category)
        ] = int(respondentsWhoCouldThinkOfAProject[reproducibilityNeedLabel].median())
        reproducibilityNeedVariables[
            "surveyReproducibilityNeed{category}MedianText".format(category=category)
        ] = importanceScoreDict[
            reproducibilityNeedVariables[
                "surveyReproducibilityNeed{category}Median".format(category=category)
            ]
        ]

        valueCounts = respondentsWhoCouldThinkOfAProject[
            reproducibilityNeedLabel
        ].value_counts()
        reproducibilityNeedVariables[
            "surveyReproducibilityNeed{category}NotImportant".format(category=category)
        ] = valueCounts.get(0, 0) / len(respondentsWhoCouldThinkOfAProject)
        reproducibilityNeedVariables[
            "surveyReproducibilityNeed{category}SomewhatImportant".format(
                category=category
            )
        ] = valueCounts.get(1, 0) / len(respondentsWhoCouldThinkOfAProject)
        reproducibilityNeedVariables[
            "surveyReproducibilityNeed{category}ModeratelyImportant".format(
                category=category
            )
        ] = valueCounts.get(2, 0) / len(respondentsWhoCouldThinkOfAProject)
        reproducibilityNeedVariables[
            "surveyReproducibilityNeed{category}Important".format(category=category)
        ] = valueCounts.get(3, 0) / len(respondentsWhoCouldThinkOfAProject)
        reproducibilityNeedVariables[
            "surveyReproducibilityNeed{category}VeryImportant".format(category=category)
        ] = valueCounts.get(4, 0) / len(respondentsWhoCouldThinkOfAProject)

        reproducibilityNeedVariables[
            "surveyReproducibilityNeed{category}PercentImportantOrVeryImportant".format(
                category=category
            )
        ] = round(
            100
            * (
                reproducibilityNeedVariables[
                    "surveyReproducibilityNeed{category}Important".format(
                        category=category
                    )
                ]
                + reproducibilityNeedVariables[
                    "surveyReproducibilityNeed{category}VeryImportant".format(
                        category=category
                    )
                ]
            ),
            2,
        )

        reproducibilityNeedVariables[
            "surveyReproducibilityNeed{category}Sparkline".format(category=category)
        ] = "\\barchartfive{{\{NotImportant}}}{{\{SomewhatImportant}}}{{\{ModeratelyImportant}}}{{\{Important}}}{{\{VeryImportant}}}".format(
            NotImportant="surveyReproducibilityNeed{category}NotImportant".format(
                category=category
            ),
            SomewhatImportant="surveyReproducibilityNeed{category}SomewhatImportant".format(
                category=category
            ),
            ModeratelyImportant="surveyReproducibilityNeed{category}ModeratelyImportant".format(
                category=category
            ),
            Important="surveyReproducibilityNeed{category}Important".format(
                category=category
            ),
            VeryImportant="surveyReproducibilityNeed{category}VeryImportant".format(
                category=category
            ),
        )
    return reproducibilityNeedVariables


def generateMaxDiffQualityUtilityVariables(qualityUtilitiesData):
    maxDiffQualityUtilityVariables = {}
    for i, row in qualityUtilitiesData.iterrows():
        maxDiffQualityUtilityVariables[
            "surveyMaxDiff{quality}Utility".format(quality=row["QualityAttribute"])
        ] = round(row["Utility"], 3)
    return maxDiffQualityUtilityVariables


def generateMaxDiffQualityFrequencyVariables(maxDiffSurveyData):
    maxDiffQualityFrequencyVariables = {}
    bestChoicesCounts = maxDiffSurveyData["MaxDiff_BestChoice"].dropna().value_counts()
    worstChoicesCounts = (
        maxDiffSurveyData["MaxDiff_WorstChoice"].dropna().value_counts()
    )

    qualityCharacteristics = [
        "Compatibility",
        "FunctionalSuitability",
        "Maintainability",
        "PerformanceEfficiency",
        "Portability",
        "Reliability",
        "Security",
        "Usability",
    ]
    for qualityCharacteristic in qualityCharacteristics:
        maxDiffQualityFrequencyVariables[
            "surveyMaxDiff{quality}MostUsefulPercent".format(
                quality=qualityCharacteristic
            )
        ] = round(
            100
            * bestChoicesCounts[qualityCharacteristic]
            / len(maxDiffSurveyData["MaxDiff_BestChoice"].notnull()),
            2,
        )
        maxDiffQualityFrequencyVariables[
            "surveyMaxDiff{quality}LeastUsefulPercent".format(
                quality=qualityCharacteristic
            )
        ] = round(
            100
            * worstChoicesCounts[qualityCharacteristic]
            / len(maxDiffSurveyData["MaxDiff_WorstChoice"].notnull()),
            2,
        )
    return maxDiffQualityFrequencyVariables


def performAnalysesForPaper(regularSurveyData, maxDiffSurveyData, qualityUtilitiesData):
    """
    Performs analyses on the survey data, producing logs of what was done, various charts and figures, and
    a surveyDataVariables.tex file which can be uploaded to the Overleaf document to update the paper with
    the latest information.
    """

    # paperVariables is a dictionary that stores information to be used in the paper.
    # For example, "surveyStartDate : 'June 2023'" will be converted to "\newcommand \surveyStartDate{June 2023}"
    # in surveyDataVariables.tex, which can then be called anywhere in the paper to insert this information.
    paperVariables = {}

    paperVariables = (
        paperVariables
        | generateDemographicVariables(regularSurveyData, maxDiffSurveyData)
        | generateProjectInformationVariables(regularSurveyData, maxDiffSurveyData)
        | generateBeliefsSectionVariables(regularSurveyData)
        | generateReproducibilityNeedVariables(regularSurveyData)
        | generateMaxDiffQualityFrequencyVariables(maxDiffSurveyData)
        | generateMaxDiffQualityUtilityVariables(qualityUtilitiesData)
    )

    # regularSurveyData = sumTogetherMaxDiffChoicesAndAppendToRegularSurveyData(regularSurveyData,maxDiffSurveyData)
    theoryOfPlannedBehaviorAnalysis(regularSurveyData)

    testWhetherProjectCharacteristicsInfluenceQualityPriorities(
        regularSurveyData,
        characteristicsToTest=[
            "Project_Maturity",
            "Project_TeamSize",
            "Project_FTECount",
            "Project_TeamSoftwareTraining",
            "construct_Intention",
        ],
        alpha=0.05,
    )
    # testWhetherProjectCharacteristicsInfluenceQualityPriorities(regularSurveyData, characteristicsToTest=["Belief_HighQualitySoftwareIsMoreReproducible","Belief_ReproducibilityIsMoreImportantThanSpeedOrFeatures"],alpha=0.05)

    maxDiffChoiceExploration(regularSurveyData, maxDiffSurveyData, qualityUtilitiesData)
    # correlationMiningTest(regularSurveyData,statisticThreshold=0.00001,levelOfSignificanceThreshold=0.05)

    print("-----------------------------")
    # reproducibilityNeeds = ["ReproducibilityNeed_Compatibility","ReproducibilityNeed_FunctionalSuitability","ReproducibilityNeed_Maintainability","ReproducibilityNeed_PerformanceEfficiency",
    #    "ReproducibilityNeed_Portability","ReproducibilityNeed_Reliability","ReproducibilityNeed_Security","ReproducibilityNeed_Usability"]
    # maxDiffQualityUtilityScores = ["MaxDiffQualityUtilityScore_Compatibility","MaxDiffQualityUtilityScore_FunctionalSuitability","MaxDiffQualityUtilityScore_Maintainability","MaxDiffQualityUtilityScore_PerformanceEfficiency",
    #    "MaxDiffQualityUtilityScore_Portability","MaxDiffQualityUtilityScore_Reliability","MaxDiffQualityUtilityScore_Security","MaxDiffQualityUtilityScore_Usability"]

    for qualityCharacteristic in [
        "Compatibility",
        "FunctionalSuitability",
        "Maintainability",
        "PerformanceEfficiency",
        "Portability",
        "Reliability",
        "Security",
        "Usability",
    ]:
        tmp = regularSurveyData[
            (
                regularSurveyData[
                    "ReproducibilityNeed_" + qualityCharacteristic
                ].notnull()
            )
            & (
                regularSurveyData[
                    "MaxDiffQualityUtilityScore_" + qualityCharacteristic
                ].notnull()
            )
        ]
        statistic, pvalue = scipy.stats.pearsonr(
            tmp["ReproducibilityNeed_" + qualityCharacteristic],
            tmp["MaxDiffQualityUtilityScore_" + qualityCharacteristic],
        )
        if pvalue <= 0.05:
            print(
                "ReproducibilityNeed_" + qualityCharacteristic,
                "MaxDiffQualityUtilityScore_" + qualityCharacteristic,
                "Pearson's r:",
                statistic,
                "p-value",
                pvalue,
            )

    for qualityCharacteristic in [
        "Compatibility",
        "FunctionalSuitability",
        "Maintainability",
        "PerformanceEfficiency",
        "Portability",
        "Reliability",
        "Security",
        "Usability",
    ]:
        tmp = regularSurveyData[
            (
                regularSurveyData[
                    "ReproducibilityNeed_" + qualityCharacteristic
                ].notnull()
            )
            & (regularSurveyData["QualityPriority_" + qualityCharacteristic].notnull())
        ]
        statistic, pvalue = scipy.stats.pearsonr(
            tmp["ReproducibilityNeed_" + qualityCharacteristic],
            tmp["QualityPriority_" + qualityCharacteristic],
        )
        if pvalue <= 0.05:
            print(
                "ReproducibilityNeed_" + qualityCharacteristic,
                "QualityPriority_" + qualityCharacteristic,
                "Pearson's r:",
                statistic,
                "p-value",
                pvalue,
            )

    for qualityCharacteristic in [
        "Compatibility",
        "FunctionalSuitability",
        "Maintainability",
        "PerformanceEfficiency",
        "Portability",
        "Reliability",
        "Security",
        "Usability",
    ]:
        tmp = regularSurveyData[
            (
                regularSurveyData[
                    "MaxDiffQualityUtilityScore_" + qualityCharacteristic
                ].notnull()
            )
            & (regularSurveyData["QualityPriority_" + qualityCharacteristic].notnull())
        ]
        statistic, pvalue = scipy.stats.pearsonr(
            tmp["MaxDiffQualityUtilityScore_" + qualityCharacteristic],
            tmp["QualityPriority_" + qualityCharacteristic],
        )
        if pvalue <= 0.05:
            print(
                "MaxDiffQualityUtilityScore_" + qualityCharacteristic,
                "QualityPriority_" + qualityCharacteristic,
                "Pearson's r:",
                statistic,
                "p-value",
                pvalue,
            )

    print("-----------------------------")
    # respondentsWhoCouldThinkOfAProject = regularSurveyData.loc[
    #     (regularSurveyData['CanThinkOfProject']
    #      == "Yes, I can think of a project.")
    # ]
    # respondentsWhoCouldThinkOfAProject.to_csv(
    #     "COMBINEDTMP_OnlyPeopleWithProjects.csv", index=False
    # )

    correlationMiningTest(
        regularSurveyData,
        statisticThreshold=0.001,
        levelOfSignificanceThreshold=0.05,
        filterByIndices=[
            "Belief_ReproducibilityIsFundamental",
            "Belief_ReproducibilityEnablesProgress",
            "Belief_HighQualitySoftwareIsMoreReproducible",
            "Belief_ReproducibilityIsMoreImportantThanSpeedOrFeatures",
            "Belief_WillingToInvestExtraInQuality",
            "Belief_WillingToSetUpComplexSolutions",
            "Belief_WillingToSacrificeQuality",
            "Belief_AwareOfBestPractices",
            "Belief_HasKnowledgeToolsAndResources",
            "Belief_CanTakeActionOnReproducibility",
            "Belief_ProjectHasEffectivePractices",
            "Belief_ReproducibilityNotAHindrance",
            "Belief_CommunityValuesReproducibility",
            "Belief_InstitutionValuesReproducibility",
            "Belief_StakeholdersValueReproducibility",
            "Belief_CanLearnFromPeers",
            "Belief_NeedForStandardGuidelines",
            "Belief_NeedForTrainingOpportunities",
        ],
        bonferroniCorrectionFactor=18,
    )

    print("-----------------------------")
    correlationMiningTest(
        regularSurveyData, statisticThreshold=0.001, levelOfSignificanceThreshold=0.05
    )

    return paperVariables


def outputVariablesToFile(variables, fileName="surveyDataVariables.tex"):
    with open(fileName, 'w') as variablesFile:
        for variableName in variables:
            variablesFile.write(
                "\\newcommand \\{varName}{{{varContent}}}\n".format(
                    varName=variableName, varContent=variables[variableName]
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Perform various analyses on the survey data, produce relevant figures and info for survey research paper.'
    )
    parser.add_argument(
        'regularSurveyDataTransformed',
        type=str,
        help='The regularSurveyData file produced by transformSurveyData.py',
    )
    parser.add_argument(
        'maxDiffSurveyDataTransformed',
        type=str,
        help='The maxDiffSurveyData file produced by transformSurveyData.py',
    )
    parser.add_argument(
        'qualityUtilities',
        type=str,
        help='The estimated overall utilities for software quality characteristics produced by analyzeMaxDiffUtilities.py',
    )
    parser.add_argument(
        'qualityUtilitiesPerPerson',
        type=str,
        help='The estimated utilities per person for software quality characteristics produced by analyzeMaxDiffUtilities.py',
    )
    args = parser.parse_args()
    regularSurveyData = pd.read_csv(args.regularSurveyDataTransformed)
    maxDiffSurveyData = pd.read_csv(args.maxDiffSurveyDataTransformed)
    qualityUtilitiesData = pd.read_csv(args.qualityUtilities)
    qualityUtilitiesPerPersonData = pd.read_csv(args.qualityUtilitiesPerPerson)

    # Add in quality utilities per person
    qualityUtilitiesPerPersonDict = {
        "Compatibility": "MaxDiffQualityUtilityScore_Compatibility",
        "FunctionalSuitability": "MaxDiffQualityUtilityScore_FunctionalSuitability",
        "Maintainability": "MaxDiffQualityUtilityScore_Maintainability",
        "PerformanceEfficiency": "MaxDiffQualityUtilityScore_PerformanceEfficiency",
        "Portability": "MaxDiffQualityUtilityScore_Portability",
        "Reliability": "MaxDiffQualityUtilityScore_Reliability",
        "Security": "MaxDiffQualityUtilityScore_Security",
        "Usability": "MaxDiffQualityUtilityScore_Usability",
    }
    qualityUtilitiesPerPersonData = qualityUtilitiesPerPersonData.rename(
        columns=qualityUtilitiesPerPersonDict
    )
    regularSurveyData = regularSurveyData.merge(
        qualityUtilitiesPerPersonData[
            list(qualityUtilitiesPerPersonDict.values()) + ['ResponseID']
        ],
        on=['ResponseID'],
        how='left',
    )
    os.makedirs("tex", exist_ok=True)
    paperVariables = performAnalysesForPaper(
        regularSurveyData, maxDiffSurveyData, qualityUtilitiesData
    )
    outputVariablesToFile(paperVariables, fileName="tex/surveyDataVariables.tex")
