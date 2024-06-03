"""
Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
"""

import argparse
import pandas as pd
import numpy as np


def transformSurveyData(regularSurveyData, maxDiffSurveyData):
    yearsOfExperienceDict = {
        "Less than a year": 0,
        "1-5 years": 1,
        "6-10 years": 2,
        "11-15 years": 3,
        "16-20 years": 4,
        "20+ years": 5,
    }
    regularSurveyData["YearsOfExperience"] = regularSurveyData["YearsOfExperience"].map(
        yearsOfExperienceDict
    )

    levelOfEducationDict = {
        "High school diploma or equivalent": 0,
        "Associate's degree (e.g., community college degree)": 1,
        "Bachelor's degree": 2,
        "Master's degree": 3,
        "Doctorate degree (Ph.D., MD, JD, etc.)": 4,
    }
    regularSurveyData["LevelOfEducation"] = regularSurveyData["LevelOfEducation"].map(
        levelOfEducationDict
    )

    softwareDevelopmentExperienceDict = {
        "I have no experience or knowledge in software development.": 0,
        "I have a limited understanding and would need assistance with software development tasks.": 1,
        "I have a moderate level of experience and can complete basic tasks independently.": 2,
        "I have a significant amount of experience and can handle complex tasks.": 3,
        "I am highly experienced and could mentor others in software development.": 4,
    }
    regularSurveyData["SoftwareDevelopmentExperience"] = regularSurveyData[
        "SoftwareDevelopmentExperience"
    ].map(softwareDevelopmentExperienceDict)

    projectMaturityDict = {
        "Early research stages": 0,
        "Very exploratory (e.g., proof of principle)": 1,
        "Somewhat exploratory (e.g., working prototype)": 2,
        "Somewhat productionized (e.g., somewhat stable but actively evolving)": 3,
        "Very productionized (e.g., regularly released, maintained)": 4,
    }
    regularSurveyData["Project_Maturity"] = regularSurveyData["Project_Maturity"].map(
        projectMaturityDict
    )

    projectTeamSizeDict = {
        "1 Person": 0,
        "2-3 People": 1,
        "4-5 People": 2,
        "6-10 People": 3,
        "11-15 People": 4,
        "16-20 People": 5,
        "21-25 People": 6,
        "26+ People": 7,
    }
    regularSurveyData["Project_TeamSize"] = regularSurveyData["Project_TeamSize"].map(
        projectTeamSizeDict
    )

    projectFTECountDict = {
        "I Don't Know": '',
        "Less than 0.5": 0,
        "0.5 to 1.49": 1,
        "1.5 to 2.49": 2,
        "2.5 to 3.49": 3,
        "3.5 to 4.49": 4,
        "4.5+": 5,
    }
    regularSurveyData["Project_FTECount"] = regularSurveyData["Project_FTECount"].map(
        projectFTECountDict
    )

    projectTeamSoftwareTrainingDict = {
        "I Don't Know": '',
        "None": 0,
        "A quarter or less": 0.25,
        "About half": 0.5,
        "Three-quarters or more": 0.75,
        "All": 1,
    }
    regularSurveyData["Project_TeamSoftwareTraining"] = regularSurveyData[
        "Project_TeamSoftwareTraining"
    ].map(projectTeamSoftwareTrainingDict)

    # Scaled Borda Count transform of software quality priorities
    # Note: Here we scale all the values to sum to 1, that way people don't more 'points' for selecting additional qualities.
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

    qualityRankBackupDict = {
        "QualityPriority_Maintainability": "QualityPriority_Maintainability_OriginalRank",
        "QualityPriority_Portability": "QualityPriority_Portability_OriginalRank",
        "QualityPriority_Reliability": "QualityPriority_Reliability_OriginalRank",
        "QualityPriority_Usability": "QualityPriority_Usability_OriginalRank",
        "QualityPriority_Compatibility": "QualityPriority_Compatibility_OriginalRank",
        "QualityPriority_PerformanceEfficiency": "QualityPriority_PerformanceEfficiency_OriginalRank",
        "QualityPriority_FunctionalSuitability": "QualityPriority_FunctionalSuitability_OriginalRank",
        "QualityPriority_Security": "QualityPriority_Security_OriginalRank",
    }
    for priority in qualityRankBackupDict:
        regularSurveyData[qualityRankBackupDict[priority]] = regularSurveyData[priority]

    def bordaCount(row):
        for quality in qualityPriorityRankCategories:
            if pd.isna(row[quality]):
                row[quality] = 0
            else:
                row[quality] = len(qualityPriorityRankCategories) - row[quality]
        return row

    def normalizeBordaScores(row):
        if np.sum(row[qualityPriorityRankCategories]) > 0:
            return row[qualityPriorityRankCategories] / np.sum(
                row[qualityPriorityRankCategories]
            )
        else:
            return row

    for i, row in regularSurveyData.iterrows():
        regularSurveyData.loc[i] = bordaCount(row)
        regularSurveyData.loc[i, qualityPriorityRankCategories] = normalizeBordaScores(
            row
        )

    agreementDict = {
        "Strongly Disagree": -2,
        "Disagree": -1,
        "Neither Agree Nor Disagree": 0,
        "Agree": 1,
        "Strongly Agree": 2,
    }
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
    for question in beliefQuestions:
        regularSurveyData[question] = regularSurveyData[question].map(agreementDict)

    importanceDict = {
        "Not Important": 0,
        "Somewhat Important": 1,
        "Moderately Important": 2,
        "Important": 3,
        "Very Important": 4,
    }
    reproducibilityNeedQuestions = [
        "ReproducibilityNeed_NeededForMonths",
        "ReproducibilityNeed_NeededForYears",
        "ReproducibilityNeed_NeededForDecades",
        "ReproducibilityNeed_SelfUse",
        "ReproducibilityNeed_Usability",
        "ReproducibilityNeed_Portability",
        "ReproducibilityNeed_Maintainability",
        "ReproducibilityNeed_Reliability",
        "ReproducibilityNeed_FunctionalSuitability",
        "ReproducibilityNeed_PerformanceEfficiency",
        "ReproducibilityNeed_Compatibility",
        "ReproducibilityNeed_Security",
    ]
    for question in reproducibilityNeedQuestions:
        regularSurveyData[question] = regularSurveyData[question].map(importanceDict)

    maxDiffChoicesDict = {
        "Maintainable": "Maintainability",
        "Portable": "Portability",
        "Reliable": "Reliability",
        "Usable": "Usability",
        "Compatible": "Compatibility",
        "Performance Efficient": "PerformanceEfficiency",
        "Functionally Suitable": "FunctionalSuitability",
        "Secure": "Security",
    }

    # Remove non-answers from max diff data (people who didn't have a project to report end up with empty rows in the max diff CSV file)
    maxDiffSurveyData = maxDiffSurveyData.loc[
        (maxDiffSurveyData['MaxDiff_AttributeChoice1'].notnull())
        & (maxDiffSurveyData['MaxDiff_AttributeChoice2'].notnull())
        & (maxDiffSurveyData['MaxDiff_AttributeChoice3'].notnull())
    ]

    # Rename max diff choices to be easier to work with
    for maxDiffAttribute in [
        "MaxDiff_AttributeChoice1",
        "MaxDiff_AttributeChoice2",
        "MaxDiff_AttributeChoice3",
        "MaxDiff_BestChoice",
        "MaxDiff_WorstChoice",
    ]:
        for choice in maxDiffChoicesDict:
            maxDiffSurveyData[maxDiffAttribute] = maxDiffSurveyData[
                maxDiffAttribute
            ].str.replace(
                r"(^.*{text}.*$)".format(text=choice), maxDiffChoicesDict[choice]
            )

    # Remove new lines from strings.
    regularSurveyData = regularSurveyData.replace(r'\n', ' ', regex=True)
    regularSurveyData = regularSurveyData.replace(r'\'', ' ', regex=True)
    regularSurveyData = regularSurveyData.replace(r'"', ' ', regex=True)

    unlabledDescriptionData = pd.DataFrame()
    respondentsWhoCouldThinkOfAProject = regularSurveyData.loc[
        (regularSurveyData['CanThinkOfProject'] == "Yes, I can think of a project.")
        & (regularSurveyData["SurveyBranch"].notnull())
        & (regularSurveyData["ResponseID"].notnull())
        & (regularSurveyData["Project_Description"].notnull())
    ]

    unlabledDescriptionData["SurveyBranch"] = respondentsWhoCouldThinkOfAProject[
        "SurveyBranch"
    ].values
    unlabledDescriptionData["ResponseID"] = respondentsWhoCouldThinkOfAProject[
        "ResponseID"
    ].values
    unlabledDescriptionData["Project_Description"] = respondentsWhoCouldThinkOfAProject[
        "Project_Description"
    ].values
    unlabledDescriptionData["Project_Category"] = ''

    regularSurveyData = regularSurveyData.fillna(
        ''
    )  # Replace any NaNs with the empty string ''.
    maxDiffSurveyData = maxDiffSurveyData.fillna(
        ''
    )  # Replace any NaNs with the empty string ''.

    return regularSurveyData, unlabledDescriptionData, maxDiffSurveyData


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Transform CSV survey data into a form suitable for quantitative analysis, such as converting ordinal responses into numeric data, .'
    )
    parser.add_argument(
        'regularSurveyData',
        type=str,
        help='The regularSurveyData file produced by combineSurveyData.py',
    )
    parser.add_argument(
        'maxDiffSurveyData',
        type=str,
        help='The maxDiffSurveyData file produced by combineSurveyData.py',
    )
    args = parser.parse_args()
    regularSurveyData = pd.read_csv(args.regularSurveyData)
    maxDiffSurveyData = pd.read_csv(args.maxDiffSurveyData)
    (outputRegularSurveyData, unlabledDescriptionData, outputMaxDiffSurveyData) = (
        transformSurveyData(regularSurveyData, maxDiffSurveyData)
    )
    outputRegularSurveyData.to_csv(
        "data/regularSurveyData_transformed.csv", index=False
    )
    # outputRegularSurveyData.to_csv(
    #     "labelsForProjectDescriptions_blank.csv", index=False
    # )
    outputMaxDiffSurveyData.to_csv(
        "data/maxDiffSurveyData_transformed.csv", index=False
    )
