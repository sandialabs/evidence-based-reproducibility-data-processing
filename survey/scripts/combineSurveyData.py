"""
Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
"""

import argparse
import pandas as pd
import os

software_quality_strings_dict = {
    "Compatible (Can more easily be run in combination with other software; more interoperable and able to run alongside other software without causing issues.)": 'Compatibility',
    "Functionally Suitable (Better able to fulfill all its intended needs; more feature complete, more correct, and able to cover more potential use cases.)": 'FunctionalSuitability',
    "Maintainable (Easier to improve, correct, and adapt to change; more modifiable, reusable, analyzable, testable and modular)": 'Maintainability',
    "Performance Efficient (Can run more quickly and efficiently; faster more resource efficient, and performs better even at its maximum limits. )": 'PerformanceEfficiency',
    "Portable (Can be run in more current/future hardware and software environments; more adaptable, easier to install)": 'Portability',
    "Reliable (Can run for a long time without breaking or failing; more mature, fault tolerant, able to recover from errors, and available whenever it's needed.)": 'Reliability',
    "Secure (Better able to protect information and data from unwanted access or manipulation; able to prove users' identities, protect against tampering, and to track any malicious activity. )": 'Security',
    "Usable (Can more easily be used by different people;  more self-explanatory, easier to learn and use, able to guard against user error,  and more accessible and convenient to use)": 'Usability',
}

questionTexts = {
    # Survey Response ID
    "ResponseID": ["ResponseID"],
    # How would you describe your job position (check all that apply)?
    "Role_SoftwareDeveloper": [
        "I develop software for science or engineering applications"
    ],
    "Role_SoftwareUser": ["I use science and engineering software developed by others"],
    "Role_Researcher": ["I conduct research"],
    "Role_Engineer": ["I engage in engineering activities"],
    "Role_Manager": [
        "I manage or supervise the work of people who do any of the above"
    ],
    # How many years of professional experience do you have?
    "YearsOfExperience": ["How many years of professional experience do you have?"],
    # What is the highest level of education you have completed?
    "LevelOfEducation": ["What is the highest level of education you have completed?"],
    # Considering your experience with software development, which of the following best describes your level of competence?
    "SoftwareDevelopmentExperience": [
        "Considering your experience with software development, which of the following best describes your level of competence?",
        "softwareDevelopmentSkill",
    ],
    # What kind of institution do you work for?
    "Institution": ["What kind of institution do you work for?"],
    # Are you able to think of a computational science or engineering software project that you have contributed to and are knowledgeable about?
    "CanThinkOfProject": [
        "Are you able to think of a computational science or engineering software project that you have contributed to and are knowledgeable about?"
    ],
    # Think of a software project that you have contributed to or interacted with regularly. Please briefly describe the project or software (e.g., scientific domain, purpose, goal).
    "Project_Description": [
        "Think of a software project that you have contributed to or interacted with regularly. Please briefly describe the project or software (e.g., scientific domain, purpose, goal)."
    ],
    # Software projects can range from experimental prototypes to mature, production-oriented code-bases. Which of the following best describes your project?
    "Project_Maturity": [
        "Software projects can range from experimental prototypes to mature, production-oriented code-bases. Which of the following best describes your project?"
    ],
    # Around how many team members are/were on the project?
    "Project_TeamSize": ["Around how many team members are/were on the project?"],
    # Around how many full-time employees (FTEs) are/were allocated to the project?
    "Project_FTECount": [
        "Around how many full-time employees (FTEs) are/were allocated to the project?"
    ],
    # What percentage of your team has/had formal background or training in software development?
    "Project_TeamSoftwareTraining": [
        "What percentage of your team has/had formal background or training in software development?"
    ],
    # What software product qualities do you believe are most important to the project you chose?
    "QualityPriority_Maintainability": [
        "Maintainability:What software product qualities do you believe are most important to the project you chose?",
        "The software is easier to improve, correct, and adapt to change.",
    ],
    "QualityPriority_Portability": [
        "Portability:What software product qualities do you believe are most important to the project you chose?",
        "The software can be run in many different hardware and software environments.",
    ],
    "QualityPriority_Reliability": [
        "Reliability:What software product qualities do you believe are most important to the project you chose?",
        "The software can perform as intended for as  long as needed without failing or breaking.",
    ],
    "QualityPriority_Usability": [
        "Usability:What software product qualities do you believe are most important to the project you chose?",
        "The software can easily be used by different people to achieve their goals.",
    ],
    "QualityPriority_Compatibility": [
        "Compatibility:What software product qualities do you believe are most important to the project you chose?",
        "The software can easily be used in combination with other software.",
    ],
    "QualityPriority_PerformanceEfficiency": [
        "Performance Efficiency:What software product qualities do you believe are most important to the project you chose?",
        "The software is able to run quickly and efficiently. Performant software tends to be faster, more resource efficient, and runs well even at its maximum limits.",
    ],
    "QualityPriority_FunctionalSuitability": [
        "Functional Suitability:What software product qualities do you believe are most important to the project you chose?",
        "The software is able to fulfill all its intended needs.",
    ],
    "QualityPriority_Security": [
        "Security:What software product qualities do you believe are most important to the project you chose?",
        "The software is able to protect information and data from unwanted access or manipulation.",
    ],
    # The following questions help us understand your views on reproducibility and the extent to which you feel able and supported in ensuring it.
    "Belief_ReproducibilityIsFundamental": [
        "Science and engineering software needs to be reproducible to be useful.",
        "Computational reproducibility is a fundamental requirement for science and engineering software.",
    ],
    "Belief_ReproducibilityEnablesProgress": [
        "Reproducibility enables scientific progress by improving collaboration and the ability to build upon previous findings."
    ],
    "Belief_HighQualitySoftwareIsMoreReproducible": [
        "High-quality software is more likely to yield reproducible results than low-quality software."
    ],
    "Belief_ReproducibilityIsMoreImportantThanSpeedOrFeatures": [
        "It's more important for software to be reproducible than fast or feature-rich."
    ],
    "Belief_WillingToInvestExtraInQuality": [
        "I’m willing to invest extra effort in ensuring software quality (like usability or maintainability) if it means the software is more reproducible."
    ],
    "Belief_WillingToSetUpComplexSolutions": [
        "willing to invest extra effort in setting up complex solutions (like containers) if it means the software is more reproducible."
    ],
    "Belief_WillingToSacrificeQuality": [
        "I’m willing to sacrifice some aspects of software quality (like performance or portability) if it means the software is more reproducible."
    ],
    "Belief_AwareOfBestPractices": [
        "I am aware of best practices for ensuring reproducibility of my software results."
    ],
    "Belief_HasKnowledgeToolsAndResources": [
        "I have the knowledge, tools, and resources necessary to ensure reproducibility of my software results."
    ],
    "Belief_CanTakeActionOnReproducibility": [
        "I can quickly identify and take action on opportunities for improvement in my code base that would increase reproducibility."
    ],
    "Belief_ProjectHasEffectivePractices": [
        "My project has implemented effective practices for ensuring reproducibility."
    ],
    "Belief_ReproducibilityNotAHindrance": [
        "On my project, reproducibility considerations do not hinder my team's productivity."
    ],
    "Belief_CommunityValuesReproducibility": [
        "Broadly speaking, the communities I am part of value reproducibility."
    ],
    "Belief_InstitutionValuesReproducibility": [
        "My institution values and prioritizes reproducibility of software results."
    ],
    "Belief_StakeholdersValueReproducibility": [
        "My stakeholders (customers, users, etc.)  have a good understanding of the time, cost, and effort required to ensure reproducibility of software results."
    ],
    "Belief_CanLearnFromPeers": [
        "I regularly collaborate with my peers to share and learn about reproducibility best practices."
    ],
    "Belief_NeedForStandardGuidelines": [
        "Computational guidelines and best practices for reproducibility should be standardized across different science and engineering disciplines."
    ],
    "Belief_NeedForTrainingOpportunities": [
        "There should be training and professional development opportunities to help people learn more about best practices for reproducibility."
    ],
    # "Good enough" reproducibility can mean different things to different people. How important are each of the following to you?
    "ReproducibilityNeed_NeededForMonths": [
        "ReproducibilityMonths",
        "Results from the software should remain reproducible <strong>months</strong> after the original run",
    ],
    "ReproducibilityNeed_NeededForYears": [
        "ReproducibilityYears",
        "Results from the software should remain reproducible&nbsp;<strong>years</strong>&nbsp;after the original run",
    ],
    "ReproducibilityNeed_NeededForDecades": [
        "ReproducibilityDecades",
        "Results from the software should remain reproducible&nbsp;<strong>decades</strong>&nbsp;after the original run",
    ],
    "ReproducibilityNeed_SelfUse": [
        "SelfUseCheck",
        "Results from the software need to be reproducible <strong>by the development team themselves</strong>",
    ],
    "ReproducibilityNeed_Usability": [
        "UsabilityCheck",
        "The software needs to make it easy <strong>for different users/teams </strong>to get the same reproducible results",
    ],
    "ReproducibilityNeed_Portability": [
        "PortabilityCheck",
        "The software needs to provide reproducible results <strong>on many different platforms</strong>",
    ],
    "ReproducibilityNeed_Maintainability": [
        "MaintainabilityCheck",
        "Reproducibility issues in the software should be <strong>easy to diagnose and fix</strong>",
    ],
    "ReproducibilityNeed_Reliability": [
        "ReliabilityCheck",
        "The software needs to give reproducible results <strong>even in the presence of errors and hardware failures</strong>",
    ],
    "ReproducibilityNeed_FunctionalSuitability": [
        "FunctionalityCheck",
        "The software needs to give reproducible results&nbsp;<strong>across many different use cases</strong>",
    ],
    "ReproducibilityNeed_PerformanceEfficiency": [
        "PerformanceCheck",
        "The software needs to give reproducible results <strong>while running efficiently at scale</strong>",
    ],
    "ReproducibilityNeed_Compatibility": [
        "CompatibilityCheck",
        "The software needs to give reproducible results<strong> even as other software around it changes</strong> (like dependencies)",
    ],
    "ReproducibilityNeed_Security": [
        "SecurityCheck",
        "The software needs to give reproducible results that are <strong>tamper-proof and protected from malicious activity.</strong>",
    ],
    # The software engineers want to better understand what your quality goals are with respect to reproducibility.  Below, you are presented with sets of three software quality aspects.
    # Imagine the software engineering team had to focus on improving one of these three aspects of software quality on your project.  For each set, pick which quality aspects you believe are most useful and least useful for the
    # software engineering team to focus on to improve the reproducibility of your software.
    "MaxDiff_SetQuestionNumber": ["_NumberOfSet"],
    "MaxDiff_AttributeChoice1": ["_Attribute_1"],
    "MaxDiff_AttributeChoice2": ["_Attribute_2"],
    "MaxDiff_AttributeChoice3": ["_Attribute_3"],
    "MaxDiff_BestChoice": ["_BEST"],
    "MaxDiff_WorstChoice": ["_WORST"],
}

surveyBranches = {
    # The Sandia-internal branch of the main survey
    "MainClosed": "General_Survey_of_Reproducibility_and_Software_Quality_Priorities_closed",
    # The Sandia-external branch of the main survey
    "MainOpen": "General_Survey_of_Reproducibility_and_Software_Quality_Priorities_open",
    # Distributed to Christian's astronomy colleagues
    "Astro": "_Astro_General_Survey_of_Reproducibility_and_Software_Quality_Priorities",
    # Distributed to the IDEAS mailing list
    "IDEAS": "_IDEAS_General_Survey_of_Reproducibility_and_Software_Quality_Priorities",
    # Distributed to case study partners ahead of main survey release.
    "CaseStudy": "Reproducibility_and_Quality_Survey_Case_Study_Partners",
    # Distributed to the US-RSE (raffle was used).
    "USRSE": "_US_RSE_General_Survey_of_Reproducibility_and_Software_Quality_Priorities",
    # Distributed to attendees of the ACM REP conference (raffle was used).
    "ACMREP": "Reproducibility_and_Quality_Survey_ACM_REP23",
}


def matchColumn(columnName, questionTexts):
    for questionCode in questionTexts:
        for textToMatch in questionTexts[questionCode]:
            if textToMatch in columnName:
                return True, questionCode
    return False, None


def getSurveyBranchCode(fileName):
    global surveyBranches
    for branchCode in surveyBranches:
        if surveyBranches[branchCode] in fileName:
            return branchCode
    print(
        "\t\tWARNING: ",
        fileName,
        "could not be matched to a survey branch, marking as unknown",
    )
    return "UnknownBranch"


def isMaxDiffData(fileName, data):
    if "MaxDiff_SetQuestionNumber" in data.columns:
        if "max-diff" in fileName:
            return True
        else:
            raise RuntimeError(
                "ERROR: ",
                fileName,
                " has max-diff data fields but is NOT named as a max-diff file.",
            )
    else:
        if "max-diff" not in fileName:
            return False
        else:
            raise RuntimeError(
                "ERROR: ",
                fileName,
                " does not have max-diff data fields but IS named as a max-diff file.",
            )


def extractDataFromCSV(fileName):
    global questionTexts
    extractedData = pd.DataFrame()
    data = pd.read_csv(fileName)
    for columnName in data.columns:
        columnMatchesEntry, questionCode = matchColumn(columnName, questionTexts)
        if columnMatchesEntry and questionCode not in extractedData:
            extractedData[questionCode] = data[columnName]
    return extractedData


def processCSVFiles(csvFiles):
    aggregatedRegularSurveyData = pd.DataFrame()
    aggregatedMaxDiffSurveyData = pd.DataFrame()

    for fileName in csvFiles:
        print("Processing", fileName)
        surveyBranchCode = getSurveyBranchCode(fileName)
        extractedData = extractDataFromCSV(fileName)
        extractedData.insert(0, 'SurveyBranch', surveyBranchCode)
        print("\t\tSurvey belongs to", surveyBranchCode, "survey branch")
        isMaxDiffSurveyFile = isMaxDiffData(fileName, extractedData)

        # Minor cleanup: We don't ask people on the main and case study branches of the survey since we already know the answer:
        # they're all Sandians. Here we populate the data with that information.
        if (
            surveyBranchCode in ["MainClosed", "MainOpen", "CaseStudy"]
            and not isMaxDiffSurveyFile
        ):
            extractedData["Institution"] = (
                "National Laboratory (or other government-affiliated institution)"
            )

        if isMaxDiffSurveyFile:
            if aggregatedMaxDiffSurveyData.empty:
                aggregatedMaxDiffSurveyData = extractedData
            else:
                aggregatedMaxDiffSurveyData = pd.concat(
                    [aggregatedMaxDiffSurveyData, extractedData]
                )
        else:
            if aggregatedRegularSurveyData.empty:
                aggregatedRegularSurveyData = extractedData
            else:
                aggregatedRegularSurveyData = pd.concat(
                    [aggregatedRegularSurveyData, extractedData]
                )

    for key in software_quality_strings_dict:
        aggregatedMaxDiffSurveyData = aggregatedMaxDiffSurveyData.replace(
            key, software_quality_strings_dict[key]
        )

    return aggregatedRegularSurveyData, aggregatedMaxDiffSurveyData


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process survey data and export it as a unified spreadsheet.'
    )
    parser.add_argument(
        'csv_files',
        metavar='F',
        type=str,
        nargs='+',
        help='the csv files that need to be processed',
    )
    args = parser.parse_args()
    regularSurveyData, maxDiffSurveyData = processCSVFiles(args.csv_files)
    os.makedirs("data", exist_ok=True)
    regularSurveyData.to_csv("data/regularSurveyData.csv", index=False)
    maxDiffSurveyData.to_csv("data/maxDiffSurveyData.csv", index=False)
    print(regularSurveyData)
    print(maxDiffSurveyData)
