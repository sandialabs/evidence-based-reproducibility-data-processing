"""
Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
"""

# script to collect everyone's codes into a DataFrame with context
from lxml import etree
import zipfile
import os
import pandas as pd

# import argparse

ooXMLns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}


def get_document_comments(docxFileName: str) -> set[dict]:
    """
    Function to fetch all comments with their referenced text
    from deep in
    https://stackoverflow.com/questions/47390928/extract-docx-comments

    Args:
        docxFileName (str): .docx file name and location to get comments from

    Returns:
        dict: comment id as keys and context string as values
        dict: comment id as keys and comment string as values
    """
    comments_dict = {}
    contexts_dict = {}
    docx_zip = zipfile.ZipFile(docxFileName)
    comments_xml = docx_zip.read('word/comments.xml')
    contexts_xml = docx_zip.read('word/document.xml')
    et_comments = etree.XML(comments_xml)
    et_contexts = etree.XML(contexts_xml)
    comments = et_comments.xpath('//w:comment', namespaces=ooXMLns)
    contexts = et_contexts.xpath('//w:commentRangeStart', namespaces=ooXMLns)
    # getting comments
    for c in comments:
        comment = c.xpath('string(.)', namespaces=ooXMLns)
        comment_id = c.xpath('@w:id', namespaces=ooXMLns)[0]
        comments_dict[comment_id] = comment
    # getting contexts
    for c in contexts:
        contexts_id = c.xpath('@w:id', namespaces=ooXMLns)[0]
        parts = et_contexts.xpath(
            "//w:r[preceding-sibling::w:commentRangeStart[@w:id="
            + contexts_id
            + "] and following-sibling::w:commentRangeEnd[@w:id="
            + contexts_id
            + "]]",
            namespaces=ooXMLns,
        )
        comment_of = ''
        for part in parts:
            comment_of += part.xpath('string(.)', namespaces=ooXMLns)
            contexts_dict[contexts_id] = comment_of
    return contexts_dict, comments_dict


code_corrections = {
    "why practice is useful in general": "why practice is useful generally",
    "why practice is generally useful": "why practice is useful generally",
    "expectation of users": "expectations of users",
    "user expectations": "expectations of users",
    "tool use": "tools",
    "why practice is useful": "why practice is useful generally",
    "why practice helps": "why practice is useful generally",
    "team background": "team member background",
    "practice level": "practice levels",
    "quality metric": "quality metrics",
    "barriers and mitigations": "barriers and mitigation",
    "use practice": "useful practice",
}

practice_corrections = {
    "ci": "automated testing",
    "dr": "development pipeline",
    "git/github": "version control",
    "d, at": "automated testing",
}

if __name__ == "__main__":
    coded_files = os.listdir("./Collected/")
    coded_files = [
        file
        for file in coded_files
        if (file.endswith(".docx") and not file.startswith("~"))
    ]

    teams = sorted(set([file.split("_")[0] for file in coded_files]))
    teams_index = {teams[i]: i for i in range(len(teams))}

    codebook = pd.read_excel('codebook.xlsx')
    codes = list(codebook["Unnamed: 1"][2:26])

    practices = list(codebook["Codebook"][-8:])
    # practice_initials = list(codebook["Unnamed: 1"][-8:])
    practice_initials = ['T', 'AT', 'CR', 'DP', 'D', 'PP', 'TC', 'VC']
    practice_initials_to_practice = {
        practice_initials[i].lower(): practices[i].lower()
        for i in range(len(practice_initials))
    }
    practice_initials_to_practice['doc'] = "documentation"
    practice_initials_to_practice['vcs'] = "version control"

    df = pd.DataFrame()
    df["Team"] = teams
    for code in codes:
        df[code.lower()] = ["" for i in range(len(teams))]
    for practice in practices:
        df[practice.lower()] = ["" for i in range(len(teams))]
    df["Other code"] = ["" for i in range(len(teams))]
    df["Other practice"] = ["" for i in range(len(teams))]

    # filling out each row of DataFrame
    for team in teams:
        team_coded_files = [file for file in coded_files if file.startswith(team)]
        for team_coded_file in team_coded_files:
            contexts, comments = get_document_comments("./Collected/" + team_coded_file)
            for i in range(len(comments)):
                key = str(i)
                comment = comments[str(i)]
                split_comment = comment.split(":")
                comment_without_code = split_comment[-1]
                code = (split_comment[0]).lower()
                split_code_parenthesis = code.split(" (")
                split_code_bracket = code.split(" [")
                if len(split_code_bracket) > len(split_code_parenthesis):
                    split_code = split_code_bracket
                else:
                    split_code = split_code_parenthesis
                code_wo_practice = split_code[0]

                # add comment to code columns
                if code_wo_practice in code_corrections.keys():
                    code_wo_practice = code_corrections[code_wo_practice]
                context = contexts[str(i)] if key in contexts.keys() else ""
                if code_wo_practice in df.columns:
                    if df[code_wo_practice][teams_index[team]] == "":
                        string_before = ""
                    else:
                        string_before = df[code_wo_practice][teams_index[team]] + "\n\n"
                    df[code_wo_practice][teams_index[team]] = (
                        string_before
                        + "Comment: "
                        + comment_without_code
                        + "\nContext: "
                        + context
                    )
                else:
                    if df["Other code"][teams_index[team]] == "":
                        string_before = ""
                    else:
                        string_before = df["Other code"][teams_index[team]] + "\n\n"
                    df["Other code"][teams_index[team]] = (
                        string_before + comment + "\nContext: " + context
                    )

                # add comment to practice columns
                if len(split_code) > 1:
                    practice = split_code[1][:-1]
                    if practice in practice_initials_to_practice.keys():
                        practice = practice_initials_to_practice[practice]
                    elif practice in practice_corrections.keys():
                        practice = practice_corrections[practice]

                    if practice in df.columns:
                        if df[practice][teams_index[team]] == "":
                            string_before = ""
                        else:
                            string_before = df[practice][teams_index[team]] + "\n\n"
                        df[practice][teams_index[team]] = (
                            string_before
                            + team_coded_file
                            + " Comment: "
                            + comment
                            + "\nContext: "
                            + context
                        )
                    else:
                        if df["Other practice"][teams_index[team]] == "":
                            string_before = ""
                        else:
                            string_before = (
                                df["Other practice"][teams_index[team]] + "\n\n"
                            )
                        df["Other practice"][teams_index[team]] = (
                            string_before + comment + "\nContext: " + context
                        )

    df.rename(columns={code.lower(): code for code in codes}, inplace=True)
    df.rename(
        columns={practice.lower(): practice for practice in practices}, inplace=True
    )
    df.to_csv("./Collected/collectedCodes.csv", index=False)
