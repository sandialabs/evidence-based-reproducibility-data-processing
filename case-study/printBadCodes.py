"""
Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
"""

# script to print where codes are not formatted correctly
# from docx import Document  # install with pip install python-docx
from lxml import etree
import zipfile
import os
import argparse

ooXMLns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}


def get_document_comments(docxFileName: str) -> dict:
    """Function to extract all the comments from a .docx

    Args:
        docxFileName (str): .docx file name and location to get comments from

    Returns:
        dict: a dictionary with comment id as key and comment string as value
    """
    comments_dict = {}
    docxZip = zipfile.ZipFile(docxFileName)
    commentsXML = docxZip.read('word/comments.xml')
    et = etree.XML(commentsXML)
    comments = et.xpath('//w:comment', namespaces=ooXMLns)
    for c in comments:
        comment = c.xpath('string(.)', namespaces=ooXMLns)
        comment_id = c.xpath('@w:id', namespaces=ooXMLns)[0]
        comments_dict[comment_id] = comment
    return comments_dict


if __name__ == "__main__":
    # parsing ARGS
    parser_desc = 'Print code comments that seem to have the wrong formatting'
    parser = argparse.ArgumentParser(description=parser_desc)
    parser.add_argument(
        'coder', metavar='coder', type=str, nargs=1, help='The coder folder name'
    )
    args = parser.parse_args()

    # reading in .csv's and preprocessing
    coderName = args.coder[0]

    codes = [
        "Team member background",
        "Team size",
        "Team dynamics",
        "Institutional context",
        "Project description",
        "Project over time",
        "Stakeholders and users",
        "Expectations of users",
        "Software quality understanding",
        "Software quality priorities",
        "Reproducibility understanding",
        "Reproducibility priorities",
        "Barriers and mitigation",
        "Quality seeking",
        "Quality metrics",
        "Useful practice",
        "Why practice is useful for reproducibility",
        "Why practice is useful generally",
        "Practice advice",
        "Practice levels",
        "Practice learning",
        "Tools",
        "Location of tool",
        "Time of integration",
    ]
    codes = [code.lower() for code in codes]
    useful_practices = [
        "Testing",
        "Automated testing",
        "Code Reviews",
        "Development Pipeline",
        "Documentation",
        "Public Processes",
        "Team Communication",
        "Version Control",
        "T",
        "AT",
        "CR",
        "DP",
        "D",
        "DOC",
        "PP",
        "TC",
        "VC",
        "VCS",
    ]
    useful_practices = [practice.lower() for practice in useful_practices]

    def print_suggestion_str(string: str):
        print("consider changing code to " + string + "\n")

    code_suggestions = {
        "room for improvement": "Quality seeking",
        "reproducibility goals": "Reproducibility priorities",
        "reproducibility understanding and opinions": "Reproducibility understanding",
        "tool": "Tools",
        "organizational context": "Institutional context",
        "expectations of user": "Expectations of users",
        "quality metric": "Quality metrics",
        "user expectations": "Expectations of users",
        "stakeholder and users": "Stakeholders and users",
        "practice level": "Practice levels",
        "why practice is useful in general": "Why practice is useful generally",
    }

    for fileName in os.listdir("./" + coderName + "/"):
        if fileName.endswith(".docx") and not fileName.startswith("~"):
            print("Processing " + fileName + "...\n")

            teamName = fileName.split("_")[0]
            comments = get_document_comments("./" + coderName + "/" + fileName)

            for cID in comments:
                comment = comments[cID]

                # code with practice tag (if it has one)
                code = (comment.split(":")[0]).lower()
                split_code_parenthesis = code.split(" (")
                split_code_bracket = code.split(" [")
                if len(split_code_bracket) > len(split_code_parenthesis):
                    split_code = split_code_bracket
                else:
                    split_code = split_code_parenthesis
                code_without_practice_tag = split_code[0]

                # if code has a practice tag, and tag isn't valid, print code
                if (len(split_code) > 1) and (
                    split_code[1][:-1] not in useful_practices
                ):
                    print(coderName + ", " + teamName + ", " + comment)
                    print("Practice not found in useful practice list\n")

                # if no practice tag issues, but code itself isn't
                # valid, print code
                elif code_without_practice_tag not in codes:
                    print(coderName + ", " + teamName + ", " + comment)
                    if code_without_practice_tag in code_suggestions.keys():
                        print_suggestion_str(
                            code_suggestions[code_without_practice_tag]
                        )
                    else:
                        print("No suggestion found\n")
