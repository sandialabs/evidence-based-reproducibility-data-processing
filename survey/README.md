While the data is not provided here, to replicate the analysis and make way too many plots, you would:

- Download survey data from https://zenodo.org/records/11150653 and place it in a `survey/raw_data` folder
- From this directory, run `. ./performAnalysis.sh` (you may have to add execute permissions with `chmod +x`)
- The outputs will be the combined data in `survey/data`, figures in `survey/figures`, and LaTeX variables for use in the paper in `survey/tex`

Script descriptions:
- `combineSurveyData.py` can take data from `../raw_data` and combine it into one file
- `transformSurveyData.py` converts the categorical data that was just combined intro numerical data
- `analyzeMaxDiffUtilities.py` takes the transformed data and estimates multinomial logit MaxDiff utilities from the survey responses
- `analyzeSurveyData.py` looks for correlations and such in the survey responses
- `generatePlots.py` makes many Likert and box and whisker plots including Figure 2 from the paper
- `prioritiesPlot.py` makes several more plots related to software quality priorities including Figure 3 from the paper
