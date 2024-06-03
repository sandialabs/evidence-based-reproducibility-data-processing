While the data is not provided here, to replicate the analysis and make way too many plots, you would:

From this directory, run `. ./performAnalysis.sh`

- `combineSurveyData.py` can take data from `../raw_data` and combine it into one file
- `transformSurveyData.py` converts the categorical data that was just combined intro numerical data
- `analyzeMaxDiffUtilities.py` takes the transformed data and estimates multinomial logit MaxDiff utilities from the survey responses
- `analyzeSurveyData.py` looks for correlations and such in the survey responses
- `generatePlots.py` makes many Likert and box and whisker plots including Figure 2
- `prioritiesPlot.py` makes several more plots related to software quality priorities including Figure 3
