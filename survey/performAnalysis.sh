pip install -r requirements.txt
cd scripts
python combineSurveyData.py \
  ../raw_data/General_Survey_of_Reproducibility_and_Software_Quality_Priorities_closed-max-diff-responses-2023-08-19.csv \
  ../raw_data/General_Survey_of_Reproducibility_and_Software_Quality_Priorities_closed-responses-2023-08-19.csv \
  ../raw_data/General_Survey_of_Reproducibility_and_Software_Quality_Priorities_open-max-diff-responses-2023-08-19.csv \
  ../raw_data/General_Survey_of_Reproducibility_and_Software_Quality_Priorities_open-responses-2023-08-19.csv \
  ../raw_data/Reproducibility_and_Quality_Survey_ACM_REP23_-max-diff-responses-2023-08-19.csv \
  ../raw_data/Reproducibility_and_Quality_Survey_ACM_REP23_-responses-2023-08-19.csv \
  ../raw_data/Reproducibility_and_Quality_Survey_Case_Study_Partners_-max-diff-responses-2023-08-19.csv \
  ../raw_data/Reproducibility_and_Quality_Survey_Case_Study_Partners_-responses-2023-08-19.csv \
  ../raw_data/_Astro_General_Survey_of_Reproducibility_and_Software_Quality_Priorities-max-diff-responses-2023-08-19.csv \
  ../raw_data/_Astro_General_Survey_of_Reproducibility_and_Software_Quality_Priorities-responses-2023-08-19.csv \
  ../raw_data/_IDEAS_General_Survey_of_Reproducibility_and_Software_Quality_Priorities-max-diff-responses-2023-08-19.csv \
  ../raw_data/_IDEAS_General_Survey_of_Reproducibility_and_Software_Quality_Priorities-responses-2023-08-19.csv \
  ../raw_data/_US_RSE_General_Survey_of_Reproducibility_and_Software_Quality_Priorities-max-diff-responses-2023-08-19.csv \
  ../raw_data/_US_RSE_General_Survey_of_Reproducibility_and_Software_Quality_Priorities-responses-2023-08-19.csv
python transformSurveyData.py \
  data/regularSurveyData.csv \
  data/maxDiffSurveyData.csv
python analyzeMaxDiffUtilities.py \
  data/regularSurveyData_transformed.csv \
  data/maxDiffSurveyData_transformed.csv
python analyzeSurveyData.py \
  data/regularSurveyData_transformed.csv \
  data/maxDiffSurveyData_transformed.csv \
  data/software_quality_utilities.csv \
  data/software_quality_utilities_per_person.csv
python generatePlots.py \
  data/regularSurveyData.csv \
  data/software_quality_utilities.csv \
  data/software_quality_utilities_per_person.csv \
  data/software_quality_prioritized_utilities.csv
python prioritiesPlot.py \
  data/regularSurveyData_transformed.csv
cd ..