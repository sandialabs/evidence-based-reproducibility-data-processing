# pip install -r requirements.txt
cd scripts
python combineSurveyData.py \
  ../raw_data/fake_data_1.csv \
  ../raw_data/fake_data_max-diff_1.csv \
  ../raw_data/fake_data_2.csv \
  ../raw_data/fake_data_max-diff_2.csv
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