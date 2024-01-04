SYSTEMPROMPT = 'You are an experienced doctor in the field of COVID-19 treatment.'
USERPROMPT = '''I will provide you with medical information from multiple Intensive Care Unit (ICU) visits of a patient, each characterized by a fixed number of features.

{INPUT_FORMAT_DESCRIPTION}

{TASK_DESCRIPTION_AND_RESPONSE_FORMAT}

In situations where the data does not allow for a reasonable conclusion, respond with the phrase `I do not know` without any additional explanation.

{UNIT_RANGE_CONTEXT}

{EXAMPLE}

Input information of a patient:

The patient is a {SEX}, aged {AGE} years.
The patient had {LENGTH} visits that occurred at {RECORD_TIME_LIST}.
Details of the features for each visit are as follows:

{DETAIL}

{RESPONSE_FORMAT}
RESPONSE:
'''

UNIT = {
    'tjh': 'prompts/tjh/unit.json',
    'mimic-iv': 'prompts/mimic-iv/unit.json',
}

REFERENCE_RANGE = {
    'tjh': 'prompts/tjh/range.json',
    'mimic-iv': 'prompts/mimic-iv/range.json',
}

INPUT_FORMAT_DESCRIPTION = {
    'string': 'Present multiple visit data of a patient in one batch. Represent each feature within this data as a string of values, separated by commas.',
    'list': 'Display multiple visit data of a patient in one batch, expressing each feature as a list of values, separated by commas.',
    'batches': 'Organize visit data of a patient into separate batches, each batch corresponding to one visit.',
}

TASK_DESCRIPTION_AND_RESPONSE_FORMAT = {
    'outcome': 'Your task is to assess the provided medical data and analyze the health records from ICU visits to determine the likelihood of the patient not surviving their hospital stay. Please respond with only a floating-point number between 0 and 1, where a higher number suggests a greater likelihood of death.',
    'los': 'Your task is to Evaluate the provided medical data to estimate the remaining duration of the ICU stay. Consider the progression of health across multiple visits to forecast the length of intensive care needed. Please respond with only an integer indicating the number of days expected in the ICU.',
    'readmission': 'Your task is to analyze the medical history to predict the probability of readmission within 30 days post-discharge. Include cases where a patient passes away within 30 days from the discharge date. Please respond with only a floating-point number between 0 and 1, where a higher number suggests a greater likelihood of readmission.',
}

RESPONSE_FORMAT = {
    'outcome': 'Please respond with only a floating-point number between 0 and 1, where a higher number suggests a greater likelihood of death. Do not include any additional explanation.',
    'los': 'Please respond with only an integer indicating the number of days expected in the ICU. Do not include any additional explanation.',
    'readmission': 'Please respond with only a floating-point number between 0 and 1, where a higher number suggests a greater likelihood of readmission. Do not include any additional explanation.',
}

EXAMPLE = {
    'tjh': {
        'string': [
'''
Input information of a patient:
The patient is a male, aged 52.0 years.
The patient had 5 visits that occurred at 2020-02-09, 2020-02-10, 2020-02-13, 2020-02-14, 2020-02-17.
Details of the features for each visit are as follows:
- Hypersensitive cardiac troponinI: "1.9, 1.9, 1.9, 1.9, 1.9"
- hemoglobin: "139.0, 139.0, 142.0, 142.0, 142.0"
- Serum chloride: "103.7, 103.7, 104.2, 104.2, 104.2"
- Prothrombin time: "15.1, 15.1, 14.3, 14.3, 14.3"
- procalcitonin: "0.04, 0.04, 0.04, 0.04, 0.04"
- eosinophils(%): "0.3, 0.3, 2.1, 2.1, 2.1"
- Interleukin 2 receptor: "1186.0, 1186.0, 1186.0, 1186.0, 1186.0"
- Alkaline phosphatase: "47.0, 47.0, 48.0, 48.0, 48.0"
- albumin: "34.8, 34.8, 38.7, 38.7, 38.7"
- basophil(%): "0.0, 0.0, 0.0, 0.0, 0.0"
- Interleukin 10: "10.4, 10.4, 10.4, 10.4, 10.4"
- Total bilirubin: "4.7, 4.7, 9.4, 9.4, 9.4"
- Platelet count: "171.0, 171.0, 217.0, 217.0, 217.0"
- monocytes(%): "8.1, 8.1, 10.7, 10.7, 10.7"
- antithrombin: "88.0, 88.0, 88.0, 88.0, 88.0"
- Interleukin 8: "48.9, 48.9, 48.9, 48.9, 48.9"
- indirect bilirubin: "2.8, 2.8, 6.0, 6.0, 6.0"
- Red blood cell distribution width : "11.7, 11.7, 11.3, 11.3, 11.3"
- neutrophils(%): "61.8, 61.8, 51.5, 51.5, 51.5"
- total protein: "66.5, 66.5, 72.2, 72.2, 72.2"
- Quantification of Treponema pallidum antibodies: "0.05, 0.05, 0.05, 0.05, 0.05"
- Prothrombin activity: "78.0, 78.0, 84.0, 84.0, 84.0"
- HBsAg: "0.0, 0.0, 0.0, 0.0, 0.0"
- mean corpuscular volume: "91.5, 91.5, 88.6, 88.6, 88.6"
- hematocrit: "41.1, 41.1, 41.2, 41.2, 41.2"
- White blood cell count: "3.56, 3.56, 3.36, 3.36, 3.36"
- Tumor necrosis factorα: "9.1, 9.1, 9.1, 9.1, 9.1"
- mean corpuscular hemoglobin concentration: "338.0, 338.0, 345.0, 345.0, 345.0"
- fibrinogen: "2.82, 2.82, 5.09, 5.09, 5.09"
- Interleukin 1β: "5.0, 5.0, 5.0, 5.0, 5.0"
- Urea: "4.0, 4.0, 4.3, 4.3, 4.3"
- lymphocyte count: "1.06, 1.06, 1.2, 1.2, 1.2"
- PH value: "6.988, 6.5, 6.5, 6.5, 6.5"
- Red blood cell count: "4.49, 4.49, 4.65, 4.65, 4.65"
- Eosinophil count: "0.01, 0.01, 0.07, 0.07, 0.07"
- Corrected calcium: "2.25, 2.25, 2.4, 2.4, 2.4"
- Serum potassium: "4.31, 4.31, 4.51, 4.51, 4.51"
- glucose: "5.31, 5.31, 5.32, 5.32, 5.32"
- neutrophils count: "2.2, 2.2, 1.73, 1.73, 1.73"
- Direct bilirubin: "1.9, 1.9, 3.4, 3.4, 3.4"
- Mean platelet volume: "11.2, 11.2, 11.0, 11.0, 11.0"
- ferritin: "544.65, 544.65, 544.65, 544.65, 544.65"
- RBC distribution width SD: "39.5, 39.5, 36.5, 36.5, 36.5"
- Thrombin time: "18.4, 18.4, 15.7, 15.7, 15.7"
- (%)lymphocyte: "29.8, 29.8, 35.7, 35.7, 35.7"
- HCV antibody quantification: "0.05, 0.05, 0.05, 0.05, 0.05"
- D-D dimer: "5.51, 5.51, 1.26, 1.26, 1.26"
- Total cholesterol: "3.61, 3.61, 4.3, 4.3, 4.3"
- aspartate aminotransferase: "28.0, 28.0, 19.0, 19.0, 19.0"
- Uric acid: "238.0, 238.0, 235.0, 235.0, 235.0"
- HCO3-: "25.8, 25.8, 26.9, 26.9, 26.9"
- calcium: "2.01, 2.01, 2.23, 2.23, 2.23"
- Amino-terminal brain natriuretic peptide precursor(NT-proBNP): "27.0, 27.0, 81.0, 81.0, 81.0"
- Lactate dehydrogenase: "211.0, 211.0, 177.0, 177.0, 177.0"
- platelet large cell ratio : "33.6, 33.6, 32.7, 32.7, 32.7"
- Interleukin 6: "37.3, 37.3, 37.3, 37.3, 37.3"
- Fibrin degradation products: "25.8, 25.8, 25.8, 25.8, 25.8"
- monocytes count: "0.29, 0.29, 0.36, 0.36, 0.36"
- PLT distribution width: "13.9, 13.9, 12.7, 12.7, 12.7"
- globulin: "31.7, 31.7, 33.5, 33.5, 33.5"
- γ-glutamyl transpeptidase: "14.0, 14.0, 12.0, 12.0, 12.0"
- International standard ratio: "1.17, 1.17, 1.11, 1.11, 1.11"
- basophil count(#): "0.0, 0.0, 0.0, 0.0, 0.0"
- mean corpuscular hemoglobin : "31.0, 31.0, 30.5, 30.5, 30.5"
- Activation of partial thromboplastin time: "44.9, 44.9, 38.5, 38.5, 38.5"
- High sensitivity C-reactive protein: "7.4, 7.4, 2.5, 2.5, 2.5"
- HIV antibody quantification: "0.08, 0.08, 0.08, 0.08, 0.08"
- serum sodium: "140.3, 140.3, 143.0, 143.0, 143.0"
- thrombocytocrit: "0.19, 0.19, 0.24, 0.24, 0.24"
- ESR: "12.0, 11.0, 11.0, 11.0, 11.0"
- glutamic-pyruvic transaminase: "19.0, 19.0, 16.0, 16.0, 16.0"
- eGFR: "93.7, 93.7, 100.6, 100.6, 100.6"
- creatinine: "70.0, 70.0, 66.0, 66.0, 66.0"

RESPONSE:
0.25
''',
'''
Input information of a patient:
The patient is a female, aged 71.0 years.
The patient had 5 visits that occurred at 2020-02-01, 2020-02-02, 2020-02-09, 2020-02-10, 2020-02-11.
Details of the features for each visit are as follows:
- Hypersensitive cardiac troponinI: "5691.05, 11970.22, 9029.88, 6371.5, 3638.55"
- hemoglobin: "105.68, 132.84, 54.19, 136.33, 123.69"
- Serum chloride: "89.18, 101.54, 90.35, 103.99, 102.06"
- Prothrombin time: "26.55, 8.79, 30.51, 31.2, 3.43"
- procalcitonin: "7.43, 15.53, 14.79, 2.51, 4.25"
- eosinophils(%): "0.14, 0.23, 0.04, 0.3, 0.81"
- Interleukin 2 receptor: "618.04, 1318.73, 1478.01, 811.21, 3549.11"
- Alkaline phosphatase: "86.68, 179.2, 73.76, 126.39, 67.84"
- albumin: "33.86, 20.69, 28.47, 32.61, 25.5"
- basophil(%): "0.3, 0.13, 0.15, 0.09, 0.15"
- Interleukin 10: "129.13, 58.81, 100.0, 105.21, 213.71"
- Total bilirubin: "9.96, 33.78, 49.0, 35.95, 17.48"
- Platelet count: "212.96, 28.47, 84.25, 97.13, 159.37"
- monocytes(%): "2.68, 4.17, 0.68, 2.9, 2.09"
- antithrombin: "77.22, 54.11, 73.52, 86.99, 56.75"
- Interleukin 8: "836.26, 57.22, 1264.56, 820.74, 1022.37"
- indirect bilirubin: "26.18, 3.31, 3.05, 25.81, 12.74"
- Red blood cell distribution width : "14.35, 15.07, 12.28, 14.39, 15.86"
- neutrophils(%): "91.57, 92.14, 89.08, 86.1, 84.59"
- total protein: "60.65, 64.92, 73.03, 35.71, 63.22"
- Quantification of Treponema pallidum antibodies: "0.7, 1.47, 0.12, 0.03, 0.65"
- Prothrombin activity: "91.38, 51.81, 56.8, 98.43, 80.05"
- HBsAg: "49.36, 13.65, 55.77, 19.29, 57.02"
- mean corpuscular volume: "84.66, 79.87, 89.2, 92.85, 102.65"
- hematocrit: "40.84, 36.8, 40.38, 40.23, 33.84"
- White blood cell count: "14.88, 38.38, 0.68, 29.04, 13.94"
- Tumor necrosis factorα: "10.55, 2.13, 44.09, 14.21, 2.08"
- mean corpuscular hemoglobin concentration: "342.83, 335.18, 369.26, 352.65, 351.4"
- fibrinogen: "1.06, 5.08, 2.94, 3.89, 2.51"
- Interleukin 1β: "11.97, 18.62, 1.62, 12.6, 20.59"
- Urea: "9.93, 9.63, 28.11, 10.4, 32.32"
- lymphocyte count: "0.52, 0.47, 0.21, 0.25, 0.44"
- PH value: "6.65, 5.37, 7.56, 6.74, 5.81"
- Red blood cell count: "51.69, 3.09, 6.05, 52.1, 7.15"
- Eosinophil count: "0.06, 0.01, 0.04, 0.1, 0.05"
- Corrected calcium: "2.33, 2.56, 2.14, 2.3, 2.44"
- Serum potassium: "5.79, 4.53, 5.32, 5.72, 4.21"
- glucose: "11.69, 7.12, 8.49, 19.74, 1.87"
- neutrophils count: "20.91, 0.84, 2.39, 7.16, 19.42"
- Direct bilirubin: "38.18, 40.5, 22.31, 37.53, 26.16"
- Mean platelet volume: "12.01, 13.19, 11.31, 11.15, 11.55"
- ferritin: "480.03, 14115.76, 1040.85, 6632.09, 284.75"
- RBC distribution width SD: "45.3, 45.88, 38.96, 42.36, 49.51"
- Thrombin time: "7.21, 35.88, 25.02, 26.95, 12.38"
- (%)lymphocyte: "10.81, 8.12, 12.17, 6.8, 4.64"
- HCV antibody quantification: "0.41, 0.27, 0.02, 0.76, 0.17"
- D-D dimer: "9.83, 10.69, 18.68, 13.41, 7.22"
- Total cholesterol: "3.62, 2.07, 3.56, 1.66, 3.99"
- aspartate aminotransferase: "278.71, 210.03, 118.04, 83.08, 6.35"
- Uric acid: "225.95, 829.72, 230.05, 398.57, 424.12"
- HCO3-: "17.27, 32.55, 20.97, 29.9, 14.69"
- calcium: "1.84, 2.17, 2.18, 2.18, 2.18"
- Amino-terminal brain natriuretic peptide precursor(NT-proBNP): "8726.72, 27431.32, 1018.34, 3227.64, 19588.69"
- Lactate dehydrogenase: "1208.28, 803.67, 530.14, 1020.15, 1254.85"
- platelet large cell ratio : "33.69, 42.57, 32.57, 31.01, 36.11"
- Interleukin 6: "320.19, 1584.66, 2081.52, 357.65, 293.17"
- Fibrin degradation products: "5.92, 46.25, 22.25, 41.3, 148.9"
- monocytes count: "0.38, 0.58, 0.19, 0.24, 0.74"
- PLT distribution width: "18.07, 13.01, 14.67, 16.97, 14.53"
- globulin: "37.65, 34.26, 33.69, 29.35, 43.63"
- γ-glutamyl transpeptidase: "46.9, 95.52, 136.81, 67.8, 185.34"
- International standard ratio: "3.02, 2.56, 0.96, 0.43, 1.2"
- basophil count(#): "0.02, 0.01, 0.0, 0.02, 0.07"
- mean corpuscular hemoglobin : "28.19, 30.54, 30.27, 28.1, 34.6"
- Activation of partial thromboplastin time: "46.66, 48.13, 54.7, 46.62, 45.08"
- Hypersensitive c-reactive protein: "205.0, 308.2, 269.74, 80.29, 159.29"
- HIV antibody quantification: "0.08, 0.08, 0.08, 0.13, 0.18"
- serum sodium: "158.03, 158.17, 144.86, 154.06, 157.21"
- thrombocytocrit: "0.08, 0.29, 0.33, 0.09, 0.18"
- ESR: "91.04, 43.32, 11.18, 22.03, 18.92"
- glutamic-pyruvic transaminase: "91.66, 119.96, 24.23, 4.08, 180.32"
- eGFR: "62.3, 80.36, 40.23, 101.32, 31.25"
- creatinine: "158.44, 294.55, 68.94, 267.4, 299.59"

RESPONSE:
0.85
''',
'''
Input information of a patient:
The patient is a female, aged 53.0 years.
The patient had 5 visits that occurred at 2020-01-20, 2020-01-22, 2020-01-27, 2020-01-28, 2020-01-29.
Details of the features for each visit are as follows:
- Hypersensitive cardiac troponinI: "14.98, 51.49, 49.99, 23.52, 67.93"
- hemoglobin: "140.19, 122.73, 116.95, 114.34, 161.72"
- Serum chloride: "101.94, 98.23, 92.9, 94.47, 99.78"
- Prothrombin time: "12.04, 13.68, 14.26, 13.28, 14.48"
- procalcitonin: "0.19, 0.51, 0.04, 0.41, 0.37"
- eosinophils(%): "1.11, 0.39, 2.09, 2.8, 1.47"
- Interleukin 2 receptor: "1670.7, 1265.5, 969.0, 86.58, 55.46"
- Alkaline phosphatase: "6.3, 90.66, 60.96, 25.45, 109.85"
- albumin: "31.42, 31.86, 30.87, 28.4, 34.44"
- basophil(%): "0.18, 0.33, 0.03, 0.15, 0.25"
- Interleukin 10: "9.16, 7.55, 1.41, 9.9, 13.53"
- Total bilirubin: "2.29, 23.49, 8.67, 6.21, 13.77"
- Platelet count: "200.56, 270.21, 38.99, 150.03, 261.33"
- monocytes(%): "8.58, 8.96, 6.96, 5.02, 11.16"
- antithrombin: "112.69, 102.94, 107.6, 78.38, 76.52"
- Interleukin 8: "0.71, 44.16, 77.52, 107.47, 66.81"
- indirect bilirubin: "7.5, 15.29, 5.73, 4.58, 12.17"
- Red blood cell distribution width : "14.17, 13.68, 11.76, 11.86, 13.8"
- neutrophils(%): "48.5, 62.99, 47.34, 59.18, 79.27"
- total protein: "59.81, 63.27, 74.44, 61.2, 61.39"
- Quantification of Treponema pallidum antibodies: "0.02, 0.09, 0.03, 0.2, 0.52"
- Prothrombin activity: "96.8, 111.24, 95.2, 93.28, 94.53"
- HBsAg: "44.61, 5.76, 7.3, 2.07, 14.56"
- mean corpuscular volume: "96.07, 87.21, 87.43, 83.14, 95.02"
- hematocrit: "38.84, 32.69, 44.68, 35.16, 37.75"
- White blood cell count: "32.68, 105.46, 87.52, 70.87, 102.46"
- Tumor necrosis factorα: "11.54, 12.14, 6.04, 9.58, 10.14"
- mean corpuscular hemoglobin concentration: "349.45, 332.32, 338.93, 351.71, 339.8"
- fibrinogen: "3.22, 4.96, 3.05, 3.5, 2.65"
- Interleukin 1β: "13.22, 0.49, 2.79, 8.04, 2.09"
- Urea: "10.14, 1.72, 3.08, 1.7, 5.01"
- lymphocyte count: "2.84, 2.21, 1.87, 3.96, 4.41"
- PH value: "7.0, 6.46, 5.3, 6.75, 6.47"
- Red blood cell count: "19.75, 20.46, 9.62, 22.39, 1.97"
- Eosinophil count: "0.06, 0.05, 0.11, 0.05, 0.02"
- Corrected calcium: "2.38, 2.5, 2.41, 2.33, 2.28"
- Serum potassium: "4.9, 4.79, 4.85, 3.83, 4.82"
- glucose: "14.83, 9.22, 7.96, 3.27, 11.66"
- neutrophils count: "4.56, 4.77, 3.8, 5.98, 9.17"
- Direct bilirubin: "4.7, 7.02, 2.85, 2.89, 3.46"
- Mean platelet volume: "10.62, 9.93, 10.27, 8.54, 9.34"
- ferritin: "776.45, 779.11, 788.39, 1374.98, 301.24"
- RBC distribution width SD: "45.57, 45.06, 44.27, 37.27, 39.07"
- Thrombin time: "16.69, 16.09, 17.34, 16.13, 13.91"
- (%)lymphocyte: "17.39, 39.95, 32.77, 24.94, 7.7"
- HCV antibody quantification: "0.15, 0.07, 0.09, 0.1, 0.04"
- D-D dimer: "2.35, 0.47, 2.43, 4.26, 4.03"
- Total cholesterol: "4.37, 4.0, 3.9, 3.76, 2.93"
- aspartate aminotransferase: "42.47, 35.18, 31.45, 46.28, 18.59"
- Uric acid: "320.14, 173.54, 142.63, 167.85, 396.04"
- HCO3-: "18.17, 24.48, 25.31, 24.28, 24.35"
- calcium: "2.35, 2.21, 2.1, 2.38, 2.45"
- Amino-terminal brain natriuretic peptide precursor(NT-proBNP): "5620.21, 15248.37, 5204.64, 4715.06, 1849.24"
- Lactate dehydrogenase: "138.87, 288.26, 274.65, 395.67, 325.58"
- platelet large cell ratio : "29.13, 25.94, 26.13, 16.1, 32.85"
- Interleukin 6: "26.29, 15.53, 31.44, 73.75, 89.24"
- Fibrin degradation products: "14.88, 14.53, 4.06, 25.76, 13.67"
- monocytes count: "0.03, 0.9, 3.34, 1.21, 0.19"
- PLT distribution width: "13.49, 10.19, 15.36, 10.53, 12.27"
- globulin: "30.58, 16.22, 34.84, 28.97, 27.27"
- γ-glutamyl transpeptidase: "26.96, 112.74, 99.65, 67.11, 80.88"
- International standard ratio: "1.06, 1.07, 1.04, 1.0, 1.09"
- basophil count(#): "0.02, 0.02, 0.04, 0.02, 0.03"
- mean corpuscular hemoglobin : "31.7, 30.06, 30.23, 28.67, 30.19"
- Activation of partial thromboplastin time: "34.53, 29.59, 37.83, 37.55, 42.07"
- Hypersensitive c-reactive protein: "1.59, 43.03, 61.64, 3.55, 4.71"
- HIV antibody quantification: "0.07, 0.1, 0.09, 0.15, 0.11"
- serum sodium: "145.1, 135.29, 136.71, 135.45, 139.5"
- thrombocytocrit: "0.22, 0.26, 0.2, 0.32, 0.28"
- ESR: "2.01, 12.41, 5.66, 40.72, 25.14"
- glutamic-pyruvic transaminase: "30.78, 60.07, 62.43, 46.97, 6.61"
- eGFR: "130.59, 114.35, 73.23, 88.75, 54.57"
- creatinine: "75.82, 316.26, 118.16, 51.86, 214.94"

RESPONSE:
0.3
''',
        ]
    },
    'mimic-iv': {
        'string': [
'''
Input information of a patient:
The patient is a female, aged 52 years.
The patient had 9 visits that occurred at 0, 1, 2, 3, 4.
Details of the features for each visit are as follows:
- Capillary refill rate: "unknown, unknown, unknown, unknown"
- Glascow coma scale eye opening: "Spontaneously, Spontaneously, Spontaneously, Spontaneously"
- Glascow coma scale motor response: "Obeys Commands, Obeys Commands, Obeys Commands, Obeys Commands"
- Glascow coma scale total: "unknown, unknown, unknown, unknown"
- Glascow coma scale verbal response: "Confused, Confused, Oriented, Oriented"
- Diastolic blood pressure: "56.0, 56.0, 58.0, 53.0"
- Fraction inspired oxygen: "unknown, unknown, unknown, unknown"
- Glucose: "115.0, 115.0, 115.0, 115.0"
- Heart Rate: "94.0, 105.0, 97.0, 100.0"
- Height: "unknown, unknown, unknown, unknown"
- Mean blood pressure: "64.0, 64.0, 67.0, 60.0"
- Oxygen saturation: "97.0, 94.0, 95.0, 95.0"
- Respiratory rate: "23.0, 21.0, 20.0, 21.0"
- Systolic blood pressure: "88.0, 88.0, 95.0, 86.0"
- Temperature: "37.05555555555556, 37.05555555555556, 37.5, 37.5"
- Weight: "39.3264264, 39.3264264, 39.3264264, 39.3264264"
- pH: "unknown, unknown, unknown, unknown"

RESPONSE:
0.3
''',
'''
Input information of a patient:
The patient is a male, aged 49 years.
The patient had 4 visits that occurred at 0, 1, 2, 3.
Details of the features for each visit are as follows:
- Capillary refill rate: "unknown, unknown, unknown, unknown"
- Glascow coma scale eye opening: "To speech, To speech, To speech, Spontaneously"
- Glascow coma scale motor response: "Abnorm extensn, Obeys Commands, No Response, Localizes Pain"
- Glascow coma scale total: "unknown, unknown, unknown, unknown"
- Glascow coma scale verbal response: "Confused, Confused, Confused, Confused"
- Diastolic blood pressure: "324.96, 181.48, 618.15, 318.7"
- Fraction inspired oxygen: "0.62, 0.65, 0.63, 0.57"
- Glucose: "6587.53, 13421.45, 4827.29, 5383.21"
- Heart Rate: "68.21, 79.45, 78.74, 113.09"
- Height: "156.65, 179.6, 157.08, 162.58"
- Mean blood pressure: "577.77, 1708.11, 1543.6, 488.92"
- Oxygen saturation: "2532.85, 1282.89, 1450.5, 3892.13"
- Respiratory rate: "7.02, 19.59, 22.39, 13.76"
- Systolic blood pressure: "279.77, 81.43, 3515.27, 3862.95"
- Temperature: "23.49, 33.02, 46.53, 28.58"
- Weight: "74.35, 95.29, 92.48, 107.65"
- pH: "4037.65, 12378.75, 7459.89, 9343.3"

RESPONSE:
0.9
''',
'''
Input information of a patient:
The patient is a female, aged 68 years.
The patient had 5 visits that occurred at 0, 1, 2, 3, 4.
Details of the features for each visit are as follows:
- Capillary refill rate: "unknown, unknown, unknown, unknown, unknown"
- Glascow coma scale eye opening: "Spontaneously, Spontaneously, Spontaneously, Spontaneously, Spontaneously"
- Glascow coma scale motor response: "Obeys Commands, Obeys Commands, Obeys Commands, Obeys Commands, Obeys Commands"
- Glascow coma scale total: "unknown, 15, 13, 12, 11"
- Glascow coma scale verbal response: "Spontaneously, Inappropriate Words, Confused, Inappropriate Words, Inappropriate Words"
- Diastolic blood pressure: "250.87, 136.42, 398.25, 20.98, 117.72"
- Fraction inspired oxygen: "0.5, 0.2, 0.47, 0.44, 1.03"
- Glucose: "7527.47, 331.66, 2821.62, 2679.17, 7756.74"
- Heart Rate: "124.23, 18.91, 61.73, 150.03, 187.6"
- Height: "191.48, 171.01, 169.08, 197.79, 161.34"
- Mean blood pressure: "648.18, 3700.99, 7252.52, 986.85, 3847.55"
- Oxygen saturation: "1789.69, 67.02, 599.62, 472.91, 105.99"
- Respiratory rate: "137.05, 57.86, 281.02, 221.34, 79.01"
- Systolic blood pressure: "271.48, 237.76, 245.35, 24.73, 57.08"
- Temperature: "41.01, 34.62, 48.93, 20.55, 41.96"
- Weight: "4802.09, 3210.31, 3011.51, 2915.57, 1808.56"
- pH: "9122.89, 8101.48, 1093.88, 24542.72, 11425.86"

RESPONSE:
0.25
''',
        ]
    },
}