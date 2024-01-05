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
......

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
......

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
......

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
The patient had 4 visits that occurred at 0, 1, 2, 3.
Details of the features for each visit are as follows:
- Capillary refill rate: "unknown, unknown, unknown, unknown"
- Glascow coma scale eye opening: "Spontaneously, Spontaneously, Spontaneously, Spontaneously"
- Glascow coma scale motor response: "Obeys Commands, Obeys Commands, Obeys Commands, Obeys Commands"
...

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
...

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
...

RESPONSE:
0.25
''',
        ]
    },
}