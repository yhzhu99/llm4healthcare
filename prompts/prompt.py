SYSTEMPROMPT = {
    'tjh': 'You are an experienced doctor in the field of COVID-19 treatment.',
    'mimic-iv': 'You are an experienced doctor in Intensive Care Unit (ICU) treatment.',
}
USERPROMPT = '''I will provide you with medical information from multiple Intensive Care Unit (ICU) visits of a patient, each characterized by a fixed number of features.

{INPUT_FORMAT_DESCRIPTION}

{TASK_DESCRIPTION_AND_RESPONSE_FORMAT}

In situations where the data does not allow for a reasonable conclusion, respond with the phrase `I do not know` without any additional explanation.

{UNIT_RANGE_CONTEXT}

{EXAMPLE}

Now please predict the patient below:
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

MISSING_VALUE_DESCRIPTION = ' Missing values are represented as `nan`.'

INSTRUCTING_MISSING_VALUE = 'Values followed by `**` are initially not a number and imputed through Last Observation Carried Forward(LOCF) method. If a large number of values of a certain feature are filled, you need to consider that the credibility of the analysis results for that feature is relatively low.'

TASK_DESCRIPTION_AND_RESPONSE_FORMAT = {
    'outcome': {
        'upon-discharge': 'Your task is to assess the provided medical data and analyze the health records from ICU visits to determine the likelihood of the patient not surviving their hospital stay. Please respond with only a floating-point number between 0 and 1, where a higher number suggests a greater likelihood of death.',
        '1month': 'Your task is to assess the provided medical data and analyze the health records from ICU visits to determine the likelihood of death within 30 days post-discharge. Include cases where a patient does not survive their hospital stay. Please respond with only a floating-point number between 0 and 1, where a higher number suggests a greater likelihood of death.',
        '6months': 'Your task is to assess the provided medical data and analyze the health records from ICU visits to determine the likelihood of death within 6 months post-discharge. Include cases where a patient does not survive their hospital stay. Please respond with only a floating-point number between 0 and 1, where a higher number suggests a greater likelihood of death.',
    },
    'los': 'Your task is to evaluate the provided medical data to estimate the remaining duration of the ICU stay. Consider the progression of health across multiple visits to forecast the length of intensive care needed. Please respond with a sequence of integers with each one indicating the number of days expected in the ICU during their current visit.',
    'readmission': 'Your task is to analyze the medical history to predict the probability of readmission within 30 days post-discharge. Include cases where a patient passes away within 30 days from the discharge date. Please respond with only a floating-point number between 0 and 1, where a higher number suggests a greater likelihood of readmission.',
    'multitask': 'Your task is to assess the provided medical data and analyze the health records from ICU visits to determine the likelihood of the patient not surviving their hospital stay and predict the probability of readmission within 30 days post-discharge including cases where a patient passes away within 30 days from the discharge date. Please respond with 2 floating-point numbers between 0 and 1, the first one is the likelihood of death and the second one is the likelihood of readmission, where a higher number suggests a greater likelihood of death or readmission. Do not include any additional explanation.'
}

RESPONSE_FORMAT = {
    'outcome': 'Please respond with only a floating-point number between 0 and 1, where a higher number suggests a greater likelihood of death. Do not include any additional explanation.',
    'los': 'Please respond with a sequence of integers with each one indicating the number of days expected in the ICU during their current visit. Do not include any additional explanation.',
    'readmission': 'Please respond with only a floating-point number between 0 and 1, where a higher number suggests a greater likelihood of readmission. Do not include any additional explanation.',
    'multitask': 'Please respond with 2 floating-point numbers between 0 and 1, the first one is the likelihood of death and the second one is the likelihood of readmission, where a higher number suggests a greater likelihood of death or readmission. Do not include any additional explanation.',
    'cot': 'Please follow the Chain-of-Thought Analysis Process and respond with a floating-point number between 0 and 1 or `I do not know` at last.'
}

EXAMPLE = {
    'tjh': {
        'outcome': [
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
        ],
        'los': [
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
9, 8, 5, 4, 1
'''
        ]
    },
    'mimic-iv': {
        'outcome': [
'''
Input information of a patient:
The patient is a female, aged 52 years.
The patient had 4 visits that occurred at 0, 1, 2, 3.
Details of the features for each visit are as follows:
- Capillary refill rate: "unknown, unknown, unknown, unknown"
- Glascow coma scale eye opening: "Spontaneously, Spontaneously, Spontaneously, Spontaneously"
- Glascow coma scale motor response: "Obeys Commands, Obeys Commands, Obeys Commands, Obeys Commands"
......

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
......

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
......

RESPONSE:
0.25
''',
        ],
        'readmission': [
'''
Input information of a patient:
The patient is a female, aged 52 years.
The patient had 4 visits that occurred at 0, 1, 2, 3.
Details of the features for each visit are as follows:
- Capillary refill rate: "unknown, unknown, unknown, unknown"
- Glascow coma scale eye opening: "Spontaneously, Spontaneously, Spontaneously, Spontaneously"
- Glascow coma scale motor response: "Obeys Commands, Obeys Commands, Obeys Commands, Obeys Commands"
......

RESPONSE:
0.3
''',
        ],
        'multitask': [
'''
Input information of a patient:
The patient is a female, aged 52 years.
The patient had 4 visits that occurred at 0, 1, 2, 3.
Details of the features for each visit are as follows:
- Capillary refill rate: "unknown, unknown, unknown, unknown"
- Glascow coma scale eye opening: "Spontaneously, Spontaneously, Spontaneously, Spontaneously"
- Glascow coma scale motor response: "Obeys Commands, Obeys Commands, Obeys Commands, Obeys Commands"
......

RESPONSE:
0.3, 0.4
''',
        ]
    },
}

COT = {
    'tjh': '',
    'mimic-iv': '''
Please follow the Chain-of-Thought Analysis Process:

1. Analyze the data step by step, For example:
   - Blood pressure shows a slight downward trend, indicating...
   - Heart rate is stable, suggesting...
   - Lab results indicate [specific condition or lack thereof]...
   - The patient underwent [specific intervention], which could mean...

2. Make Intermediate Conclusions:
   - Draw intermediate conclusions from each piece of data. For example:
     - If a patient's blood pressure is consistently low, it might indicate poor cardiovascular function.
     - The patient's cardiovascular function is [conclusion].
     - [Other intermediate conclusions based on data].

3. Aggregate the Findings:
   - After analyzing each piece of data, aggregate these findings to form a comprehensive view of the patient's condition.
   - Summarize key points from the initial analysis and intermediate conclusions.

Aggregated Findings:
- Considering the patient's vital signs and lab results, the overall health status is...

4. Final Assessment:
   - Conclude with an assessment of the patient's likelihood of not surviving their hospital stay.
   - Provide a floating-point number between 0 and 1, where a higher number suggests a greater likelihood of death.
   - If the data is insufficient or ambiguous, conclude with "I do not know."

[0.XX] or "I do not know."

Here is an example of Input Information and Response:
Example #1:
Input information of a patient:
The patient is a female, aged 52 years.
The patient had 4 visits that occurred at 0, 1, 2, 3.
Details of the features for each visit are as follows:
- Mean blood pressure: "83.42, 79.5, 73.92, 73.0"
- Heart Rate: "83.56, 82.55, 81.5, 81.75"
- Respiratory rate: "16.05, 13.9, 16.53, 27.39"
......

RESPONSE:
1. Analyze the data step by step:
   - Blood pressure shows a slight downward trend, which might indicate a gradual decline in cardiovascular stability.
   - Heart rate is stable, which is a good sign, suggesting no immediate cardiac distress.
   - The respiratory rate initially is stable, but there is a significant increase in the last reading, which could indicate respiratory distress, possibly due to a lung infection, pulmonary embolism, or other respiratory complications.

2. Make Intermediate Conclusions:
   - The decreasing blood pressure could be a sign of worsening heart function or infection-related hypotension.
   - Stable heart rate is reassuring but does not completely rule out underlying issues.
   - The sudden increase in respiratory rate is concerning, indicating potential acute respiratory problems.

3. Aggregate the Findings:
   - The patient is possibly facing a cardiovascular challenge, compounded by an infection and electrolyte imbalance.

Aggregated Findings:
- Combining the trends in blood pressure, heart rate, and respiratory rate, it appears that the patient's condition is deteriorating, particularly in terms of cardiovascular and respiratory function.

4. Final Assessment:
0.75
'''
}