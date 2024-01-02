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
    'outcome': 'Your task is to assess the provided medical data and analyze the health records from ICU visits to determine the likelihood of the patient not surviving their hospital stay. Please respond with a floating-point number between 0 and 1, where a higher number suggests a greater likelihood of death.',
    'los': 'Your task is to Evaluate the provided medical data to estimate the remaining duration of the ICU stay. Consider the progression of health across multiple visits to forecast the length of intensive care needed. Please respond with an integer indicating the number of days expected in the ICU.',
    'readmission': 'Your task is to analyze the medical history to predict the probability of readmission within 30 days post-discharge. Include cases where a patient passes away within 30 days from the discharge date. Please respond with a floating-point number between 0 and 1, where a higher number suggests a greater likelihood of readmission.',
}

EXAMPLE = {
    'tjh': {
        'batches': 'prompts/tjh/N-1_batches_example.txt',
        'list': 'prompts/tjh/N-1_list_example.txt',
        'string': 'prompts/tjh/N-1_string_example.txt',
    },
    'mimic-iv': {
        'batches': 'prompts/mimic-iv/N-1_batches_example.txt',
        'list': 'prompts/mimic-iv/N-1_list_example.txt',
        'string': 'prompts/mimic-iv/N-1_string_example.txt',
    },
}