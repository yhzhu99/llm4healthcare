SYSTEMPROMPT = "You are an experienced doctor in the field of COVID-19 treatment."

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
# USERPROMPT = {
#     '1-1': 'prompts/1-1_prompt.txt',
#     'N-1_batches': 'prompts/N-1_batches_prompt.txt',
#     'N-1_list': 'prompts/N-1_list_prompt.txt',
#     'N-1_string': 'prompts/N-1_string_prompt.txt',
# }
# EXAMPLE = {
#     'tjh': {
#         '1-1': 'prompts/tjh/1-1_example.txt',
#         'N-1_batches': 'prompts/tjh/N-1_batches_example.txt',
#         'N-1_list': 'prompts/tjh/N-1_list_example.txt',
#         'N-1_string': 'prompts/tjh/N-1_string_example.txt',
#     },
#     'mimic-iv': {
#         '1-1': 'prompts/mimic-iv/1-1_example.txt',
#         'N-1_batches': 'prompts/mimic-iv/N-1_batches_example.txt',
#         'N-1_list': 'prompts/mimic-iv/N-1_list_example.txt',
#         'N-1_string': 'prompts/mimic-iv/N-1_string_example.txt',
#     },
# }