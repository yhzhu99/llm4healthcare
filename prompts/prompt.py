SYSTEMPROMPT = "You are an experienced doctor in the field of COVID-19 treatment."
USERPROMPT = {
    '1-1': 'prompts/1-1_prompt.txt',
    'N-1_batches': 'prompts/N-1_batches_prompt.txt',
    'N-1_list': 'prompts/N-1_list_prompt.txt',
    'N-1_string': 'prompts/N-1_string_prompt.txt',
}
EXAMPLE = {
    'tjh': {
        '1-1': 'prompts/tjh/1-1_example.txt',
        'N-1_batches': 'prompts/tjh/N-1_batches_example.txt',
        'N-1_list': 'prompts/tjh/N-1_list_example.txt',
        'N-1_string': 'prompts/tjh/N-1_string_example.txt',
    },
    'mimic-iv': {
        '1-1': 'prompts/mimic-iv/1-1_example.txt',
        'N-1_batches': 'prompts/mimic-iv/N-1_batches_example.txt',
        'N-1_list': 'prompts/mimic-iv/N-1_list_example.txt',
        'N-1_string': 'prompts/mimic-iv/N-1_string_example.txt',
    },
}

UNIT_RANGE_PROMPT = "Each time I provide you with the following information:\n"
UNIT = {
    'tjh': 'prompts/tjh/unit.json',
    'mimic-iv': 'prompts/mimic-iv/unit.json',
}
RANGE = {
    'tjh': 'prompts/tjh/range.json',
    'mimic-iv': 'prompts/mimic-iv/range.json',
}