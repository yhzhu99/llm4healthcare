OPENAI_API_KEY = ""

params = [
    {
        'model': 'gpt-4-1106-preview',  
        'dataset': 'tjh',    # tjh, mimic-iv
        'form': 'string',    # batches, string (, list)
        'task': 'outcome',  # outcome(tjh, mimic-iv), los(tjh), readmission(mimic-iv)
        'n_shot': 0,    # 0, 1...
        'unit': True,
        'reference_range': False,
    },
]