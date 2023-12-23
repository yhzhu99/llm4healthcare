OPENAI_API_KEY = ""

params = [
    {
        'model': 'gpt-4-1106-preview',  # gpt-4, gpt-3.5-turbo ...
        'prediction': 'N-1',    # 1-1, N-1
        'format': 'string',    # batches, list, string, only for N-1
        'forwardfill': True,
        'shot': True,
        'unit': False,
        'range': False,
    },
]