OPENAI_API_KEY = ""
TJH_FEATURES = {
    'Prediction': ['Outcome', 'LOS'],
    'Demographics': ['Sex', 'Age'], 
    'Laboratory': [
        'Hypersensitive cardiac troponinI', 'hemoglobin',
        'Serum chloride', 'Prothrombin time', 'procalcitonin', 'eosinophils(%)',
        'Interleukin 2 receptor', 'Alkaline phosphatase', 'albumin',
        'basophil(%)', 'Interleukin 10', 'Total bilirubin', 'Platelet count',
        'monocytes(%)', 'antithrombin', 'Interleukin 8', 'indirect bilirubin',
        'Red blood cell distribution width ', 'neutrophils(%)', 'total protein',
        'Quantification of Treponema pallidum antibodies',
        'Prothrombin activity', 'HBsAg', 'mean corpuscular volume',
        'hematocrit', 'White blood cell count', 'Tumor necrosis factorα',
        'mean corpuscular hemoglobin concentration', 'fibrinogen',
        'Interleukin 1β', 'Urea', 'lymphocyte count', 'PH value',
        'Red blood cell count', 'Eosinophil count', 'Corrected calcium',
        'Serum potassium', 'glucose', 'neutrophils count', 'Direct bilirubin',
        'Mean platelet volume', 'ferritin', 'RBC distribution width SD',
        'Thrombin time', '(%)lymphocyte', 'HCV antibody quantification',
        'D-D dimer', 'Total cholesterol', 'aspartate aminotransferase',
        'Uric acid', 'HCO3-', 'calcium',
        'Amino-terminal brain natriuretic peptide precursor(NT-proBNP)',
        'Lactate dehydrogenase', 'platelet large cell ratio ', 'Interleukin 6',
        'Fibrin degradation products', 'monocytes count',
        'PLT distribution width', 'globulin', 'γ-glutamyl transpeptidase',
        'International standard ratio', 'basophil count(#)',
        'mean corpuscular hemoglobin ',
        'Activation of partial thromboplastin time',
        'High sensitivity C-reactive protein', 'HIV antibody quantification',
        'serum sodium', 'thrombocytocrit', 'ESR',
        'glutamic-pyruvic transaminase', 'eGFR', 'creatinine'
    ]
}
MIMIC_FEATURES = {
    'Prediction': ['Outcome', 'LOS', 'Readmission'],
    'Demographics': ['Sex', 'Age'],
    'Laboratory': {
        'Categorical': ['Capillary refill rate', 'Glascow coma scale eye opening', 'Glascow coma scale motor response', 'Glascow coma scale total', 'Glascow coma scale verbal response'],
        'Numerical': ['Diastolic blood pressure', 'Fraction inspired oxygen', 'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight', 'pH']
    }
}
params = [
    {
        'model': 'gpt-4-1106-preview',  # gpt-4, gpt-3.5-turbo ...
        'dataset': 'tjh',    # tjh, mimic-iv
        'prediction': 'N-1',    # 1-1, N-1
        'format': 'string',    # batches, list, string, only for N-1
        'forwardfill': True,
        'shot': True,
        'unit': False,
        'range': False,
    },
]