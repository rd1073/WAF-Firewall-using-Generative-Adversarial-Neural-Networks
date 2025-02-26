import pandas as pd

file_path = "dataset/SQLInjection_XSS_MixDataset.1.0.0.csv"
try:
    #data = pd.read_csv(file_path, delimiter="\t", error_bad_lines=False, warn_bad_lines=True)
    data = pd.read_csv(file_path, delimiter="\t", on_bad_lines="skip")

    data = pd.read_csv(file_path, delimiter=",")

    #print(data.head())
except Exception as e:
    print(f"Error loading dataset: {e}")
