import pandas as pd

df = pd.read_csv(
        "data/nnUNet_raw/Dataset720_TSPrime/db.csv",
        dtype={
            "DatasetID": int,
            "SampleNumber": str,
            "DICOMRootDir": str,
        },
        index_col=None,
    )
print(df)
op = df.loc[(df["DatasetID"] == "720") & (df["SampleNumber"] == "000")]
print(op)