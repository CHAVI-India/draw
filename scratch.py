import pandas as pd

if __name__ == "__main__":
    df = pd.read_json("data/nnUNet_raw/Dataset720_TSPrime/db.json")
    print(df)
