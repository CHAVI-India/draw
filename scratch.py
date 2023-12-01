from main import predict

# def predict(preds_dir, dataset_name, root_dir, only_original):
predict(
    root_dir="data/raw/TSPrime_test",
    preds_dir="output",
    dataset_name="TSPrime",
    only_original=True,
)
