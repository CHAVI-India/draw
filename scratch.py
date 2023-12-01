from a9t.predict import folder_predict
from a9t.adapters.nnunetv2 import default_nnunet_adapter

# Env set mandatory
default_nnunet_adapter.set_env()
folder_predict(
    root_dir="data/raw/TSPrime_test",
    preds_dir="output",
    dataset_name="TSPrime",
    only_original=True,
)
