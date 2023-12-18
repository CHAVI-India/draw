import os
import subprocess
import sys


class NNUNetV2Adapter:
    """
    Provides a simplified python class for using NNUNet via the CLI.
    Assumes ``nnUNet_v2`` is installed
    """

    NNUNET_RAW = "nnUNet_raw"
    NNUNET_PREPROCESSED = "nnUNet_preprocessed"
    NNUNET_RESULTS = "nnUNet_results"

    def __init__(self, raw_dir, preprocessed_dir, preds_dir):
        self.raw_dir = raw_dir
        self.preprocessed_dir = preprocessed_dir
        self.preds_dir = preds_dir
        self.set_env()

    def set_env(self):
        os.environ[self.NNUNET_RAW] = self.raw_dir
        os.environ[self.NNUNET_PREPROCESSED] = self.preprocessed_dir
        os.environ[self.NNUNET_RESULTS] = self.preds_dir
        os.environ["nnUNet_def_n_proc"] = "4"
        os.environ["nnUNet_n_proc_DA"] = "4"
        os.environ["nnUNet_compile"] = "1"

    def postprocess(self, input_folder, output_folder, pkl_file):
        self.set_env()
        run_args = [
            "nnUNetv2_apply_postprocessing",
            "-i",
            input_folder,
            "-o",
            output_folder,
            "-pp_pkl",
            pkl_file,
        ]
        self._run_subprocess(run_args)

    def preprocess(self, dataset_id: str):
        self.set_env()
        run_args = [
            "nnUNetv2_plan_and_preprocess",
            "-d",
            dataset_id,
            "--verify_dataset_integrity",
        ]
        self._run_subprocess(run_args)

    def evaluate_on_folder(self, labels_dir, preds_dir, dj_file, p_file):
        self.set_env()
        run_args = [
            "nnUNetv2_evaluate_folder",
            labels_dir,
            preds_dir,
            "-djfile",
            dj_file,
            "-pfile",
            p_file,
            "--chill",
        ]
        self._run_subprocess(run_args)

    def train(self, dataset_id: str, model_config: str, fold: int, resume: bool = True):
        self.set_env()
        run_args = ["nnUNetv2_train", dataset_id, model_config, fold]

        if resume:
            run_args.append("--c")

        self._run_subprocess(run_args)

    def predict_folder(
        self,
        samples_dir,
        output_dir,
        model_config,
        dataset_id,
        fold,
        trainer_name="nnUNetTrainer",
        checkpoint_name="checkpoint_best.pth",
    ):
        self.set_env()
        run_args = [
            "nnUNetv2_predict",
            "-i",
            samples_dir,
            "-o",
            output_dir,
            "-c",
            model_config,
            "-d",
            dataset_id,
            "-f",
            fold,
            "-chk",
            checkpoint_name,
            "--disable_tta",
            "-device",
            "cuda",
            "-tr",
            trainer_name,
        ]
        self._run_subprocess(run_args)

    @staticmethod
    def _run_subprocess(run_args):
        """Synchronous call to nnunet"""
        run_args = [str(i) for i in run_args]
        subprocess.run(
            run_args,
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=True,
            # shell=True,
        )


default_nnunet_adapter = NNUNetV2Adapter(
    "data/nnUNet_raw",
    "data/nnUNet_preprocessed",
    "data/nnUNet_results",
)
