import os
import subprocess


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
        self.num_plan_processes = 6
        self.set_env()

    def set_env(self):
        os.environ[self.NNUNET_RAW] = self.raw_dir
        os.environ[self.NNUNET_PREPROCESSED] = self.preprocessed_dir
        os.environ[self.NNUNET_RESULTS] = self.preds_dir
        os.environ["nnUNet_def_n_proc"] = "6"
        os.environ["nnUNet_n_proc_DA"] = "6"
        os.environ["nnUNet_compile"] = "1"
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.preprocessed_dir, exist_ok=True)
        os.makedirs(self.preds_dir, exist_ok=True)

    def determine_postprocessing(self, input_folder, gt_labels_folder, dj_file, p_file):
        """
        Determine PostProcessing. Creates PKL file in input_folder. Copy these
        """
        self.set_env()
        run_args = [
            "nnUNetv2_determine_postprocessing",
            "-i",
            input_folder,
            "-ref",
            gt_labels_folder,
            "--remove_postprocessed",
            "-plans_json",
            p_file,
            "-dataset_json",
            dj_file,
        ]
        self._run_subprocess(run_args)

    def apply_postprocessing(self, input_folder, output_folder, pkl_file):
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

    def plan(self, dataset_id: str, config: str, gpu_memory_gb: int = None):
        self.set_env()
        run_args = [
            "nnUNetv2_plan_and_preprocess",
            "-d",
            dataset_id,
            "--verify_dataset_integrity",
            "--clean",
            "-c",
            config,
            "-np",
            self.num_plan_processes,
        ]
        if gpu_memory_gb is not None:
            run_args.extend(["-gpu_memory_target", gpu_memory_gb])
        self._run_subprocess(run_args)

    def evaluate_on_folder(self, gt_dir, preds_dir, dj_file, p_file):
        self.set_env()
        run_args = [
            "nnUNetv2_evaluate_folder",
            gt_dir,
            preds_dir,
            "-djfile",
            dj_file,
            "-pfile",
            p_file,
            "--chill",
        ]
        self._run_subprocess(run_args)

    def train(
        self,
        dataset_id: str,
        model_config: str,
        fold: int,
        trainer_name: str = "nnUNetTrainer",
        resume: bool = True,
        device_id: int = 0,
    ):
        self.set_env()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        run_args = [
            "nnUNetv2_train",
            dataset_id,
            model_config,
            fold,
            "-tr",
            trainer_name,
        ]

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
    def _run_subprocess(run_args, env=None):
        """Synchronous call to nnunet"""
        run_args = [str(i) for i in run_args]
        subprocess.run(
            run_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE,
            check=True,
            env=env,
        )


default_nnunet_adapter = NNUNetV2Adapter(
    "data/nnUNet_raw",
    "data/nnUNet_preprocessed",
    "data/nnUNet_results",
)
