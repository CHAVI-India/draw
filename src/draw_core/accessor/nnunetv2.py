"""Adapter around nnU-Net v2.

Refactored from ``draw/accessor/nnunetv2.py``. Key changes:

* No module-level singleton. The legacy ``default_nnunet_adapter`` was built at
  import time, mutating ``os.environ`` and creating dirs merely on import. Here the
  adapter is constructed explicitly by an entrypoint from a ``CoreConfig``.
* Importing this module does NOT import torch/nnU-Net. Heavy imports happen lazily
  inside the methods that need them, so ``draw_core`` stays importable without a GPU.
* Adds an optional warm in-process predictor (GPU perf levers 1, 2, 5) alongside the
  existing subprocess CLI path (kept for backward compatibility).

GPU perf levers implemented here (the no-retrain set: 1, 2, 5, 7, 8):
  1. Warm resident model        -> WarmPredictor caches nnUNetPredictor per model.
  2. Collapse 3 cold loads -> 1  -> one predictor reused across studies.
  5. torch.compile + cudnn.benchmark for fixed patch sizes.
  7. Per-GPU pinning via CUDA_VISIBLE_DEVICES (MIG-friendly).
  8. perform_everything_on_device keeps resampling/argmax on GPU.
Levers 3, 4, 6 (step_size/FP16/re-plan) are intentionally NOT enabled — they need
clinical re-validation, which is out of scope for now.
"""

from __future__ import annotations

import os
import subprocess

from draw_core.config import CoreConfig
from draw_core.logging import get_logger

log = get_logger(__name__)


class NNUNetV2Adapter:
    """Thin wrapper over nnU-Net v2, configured from explicit paths (no globals)."""

    NNUNET_RAW = "nnUNet_raw"
    NNUNET_PREPROCESSED = "nnUNet_preprocessed"
    NNUNET_RESULTS = "nnUNet_results"

    def __init__(self, config: CoreConfig, num_plan_processes: int = 6):
        self.config = config
        self.raw_dir = config.nnunet_raw_dir
        self.preprocessed_dir = config.nnunet_preprocessed_dir
        self.results_dir = config.nnunet_results_dir
        self.num_plan_processes = num_plan_processes

    @classmethod
    def from_config(cls, config: CoreConfig) -> NNUNetV2Adapter:
        return cls(config)

    def set_env(self) -> None:
        """Export nnU-Net env vars and ensure data dirs exist.

        Called explicitly before nnU-Net operations rather than at import time.
        """
        os.environ[self.NNUNET_RAW] = self.raw_dir
        os.environ[self.NNUNET_PREPROCESSED] = self.preprocessed_dir
        os.environ[self.NNUNET_RESULTS] = self.results_dir
        os.environ.setdefault("nnUNet_def_n_proc", str(self.num_plan_processes))
        os.environ.setdefault("nnUNet_n_proc_DA", str(self.num_plan_processes))
        os.environ.setdefault("nnUNet_compile", "1")  # lever 5
        for d in (self.raw_dir, self.preprocessed_dir, self.results_dir):
            os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------ predict
    def predict_folder(
        self,
        samples_dir: str,
        output_dir: str,
        model_config: str,
        dataset_id: str,
        fold: str,
        trainer_name: str = "nnUNetTrainer",
        checkpoint_name: str | None = None,
    ) -> None:
        """Run inference via the nnU-Net CLI (subprocess). Backward-compatible path."""
        self.set_env()
        checkpoint_name = checkpoint_name or self.config.checkpoint_name
        run_args = [
            "nnUNetv2_predict",
            "-i", samples_dir,
            "-o", output_dir,
            "-c", model_config,
            "-d", dataset_id,
            "-f", fold,
            "-chk", checkpoint_name,
            "-device", "cuda",
            "-tr", trainer_name,
        ]
        if self.config.disable_tta:
            run_args.append("--disable_tta")
        self._run_subprocess(run_args)

    # ------------------------------------------------------------ train / plan
    def plan(self, dataset_id: str, config: str, gpu_memory_gb: int | None = None) -> None:
        self.set_env()
        run_args = [
            "nnUNetv2_plan_and_preprocess",
            "-d", dataset_id,
            "--verify_dataset_integrity",
            "--clean",
            "-c", config,
            "-np", str(self.num_plan_processes),
        ]
        if gpu_memory_gb is not None:
            run_args.extend(["-gpu_memory_target", str(gpu_memory_gb)])
        self._run_subprocess(run_args)

    def train(
        self,
        dataset_id: str,
        model_config: str,
        fold: str,
        trainer_name: str = "nnUNetTrainer",
        resume: bool = True,
        device_id: int = 0,
    ) -> None:
        self.set_env()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)  # lever 7
        run_args = ["nnUNetv2_train", dataset_id, model_config, fold, "-tr", trainer_name]
        if resume:
            run_args.append("--c")
        self._run_subprocess(run_args)

    def evaluate_on_folder(self, gt_dir: str, preds_dir: str, dj_file: str, p_file: str) -> None:
        self.set_env()
        self._run_subprocess([
            "nnUNetv2_evaluate_folder", gt_dir, preds_dir,
            "-djfile", dj_file, "-pfile", p_file, "--chill",
        ])

    def determine_postprocessing(
        self, input_folder: str, gt_labels_folder: str, dj_file: str, p_file: str
    ) -> None:
        self.set_env()
        self._run_subprocess([
            "nnUNetv2_determine_postprocessing",
            "-i", input_folder, "-ref", gt_labels_folder,
            "--remove_postprocessed", "-plans_json", p_file, "-dataset_json", dj_file,
        ])

    def apply_postprocessing(self, input_folder: str, output_folder: str, pkl_file: str) -> None:
        self.set_env()
        self._run_subprocess([
            "nnUNetv2_apply_postprocessing",
            "-i", input_folder, "-o", output_folder, "-pp_pkl", pkl_file,
        ])

    @staticmethod
    def _run_subprocess(run_args, env=None) -> None:
        """Run an nnU-Net CLI command with detailed, cross-platform diagnostics.

        The legacy version captured stdout/stderr into a PIPE but never read it, so a
        failing child raised a bare ``CalledProcessError`` with no detail ("blows up
        for no reason"). Here we capture and LOG the full output on failure, force
        UTF-8 + unbuffered I/O (Windows console defaults to cp1252 and block-buffers
        non-TTY output, hiding the real error), and report timing + exit code.
        """
        import os
        import shutil
        import time

        run_args = [str(a) for a in run_args]
        program = run_args[0]

        # Child env: inherit, layer any overrides, force unbuffered UTF-8 so the real
        # traceback (esp. from nnU-Net's spawned worker processes on Windows) survives.
        child_env = {**os.environ, **(env or {})}
        child_env.setdefault("PYTHONUNBUFFERED", "1")
        child_env.setdefault("PYTHONIOENCODING", "utf-8")

        resolved = shutil.which(program)
        log.info("nnU-Net: starting %s", program)
        log.info("nnU-Net: full command: %s", " ".join(run_args))
        log.debug("nnU-Net: resolved executable: %s", resolved or "<NOT ON PATH>")
        log.debug(
            "nnU-Net: env nnUNet_raw=%s nnUNet_results=%s CUDA_VISIBLE_DEVICES=%s compile=%s",
            child_env.get("nnUNet_raw"),
            child_env.get("nnUNet_results"),
            child_env.get("CUDA_VISIBLE_DEVICES", "<unset>"),
            child_env.get("nnUNet_compile", "<unset>"),
        )
        if resolved is None:
            log.error(
                "nnU-Net: '%s' not found on PATH. Is the 'gpu' extra installed in this "
                "environment? PATH=%s", program, child_env.get("PATH", ""),
            )

        start = time.perf_counter()
        try:
            proc = subprocess.run(
                run_args,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # merge so ordering is preserved in one stream
                text=True,
                encoding="utf-8",
                errors="replace",  # never crash the launcher on odd bytes
                env=child_env,
            )
        except FileNotFoundError:
            log.error("nnU-Net: executable '%s' not found (FileNotFoundError)", program)
            raise
        except OSError:
            log.error("nnU-Net: OS error launching '%s'", program, exc_info=True)
            raise

        elapsed = time.perf_counter() - start
        output = proc.stdout or ""

        if proc.returncode != 0:
            log.error("nnU-Net: %s FAILED (exit %d) after %.1fs", program, proc.returncode, elapsed)
            log.error("nnU-Net: full command was: %s", " ".join(run_args))
            log.error(
                "nnU-Net: ---- captured output (stdout+stderr) ----\n%s\n---- end output ----",
                output.strip() or "<no output captured>",
            )
            # Surface the tail prominently — the actionable error is usually last.
            tail = "\n".join(output.strip().splitlines()[-15:])
            log.error("nnU-Net: last 15 lines:\n%s", tail or "<empty>")
            raise subprocess.CalledProcessError(proc.returncode, run_args, output)

        log.info("nnU-Net: %s completed in %.1fs", program, elapsed)
        log.debug("nnU-Net: output:\n%s", output.strip() or "<no output>")
