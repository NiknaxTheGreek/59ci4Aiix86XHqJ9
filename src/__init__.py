"""Public imports for the MonReader reproducible research workflow."""
from .common import setup, file_sha256, write_json, get_settings, load_settings
from .data import discover_manifest, audit_manifest, FrameDataset, make_loader, read_archived_result_table
from .splits import make_splits, validate_split, save_splits
from .models import ScratchCNN, get_resnet, build_optimizer, predict_frame_table, performance, fit_model, train_resnet_regime, train_grayscale, load_resnet_checkpoint, fit_resnet_two_stage
from .tuning import resolve_config, tune_one_regime_wandb
from .evaluation import source_bootstrap, paired_source_bootstrap, classify_errors
from .visualization import plot_regime_examples, plot_model_comparison, plot_grayscale_examples, cam_for_image, select_cam_cases, plot_cam_matrix, select_21_interpretability_cases, generate_activation_diagnostics
from .common import REGIMES, CLASS_TO_Y
