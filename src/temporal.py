"""Progressive T/ST proof of principle on observed chronological frame positions.

No online learning, real-time event forecasting, or unseen-source adaptation is claimed.
All train/validation/test frame assignments are made before any model evaluation.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .common import setup, write_json
from .models import get_resnet, fit_resnet_two_stage, predict_frame_table, performance


COVERAGES = (0.25, 0.50, 0.75, 0.90)
HORIZON_WIDTH = 3
MIN_FRAMES_PER_CLASS = 20
MIN_SOURCES_PER_CLASS = 8


def environment_source_keys(manifest, environment_csv, heldout_environment='ENV-03'):
    """Identify an unseen source cohort from the predeclared environment map."""
    mapping = pd.read_csv(environment_csv)
    required = {'video_id', 'environment_id'}
    if not required.issubset(mapping.columns):
        raise ValueError('Environment map requires video_id and environment_id')
    # The checked archive defines canonical video_id as label/zero-padded-number.
    source_map = mapping.assign(
        source_key=mapping.video_id.astype(str).str.replace('/', '::', regex=False)
    )
    if source_map.source_key.duplicated().any():
        raise ValueError('Duplicate source assignment in environment map')
    known = set(source_map.loc[source_map.environment_id != heldout_environment, 'source_key'])
    unseen = set(source_map.loc[source_map.environment_id == heldout_environment, 'source_key'])
    present = set(manifest.source_key.astype(str))
    if (known & unseen) or not unseen or (known | unseen) != present:
        raise AssertionError('Environment membership is incomplete or overlapping')
    return known, unseen


def chronological_checkpoint(manifest, regime, coverage, known_sources=None, unseen_sources=None):
    """Build a chronological training prefix and a protected future evaluation suffix.

    Validation uses the final at-most-20% of each *available historical prefix*.
    ST test sources are entirely excluded from training and validation.
    """
    if regime not in ('T', 'ST') or coverage not in COVERAGES:
        raise ValueError('Invalid regime or coverage')
    if regime == 'ST' and (known_sources is None or unseen_sources is None):
        raise ValueError('ST requires a frozen known/unseen source partition')

    train_parts, validation_parts, future_parts = [], [], []
    for source, group in manifest.groupby('source_key', sort=True):
        group = group.sort_values(['frame_number', 'sample_id']).reset_index(drop=True).copy()
        length = len(group)
        if length < 4:
            raise ValueError('Temporal experiment requires at least four frames per source')
        observed = max(2, math.floor(coverage * length))
        if observed >= length:
            raise AssertionError('No protected future frames remain')
        train_end = max(1, observed - max(1, math.floor(.20 * observed)))
        historical_train = group.iloc[:train_end].copy()
        historical_val = group.iloc[train_end:observed].copy()
        future = group.iloc[observed:].copy()
        future['horizon_position'] = np.arange(1, len(future) + 1)
        future['frame_position_percent'] = (
            np.arange(observed, length) + 1
        ) / length * 100

        is_training_source = regime == 'T' or source in known_sources
        is_evaluation_source = regime == 'T' or source in unseen_sources
        if is_training_source:
            train_parts.append(historical_train)
            validation_parts.append(historical_val)
        if is_evaluation_source:
            future_parts.append(future)

    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(validation_parts, ignore_index=True)
    future = pd.concat(future_parts, ignore_index=True)

    for name, part in [('train', train), ('validation', val), ('future', future)]:
        if set(part.label) != {'flip', 'notflip'}:
            raise AssertionError(f'{name}: both classes required')
    if set(train.sample_id) & set(future.sample_id):
        raise AssertionError('Training/test sample overlap')
    if set(val.sample_id) & set(future.sample_id):
        raise AssertionError('Validation/test sample overlap')
    if regime == 'ST':
        if set(train.source_key) & set(future.source_key):
            raise AssertionError('ST: unseen-source leakage')
        if set(val.source_key) & set(future.source_key):
            raise AssertionError('ST: unseen-source validation leakage')
    return train, val, future


def horizon_summary(predictions, width=HORIZON_WIDTH):
    """Pool corresponding future frame positions; omit unstable scored windows."""
    if predictions.empty:
        raise ValueError('No predictions')
    report = []
    for window_index, group in predictions.groupby(
        (predictions.horizon_position.astype(int) - 1) // width, sort=True
    ):
        start = int(window_index * width + 1)
        end = start + width - 1
        counts = group.label.value_counts()
        source_counts = group.groupby('label').source_key.nunique()
        eligible = (
            counts.get('flip', 0) >= MIN_FRAMES_PER_CLASS
            and counts.get('notflip', 0) >= MIN_FRAMES_PER_CLASS
            and source_counts.get('flip', 0) >= MIN_SOURCES_PER_CLASS
            and source_counts.get('notflip', 0) >= MIN_SOURCES_PER_CLASS
        )
        metrics = performance(group) if eligible else {}
        report.append(dict(
            start_horizon=start, end_horizon=end,
            n_frames=len(group), n_sources=int(group.source_key.nunique()),
            n_flip=int(counts.get('flip', 0)),
            n_notflip=int(counts.get('notflip', 0)),
            flip_sources=int(source_counts.get('flip', 0)),
            notflip_sources=int(source_counts.get('notflip', 0)),
            reportable=bool(eligible),
            macro_f1=metrics.get('macro_f1', float('nan')),
            flip_recall=metrics.get('flip_recall', float('nan')),
            false_negatives=metrics.get('fn', float('nan')),
            false_positives=metrics.get('fp', float('nan')),
        ))
    return pd.DataFrame(report)


def plot_progressive_results(overall, windows, out_dir):
    """No fabricated points: figures use only measured reportable outputs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for regime in ('T', 'ST'):
        subset = overall.loc[overall.regime == regime].sort_values('coverage_percent')
        fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
        for metric, label in [('future_macro_f1', 'Overall future Macro F1'),
                              ('matched_macro_f1', 'Matched final 10% Macro F1'),
                              ('future_flip_recall', 'Overall future flip recall')]:
            ax.plot(subset.coverage_percent, subset[metric], marker='o', label=label)
        ax.set(xlabel='Historical coverage (%)', ylabel='Score', ylim=(0, 1.05),
               title=f'{regime} — progressive training and held-out performance')
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=.2)
        fig.savefig(out_dir / f'{regime}_overall_future.png', dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
        for coverage, group in windows.loc[(windows.regime == regime)
                                           & (windows.reportable)].groupby('coverage_percent'):
            group = group.sort_values('start_horizon')
            ax.plot(group.start_horizon, group.macro_f1, marker='o',
                    label=f'{coverage}% coverage')
        ax.set(xlabel='Future-frame position (start of 3-frame window)',
               ylabel='Macro F1', ylim=(0, 1.05),
               title=f'{regime} — Macro F1 by chronological future horizon')
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
        fig.savefig(out_dir / f'{regime}_horizon_macro_f1.png', dpi=160)
        plt.close(fig)


def run_progressive_temporal(manifest, settings, device, environment_map_path):
    """Fine-tune two models progressively (T and ST), saving each frozen checkpoint."""
    root = Path(settings['out']) / 'progressive_temporal'
    root.mkdir(parents=True, exist_ok=True)
    known, unseen = environment_source_keys(manifest, environment_map_path)
    overall_records, horizon_records = [], []
    for regime in ('T', 'ST'):
        setup(settings['seed'])
        model = get_resnet(pretrained=True).to(device)
        for step, coverage in enumerate(COVERAGES):
            started = time.perf_counter()
            train, validation, future = chronological_checkpoint(
                manifest, regime, coverage, known, unseen
            )
            # The model is reused across checkpoints: progressive updating, not fresh starts.
            model, history, selected = fit_resnet_two_stage(
                model, train, validation, settings, device, resume=(step > 0)
            )
            folder = root / regime / f'{int(coverage*100):02d}pct'
            folder.mkdir(parents=True, exist_ok=True)
            history.to_csv(folder / 'training_history.csv', index=False)
            torch.save(dict(
                architecture='resnet18', state_dict=model.state_dict(),
                regime=regime, coverage=coverage,
                selected_validation=selected, training_recipe='progressive_two_stage',
                seed=settings['seed'],
            ), folder / 'checkpoint.pt')
            model.to(device)
            predictions = predict_frame_table(
                model, future, settings, device, batch_size=16, imagenet=True
            ).merge(future[['sample_id','horizon_position','frame_position_percent']],
                    on='sample_id', validate='one_to_one')
            predictions.to_csv(folder / 'future_predictions.csv', index=False)
            entire = performance(predictions)

            # Every model is tested on the identical last-10%-of-source suffix.
            # The 90% training prefix ends *before* these protected frames.
            common_suffix = predictions[predictions.frame_position_percent > 90]
            matched = performance(common_suffix)

            horizons = horizon_summary(predictions)
            horizons['coverage_percent'] = int(round(coverage * 100))
            horizons['regime'] = regime
            horizons.to_csv(folder / 'horizon_windows.csv', index=False)
            horizon_records.extend(horizons.to_dict('records'))
            overall_records.append(dict(
                regime=regime, coverage_percent=int(round(coverage * 100)),
                n_train=len(train), n_validation=len(validation),
                n_future=len(future), n_future_sources=int(future.source_key.nunique()),
                n_matched=len(common_suffix),
                future_macro_f1=entire['macro_f1'],
                future_flip_recall=entire['flip_recall'],
                future_flip_f1=2 * entire['flip_precision'] * entire['flip_recall'] /
                    max(entire['flip_precision'] + entire['flip_recall'], 1e-12),
                false_negatives=entire['fn'], false_positives=entire['fp'],
                matched_macro_f1=matched['macro_f1'],
                matched_flip_recall=matched['flip_recall'],
                checkpoint_selection=selected['selected_stage'],
                training_plus_evaluation_seconds=round(time.perf_counter()-started, 2),
            ))

    overall=pd.DataFrame(overall_records)
    windows=pd.DataFrame(horizon_records)
    overall.to_csv(root/'overall_results.csv',index=False)
    windows.to_csv(root/'horizon_results.csv',index=False)
    write_json(root/'protocol.json',dict(
        coverages=COVERAGES, horizon_width=HORIZON_WIDTH,
        min_frames_per_class=MIN_FRAMES_PER_CLASS,
        min_sources_per_class=MIN_SOURCES_PER_CLASS,
        unseen_environment='ENV-03', model_family='ImageNet ResNet18',
        progressive=True, classification_not_future_event_prediction=True,
        actual_elapsed_time_known=False, online_learning=False,
    ))
    plot_progressive_results(overall,windows,root/'figures')
    return overall,windows
