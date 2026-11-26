#!/usr/bin/env python3
"""Summarize coverage and paired, equal-prompt-weight CLAP diagnostics."""
import argparse
import json
import math
import random
import statistics
from pathlib import Path


def valid_score(row):
    value = row.get('score')
    return value if row.get('status') == 'ok' and isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and -1 <= value <= 1 else None


def index(pilot):
    result = {}
    for row in pilot['records']:
        key = (row['prompt_id'], row['seed'])
        if key in result:
            raise ValueError(f'duplicate prompt/seed: {key}')
        result[key] = row
    return result


def bootstrap(values, seed=24846, replicates=10000):
    if not values:
        return None
    rng = random.Random(seed)
    draws = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(replicates))
    return [draws[int((replicates - 1) * .025)], draws[int((replicates - 1) * .975)]]


def summarize(pilot, baseline=None):
    manifest = pilot['manifest']
    current = index(pilot)
    previous = index(baseline) if baseline else None
    provenance = None
    for dataset in (pilot, baseline):
        if dataset is None:
            continue
        for row in dataset['records']:
            if valid_score(row) is not None:
                observed = row.get('scorer_provenance')
                if not observed or (provenance is not None and provenance != observed):
                    raise ValueError('mixed or missing scorer provenance in successful cohort')
                provenance = observed
    if baseline:
        for field in ('manifest_sha256', 'family'):
            if pilot[field] != baseline[field]:
                raise ValueError(f'incompatible {field}')
        for field in ('checkpoint_identity', 'generation_settings'):
            if pilot['configuration'].get(field) != baseline['configuration'].get(field):
                raise ValueError(f'incompatible configuration {field}')
    report = {'family': pilot['family'], 'configuration': pilot['configuration'], 'subsets': {},
              'listening_review': 'pending human observations', 'quality_gate': None,
              'baseline_configuration': baseline['configuration'] if baseline else None,
              'scorer_provenance': provenance}
    for subset in ('primary', 'stress'):
        prompts = [p for p in manifest['prompts'] if p['subset'] == subset]
        details, prompt_means, deltas = [], [], []
        success = paired = 0
        for prompt in prompts:
            scores, changes, missing, missing_pairs = [], [], [], []
            for seed in manifest['seeds']:
                key = (prompt['id'], seed)
                row = current.get(key, {})
                score = valid_score(row)
                if score is None:
                    missing.append({'seed': seed, 'reason': row.get('reason') or row.get('status', 'not-run')})
                else:
                    scores.append(score)
                    success += 1
                if previous is not None:
                    old = previous.get(key, {})
                    before = valid_score(old)
                    if score is not None and before is not None:
                        if not row.get('scorer_provenance') or row['scorer_provenance'] != old.get('scorer_provenance'):
                            raise ValueError(f'incompatible scorer provenance at {key}')
                        changes.append(score - before)
                        paired += 1
                    else:
                        missing_pairs.append(seed)
            mean = statistics.mean(scores) if scores else None
            delta = statistics.mean(changes) if changes else None
            if mean is not None:
                prompt_means.append(mean)
            if delta is not None:
                deltas.append(delta)
            details.append({'prompt_id': prompt['id'], 'successful_seeds': len(scores), 'mean': mean,
                            'missing': missing, 'paired_seeds': len(changes), 'paired_delta': delta,
                            'missing_pair_seeds': missing_pairs})
        expected = len(prompts) * len(manifest['seeds'])
        report['subsets'][subset] = {'expected': expected, 'successful': success, 'partial': success != expected,
            'equal_prompt_mean': statistics.mean(prompt_means) if prompt_means else None,
            'median_prompt_mean': statistics.median(prompt_means) if prompt_means else None,
            'paired_count': paired if baseline else None,
            'paired_partial': paired != expected if baseline else None,
            'equal_prompt_paired_delta': statistics.mean(deltas) if deltas else None,
            'paired_prompt_bootstrap_95': bootstrap(deltas, **manifest['bootstrap']) if baseline else None,
            'bootstrap': manifest['bootstrap'], 'prompts': details}
    controls = []
    for control in pilot.get('controls', []):
        matched = valid_score(control.get('matched', {}))
        repeat = valid_score(control.get('repeat', {}))
        mismatch = valid_score(control.get('mismatch', {}))
        controls.append({**control,
            'repeat_minus_matched': repeat - matched if repeat is not None and matched is not None else None,
            'matched_minus_mismatch': matched - mismatch if matched is not None and mismatch is not None else None})
    report['controls'] = controls
    separations = [c['matched_minus_mismatch'] for c in controls if c['matched_minus_mismatch'] is not None]
    report['mean_matched_minus_mismatch'] = statistics.mean(separations) if separations else None
    report['validity_controls'] = pilot.get('validity_controls', 'unavailable')
    expected_keys = {(p['id'], s) for p in manifest['prompts'] for s in manifest['seeds']}
    if set(current) - expected_keys or (previous is not None and set(previous) - expected_keys):
        raise ValueError('records outside manifest')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pilot', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        report = summarize(json.loads(args.pilot.read_text()), json.loads(args.baseline.read_text()) if args.baseline else None)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
