"""Train-only PCA, pair learning and full-candidate retrieval evaluation."""

import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA

from .classifiers import fit_head, score_head

METRICS = ('top1', 'top3', 'mrr', 'two_choice')


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path, rows):
    if not rows:
        raise ValueError('No result rows to write')
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalize_rows(values):
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def aggregate(values):
    """Average per-image descriptors without creating additional identities."""
    return normalize_rows(np.asarray(values).mean(axis=1))


def photo_orders(n_sets=200, seed=20261102):
    return {role: np.array([
        np.random.default_rng(np.random.SeedSequence([seed, pair, code])).permutation(3)
        for pair in range(1, n_sets + 1)
    ]) for role, code in [('person', 1), ('dog', 2)]}


def selected_photos(orders, count, path=0):
    if count not in (1, 2, 3) or path not in (0, 1, 2):
        raise ValueError('Expected 1..3 photos and a cyclic path in 0..2')
    if count == 3:
        return np.tile(np.arange(3), (len(orders), 1))
    return np.roll(orders, -path, axis=1)[:, :count]


def make_set_features(per_image, orders, count, path=0):
    if count == 3:
        return aggregate(per_image)
    selected = selected_photos(orders, count, path)
    return aggregate(per_image[np.arange(len(per_image))[:, None], selected])


def fit_rep(human, dog, train, projection='separate', components=8):
    train = np.asarray(train, dtype=int)
    if len(train) != len(set(train)) or len(train) < components:
        raise ValueError('PCA training identities must be unique and sufficiently numerous')
    if projection == 'shared':
        h, d = normalize_rows(human[train]), normalize_rows(dog[train])
        hm, dm = h.mean(axis=0), d.mean(axis=0)
        pca = PCA(n_components=components, svd_solver='full', whiten=False).fit(
            np.vstack([h - hm, d - dm]))
        return dict(kind='shared', components=components,
                    value=dict(human_mean=hm, dog_mean=dm, pca=pca))
    if projection != 'separate':
        raise ValueError(f'Unknown PCA projection: {projection}')
    return dict(kind='separate', components=components,
        human=PCA(n_components=components, svd_solver='full', whiten=False).fit(human[train]),
        dog=PCA(n_components=components, svd_solver='full', whiten=False).fit(dog[train]))


def transform(rep, human, dog, indices):
    indices = np.asarray(indices, dtype=int)
    if rep['kind'] == 'shared':
        value = rep['value']
        h = value['pca'].transform(normalize_rows(human[indices]) - value['human_mean'])
        d = value['pca'].transform(normalize_rows(dog[indices]) - value['dog_mean'])
    else:
        h, d = rep['human'].transform(human[indices]), rep['dog'].transform(dog[indices])
    return normalize_rows(h), normalize_rows(d)


def cross_features(h, d):
    return (h[:, None, :, None] * d[None, :, None, :]).reshape(len(h) * len(d), -1)


def design_features(h, d, projection='separate'):
    if projection == 'separate':
        return cross_features(h, d)
    product, difference = h[:, None, :] * d[None, :, :], np.abs(h[:, None, :] - d[None, :, :])
    return np.concatenate([product, difference], axis=-1).reshape(len(h) * len(d), -1)


def pair_design(h, d, indices, seed, negatives=19, projection='separate'):
    n = len(h)
    if len(d) != n or len(indices) != n or n < 2:
        raise ValueError('Aligned training sets with at least two identities are required')
    positive = np.eye(n, dtype=bool)
    mask = positive.copy()
    count = n - 1 if negatives is None else min(int(negatives), n - 1)
    if count < 1:
        raise ValueError('At least one negative per training identity is required')
    for i, index in enumerate(indices):
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(index), 19]))
        mask[i, rng.choice(np.delete(np.arange(n), i), count, replace=False)] = True
    positions = np.flatnonzero(mask.ravel())
    y = positive.ravel()[positions].astype(int)
    weights = np.where(y == 1, .5 / n, .5 / (n * count))
    return dict(x=design_features(h, d, projection)[positions], y=y, weights=weights,
        positions=positions, positions_sha256=hashlib.sha256(positions.tobytes()).hexdigest())


def validation_features(h, d, indices, galleries, projection='separate'):
    lookup = {int(value): i for i, value in enumerate(indices)}
    positions = [i * len(d) + lookup[int(value)] for i, row in enumerate(galleries) for value in row]
    return design_features(h, d, projection)[positions]


def retrieval_metrics(values):
    """Truth is first; tied ranks receive uniform-order expected credit."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Expected finite scores for truth plus at least one distractor')
    delta = values - values[0]
    tied = np.abs(delta) <= 1e-12
    better, ties = int(((delta > 0) & ~tied).sum()), int(tied.sum())
    negatives = delta[1:]
    return dict(top1=1 / ties if better == 0 else 0.,
        top3=min(max(3 - better, 0), ties) / ties,
        mrr=float(np.mean(1 / np.arange(better + 1, better + ties + 1))),
        two_choice=float(np.mean((negatives < -1e-12) + .5 * (np.abs(negatives) <= 1e-12))),
        chance=1 / len(values), gallery_size=len(values), chance_top3=min(3, len(values)) / len(values))


def selection_metrics(scores, gallery_size=10):
    measured = [retrieval_metrics(row) for row in np.asarray(scores).reshape(-1, gallery_size)]
    return {metric: float(np.mean([row[metric] for row in measured])) for metric in ('top1', 'top3')}


def predict(model, human, dog, test):
    h, d = transform(model['representation'], human, dog, test)
    x = design_features(h, d, model['representation']['kind'])
    return score_head(model['head'], x).reshape(len(h), len(d))


def qualified_name(case, name):
    prefix = case.get('record_prefix')
    if prefix is not None:
        return name if not prefix else f'{prefix}__{name}'
    return name if case['feature_group'] == 'enhanced' else f'{case["feature_group"]}__{name}'


def query_rows(matrix, split, model, case, policy):
    test = split['test']
    lookup = {int(value): i for i, value in enumerate(test)}
    if set(split['galleries']) != {'mixed10'}:
        raise ValueError('Only full-pool mixed10 galleries are accepted')
    rows = []
    for i, candidates in enumerate(split['galleries']['mixed10']):
        if candidates[0] != test[i] or len(set(candidates)) != 10:
            raise ValueError('Expected ten distinct candidates with the actual dog first')
        values = matrix[i, [lookup[int(value)] for value in candidates]]
        rows.append(dict(model=model, protocol='mixed10', seed=split['seed'], fold=split['fold'],
            pair_id=int(test[i]) + 1, candidate_pair_ids=' '.join(str(value + 1) for value in candidates),
            **retrieval_metrics(values), train_policy=policy, photos=case['photos'], path=case['path'],
            case_id=case['id'], feature_group=case['feature_group']))
    return rows


def _validate_plan(plan):
    for split in plan['outer']:
        train, test = set(split['train']), set(split['test'])
        if (len(train) != len(split['train']) or len(test) != len(split['test'])
                or train & test or set(split['galleries']) != {'mixed10'}):
            raise ValueError('Invalid outer training/evaluation boundary or gallery policy')
        if len(split['galleries']['mixed10']) != len(split['test']):
            raise ValueError('One full-pool gallery is required per outer query')
        for query, gallery in zip(split['test'], split['galleries']['mixed10']):
            if gallery[0] != query or len(set(gallery)) != 10 or not set(gallery) <= test:
                raise ValueError('Outer gallery contains an invalid or non-held-out candidate')
        tag = f"s{split['seed']}_f{split['fold']}"
        for block in plan['inner'][tag]:
            fit, validation = set(block['train']), set(block['validation'])
            if fit & validation or fit | validation != train or set(block['galleries']) != {'mixed10'}:
                raise ValueError('Inner split must partition only the outer training identities')
            if len(block['galleries']['mixed10']) != len(block['validation']):
                raise ValueError('One full-pool gallery is required per inner query')
            for query, gallery in zip(block['validation'], block['galleries']['mixed10']):
                if gallery[0] != query or len(set(gallery)) != 10 or not set(gallery) <= validation:
                    raise ValueError('Inner gallery contains an invalid candidate')


def run_experiments(features, plan, config, output, shard=0, shards=1):
    """Fit one shard. The caller supplies only features, frozen plans and settings."""
    _validate_plan(plan)
    if not 0 <= shard < shards:
        raise ValueError('Invalid shard number')
    output = Path(output)
    (output / 'jobs').mkdir(parents=True, exist_ok=True)
    components, pair_seed = int(config.get('components', 8)), int(config.get('pair_seed', 20261102))
    completed, start = [], time.monotonic()
    for index, split in enumerate(plan['outer']):
        if index % shards != shard:
            continue
        tag = f"s{split['seed']}_f{split['fold']}"
        job = output / 'jobs' / tag
        if job.exists():
            raise FileExistsError(f'Preserve previous or partial job: {job}')
        job.mkdir()
        train, test = np.array(split['train']), np.array(split['test'])
        three_models = {}
        model_count = 0
        for case in config['cases']:
            tick = time.monotonic()
            target = job / case['id']
            target.mkdir()
            human, dog = features[f'{case["id"]}__person'], features[f'{case["id"]}__dog']
            projection = case.get('projection', 'separate')
            rep = fit_rep(human, dog, train, projection, components)
            ht, dt = transform(rep, human, dog, train)
            sampled = pair_design(ht, dt, train, pair_seed, config.get('negatives_per_query', 19), projection)
            inner_data = []
            for block in plan['inner'][tag]:
                fit, validation = np.array(block['train']), np.array(block['validation'])
                inner_rep = fit_rep(human, dog, fit, projection, components)
                hf, df = transform(inner_rep, human, dog, fit)
                hv, dv = transform(inner_rep, human, dog, validation)
                inner_data.append(dict(design=pair_design(hf, df, fit, pair_seed,
                    config.get('negatives_per_query', 19), projection), validation_features=validation_features(
                    hv, dv, validation, block['galleries']['mixed10'], projection)))
            models, traces, selections = {}, [], []
            for name in case['heads']:
                if name == 'logistic_allneg_fixed':
                    design = pair_design(ht, dt, train, pair_seed, None, projection)
                    params = {'lambda': float(config.get('fixed_logistic_lambda', .01))}
                    fitted = fit_head(design['x'], design['y'], design['weights'], 'logistic', params, split['seed'])
                    selection = dict(model=name, params=params, selected_candidate=-1)
                else:
                    candidates = []
                    for candidate_id, params in enumerate(config['candidate_grid'][name]):
                        measurements = []
                        for inner_id, block in enumerate(inner_data):
                            design = block['design']
                            head = fit_head(design['x'], design['y'], design['weights'], name, params,
                                split['seed'] + split['fold'] * 10 + inner_id)
                            measurements.append(selection_metrics(score_head(head, block['validation_features'])))
                        candidates.append(dict(model=name, candidate=candidate_id, params=params,
                            inner_top1=float(np.mean([r['top1'] for r in measurements])),
                            inner_top3=float(np.mean([r['top3'] for r in measurements]))))
                    traces.extend(candidates)
                    best = max(candidates, key=lambda r: (round(r['inner_top1'], 12), round(r['inner_top3'], 12), -r['candidate']))
                    selection = dict(model=name, params=best['params'], selected_candidate=best['candidate'],
                        inner_top1=best['inner_top1'], inner_top3=best['inner_top3'], selection_protocol='mixed10')
                    design = sampled
                    fitted = fit_head(design['x'], design['y'], design['weights'], name, best['params'], split['seed'] + split['fold'])
                model = dict(name=name, case=case, head=fitted, representation=rep, selection=selection,
                    train_indices=train, test_indices=test, pair_positions_sha256=design['positions_sha256'])
                if config.get('write_models', True):
                    joblib.dump(model, target / f'{name}.joblib', compress=3)
                models[name], model_count = model, model_count + 1
                selections.append(selection)
            matrices, queries = {}, []
            for name, model in models.items():
                record = qualified_name(case, name)
                matrix = predict(model, human, dog, test)
                matrices[f'matched_k__{record}'] = matrix
                queries += query_rows(matrix, split, record, case, 'matched_k')
                if case.get('auxiliary', False):
                    if case['photos'] < 3 and name not in three_models:
                        raise ValueError('Run the three-photo enhanced case before auxiliary one/two-photo cases')
                    fixed = model if case['photos'] == 3 else three_models[name]
                    deployment = matrix if case['photos'] == 3 else predict(fixed, human, dog, test)
                    matrices[f'trained3_fixed__{record}'] = deployment
                    queries += query_rows(deployment, split, record, case, 'trained3_fixed')
            if case['feature_group'] == 'enhanced' and case['photos'] == 3:
                three_models = models
            np.savez_compressed(target / 'scores.npz', **matrices)
            write_csv(target / 'query_metrics.csv', queries)
            write_json(target / 'selected.json', selections)
            write_json(target / 'inner_selection_trace.json', traces)
            write_json(target / 'training_design.json', dict(indices=train.tolist(),
                sampled_positions_sha256=sampled['positions_sha256'], sampled_rows=len(sampled['y']),
                positives=int(sampled['y'].sum()), negative_count_per_query=min(config.get('negatives_per_query', 19), len(train)-1)))
            print(f'DONE {tag} {case["id"]} seconds={time.monotonic()-tick:.1f}', flush=True)
        write_json(job / 'complete.json', dict(status='PASS', cases=len(config['cases']), models=model_count))
        completed.append(tag)
    write_json(output / f'shard_{shard}_complete.json', dict(status='PASS', jobs=completed, elapsed_seconds=time.monotonic()-start))
    return completed


def summarise(rows):
    groups, per_set, summary, repeats = defaultdict(list), [], [], []
    for row in rows:
        key = (row['model'], row['train_policy'], int(row['photos']), row['protocol'], int(row['seed']), int(row['pair_id']))
        groups[key].append(row)
    for key, current in groups.items():
        name, policy, photos, protocol, seed, pair = key
        if len(current) != len({int(row['path']) for row in current}):
            raise ValueError('Duplicate photo path for the same evaluated identity')
        per_set.append(dict(model=name, train_policy=policy, photos=photos, protocol=protocol, seed=seed,
            pair_id=pair, photo_paths=len(current),
            **{metric: float(np.mean([float(r[metric]) for r in current])) for metric in METRICS}))
    for name, policy, photos, protocol in sorted({key[:4] for key in groups}):
        selected = [r for r in per_set if (r['model'], r['train_policy'], r['photos'], r['protocol']) == (name, policy, photos, protocol)]
        current_repeats = []
        for seed in sorted({r['seed'] for r in selected}):
            current = [r for r in selected if r['seed'] == seed]
            repeat = dict(model=name, train_policy=policy, photos=photos, protocol=protocol, seed=seed,
                sets=len(current), **{metric: float(np.mean([r[metric] for r in current])) for metric in METRICS})
            repeats.append(repeat)
            current_repeats.append(repeat)
        summary.append(dict(model=name, train_policy=policy, photos=photos, protocol=protocol,
            sets=len({r['pair_id'] for r in selected}), repeats=len(current_repeats),
            **{metric: float(np.mean([r[metric] for r in current_repeats])) for metric in METRICS},
            **{f'{metric}_repeat_{label}': fn(r[metric] for r in current_repeats)
               for metric in ('top1', 'top3') for label, fn in [('min', min), ('max', max)]}))
    return summary, repeats, per_set


def collect_results(output, plan, config):
    """Aggregate completed jobs, averaging photo paths before sets and repeats."""
    output = Path(output)
    if (output / 'summary.json').exists():
        raise FileExistsError('Preserve previous summaries')
    rows, model_count = [], 0
    for split in plan['outer']:
        tag = f"s{split['seed']}_f{split['fold']}"
        complete = json.loads((output / 'jobs' / tag / 'complete.json').read_text(encoding='utf-8'))
        if complete['status'] != 'PASS':
            raise ValueError(f'Incomplete job: {tag}')
        model_count += complete['models']
        for case in config['cases']:
            with (output / 'jobs' / tag / case['id'] / 'query_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
                rows.extend(csv.DictReader(stream))
    expected = sum(len(split['test']) * sum(len(case['heads']) * (2 if case.get('auxiliary', False) else 1)
        for case in config['cases']) for split in plan['outer'])
    if len(rows) != expected or {row['protocol'] for row in rows} != {'mixed10'}:
        raise ValueError('Result coverage differs from the selected plan/configuration')
    summary, repeats, per_set = summarise(rows)
    for name, current in [('query_metrics.csv', rows), ('per_set_path_average.csv', per_set),
                          ('repeat_metrics.csv', repeats), ('summary.csv', summary)]:
        write_csv(output / name, current)
    write_json(output / 'summary.json', summary)
    write_json(output / 'run_complete.json', dict(status='PASS', outer_jobs=len(plan['outer']),
        outer_fitted_models=model_count, query_rows=len(rows), summary_rows=len(summary),
        inner_selection_protocol='mixed10', outer_evaluation_protocol='mixed10',
        deep_learning_used=False, reinforcement_learning_used=False))
    return summary
