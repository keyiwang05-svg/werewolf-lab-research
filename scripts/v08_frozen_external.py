"""Werewolf Lab v0.8: audit-then-score externally generated public decision samples.

Modes are designed for separate environments:
  tfidf   : .venv-sim (scikit-learn==1.7.2)
  minilm  : pinned local SentenceTransformer + sklearn==1.7.2
  rbt3    : .venv-transformer (the same torch/transformers used for v0.5)
  finalize: .venv-sim (only evaluates saved frozen predictions)

No fit(), optimizer.step(), or remote LLM calls.  All saved model loading is
restricted to trusted locally generated files. Not a human-outcome benchmark.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import sys
import tempfile
import shutil
from collections import defaultdict, Counter
from pathlib import Path

FAMILIES = ("random", "rule", "stealth", "skeptical", "stochastic", "deceptive")
MODES = ("raw", "masked")
METHODS = ("tfidf", "minilm", "rbt3")
VERSION_SK = "1.7.2"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding='utf-8') as f:
        out = [json.loads(line) for line in f if line.strip()]
    if not out:
        raise ValueError(f'Empty file: {path}')
    return out


def load_scrub(root: Path):
    """Compile ONLY the existing canonical scrub() function without game imports."""
    path = root / 'src/llm_werewolf/generalization.py'
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'scrub']
    if len(matches) != 1 or matches[0].decorator_list:
        raise ValueError('Expected exactly one undecorated v0.4 scrub()')
    node = matches[0]
    if node.args.args[0].arg != 'text':
        raise ValueError('scrub(text) signature has changed')
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    scope = {'re': re}
    exec(compile(module, str(path), 'exec'), scope)
    fn = scope['scrub']
    text = '我自称预言家，查验player_3为狼人。'
    if not isinstance(fn(text), str) or fn(text) == text:
        raise ValueError('Canonical scrub sanity check failed')
    return fn, digest(path)


def key(r: dict):
    return (r['game'], r['round'], r['player'], r['label'])


def strict_rows(rows: list[dict], *, external: bool):
    seen = set()
    for r in rows:
        k = key(r)
        if k in seen:
            raise ValueError(f'Duplicate decision-time sample: {k}')
        seen.add(k)
        if r['label'] not in (0, 1) or not isinstance(r['text'], str):
            raise ValueError(f'Invalid label or text: {k}')
        if not isinstance(r.get('numeric'), list):
            raise ValueError(f'Numeric features missing: {k}')
        if external:
            if not 1100 <= r['seed'] <= 1199 or r['family'] != 'independent_policy_v1':
                raise ValueError(f'Unexpected external seed/family: {k}')
    return seen


def context(root: Path):
    ext = root / 'data/v08-independent-100-v1'
    summary = load_json(ext / 'summary.json')
    external = read_jsonl(ext / 'dataset.jsonl')
    if (summary['games_requested'], summary['games_completed'], summary['games_failed'],
        summary['seed_start'], summary['seed_end'], summary['model_api_calls'],
        summary['train_new_model']) != (100, 100, 0, 1100, 1199, 0, False):
        raise ValueError('External dataset manifest not the frozen successful 100-game study')
    if summary['generator'] != 'independent_policy_v1' or len(external) != summary['samples']:
        raise ValueError('External dataset summary mismatch')
    strict_rows(external, external=True)
    game_ids = {r['game'] for r in external}
    if len(game_ids) != 100 or {r['seed'] for r in external} != set(range(1100, 1200)):
        raise ValueError('Expected 100 unique games with seeds 1100–1199')
    old_data = root / 'data/v04-generalization'
    old = read_jsonl(old_data / 'dataset.jsonl')
    strict_rows(old, external=False)
    splits = load_json(old_data / 'split.json')
    train, val, test = (set(splits[k]) for k in ('train', 'validation', 'test'))
    if (train & val) or (train & test) or (val & test) or len(test) != 40:
        raise ValueError('Original train/validation/test seed groups invalid')
    old_tests = {fam: [r for r in old if r['family'] == fam and r['seed'] in test] for fam in FAMILIES}
    for fam, rows in old_tests.items():
        if {r['seed'] for r in rows} != test or len({r['game'] for r in rows}) != 40:
            raise ValueError(f'Original held-out family test invalid: {fam}')
    return external, old_tests, {
        'external_dataset_sha256': digest(ext / 'dataset.jsonl'),
        'external_summary_sha256': digest(ext / 'summary.json'),
        'original_dataset_sha256': digest(old_data / 'dataset.jsonl'),
        'original_split_sha256': digest(old_data / 'split.json'),
        'policy_sha256': summary['policy_sha256'],
    }


def reference_predictions(root: Path, method: str):
    if method == 'tfidf':
        rows = read_jsonl(root / 'data/v04-generalization/predictions.jsonl')
        wanted = {'text': 'raw', 'scrubbed_text': 'masked'}
        selected = [r for r in rows if r.get('model') == 'logistic' and r.get('features') in wanted]
    elif method == 'minilm':
        rows = read_jsonl(root / 'data/v05-sentence-baseline/predictions.jsonl')
        wanted = {'sentence_text': 'raw', 'scrubbed_sentence_text': 'masked'}
        selected = [r for r in rows if r.get('model') == 'logistic' and r.get('features') in wanted]
    else:
        raise ValueError('RBT3 references are stored per fold')
    result = defaultdict(dict)
    for r in selected:
        index = (r['family'], wanted[r['features']])
        k = key(r)
        if k in result[index]:
            raise ValueError(f'Duplicate old prediction {index}: {k}')
        result[index][k] = float(r['probability'])
    for fam in FAMILIES:
        for mode in MODES:
            if (fam, mode) not in result:
                raise ValueError(f'Missing old {method} reference: {fam}/{mode}')
    return result


def verify_replay(rows: list[dict], preds, reference: dict, tolerance: float, label: str):
    import numpy as np
    if len(preds) != len(rows):
        raise ValueError(f'Length mismatch: {label}')
    indexes = [key(r) for r in rows]
    if set(indexes) != set(reference):
        missing = set(reference) - set(indexes)
        extra = set(indexes) - set(reference)
        raise ValueError(f'Old test sample keys changed ({label}); missing={len(missing)}, extra={len(extra)}')
    differences = np.abs(np.asarray(preds, dtype=float) - np.asarray([reference[k] for k in indexes]))
    if not np.isfinite(differences).all():
        raise ValueError('Non-finite replay scores')
    worst = float(differences.max())
    print(f'  REPLAY {label}: max probability error = {worst:.8g} (threshold {tolerance})', flush=True)
    if worst > tolerance:
        raise ValueError(f'Old prediction REPLAY FAILED for {label}: {worst} > {tolerance}')
    return worst


def check_sk_version():
    import sklearn
    if sklearn.__version__ != VERSION_SK:
        raise RuntimeError(f'Scikit-learn mismatch: need {VERSION_SK}, got {sklearn.__version__}. Use .venv-sim or a separate pinned env.')


def score_tfidf(root, external, old_tests, scrub):
    check_sk_version()
    import joblib
    values = []
    replay = []
    refs = reference_predictions(root, 'tfidf')
    base = root / 'data/v04-generalization/models'
    for fam in FAMILIES:
        for mode in MODES:
            variant = 'text' if mode == 'raw' else 'scrubbed_text'
            file = base / f'{fam}-{variant}-logistic.joblib'
            if not file.is_file():
                raise FileNotFoundError(file)
            obj = joblib.load(file)  # TRUSTED, locally generated artifacts ONLY
            if obj['variant'] != variant or obj['svd'] is not None:
                raise ValueError(f'Unexpected saved model contents: {file}')
            texts_old = [r['text'] if mode == 'raw' else scrub(r['text']) for r in old_tests[fam]]
            texts_ext = [r['text'] if mode == 'raw' else scrub(r['text']) for r in external]
            old_proba = obj['model'].predict_proba(obj['tfidf'].transform(texts_old))[:, 1]
            err = verify_replay(old_tests[fam], old_proba, refs[(fam, mode)], 1e-6, f'tfidf/{fam}/{mode}')
            new_proba = obj['model'].predict_proba(obj['tfidf'].transform(texts_ext))[:, 1]
            values.extend(prediction_records(external, fam, mode, 'tfidf', new_proba))
            replay.append({'method':'tfidf', 'trained_fold':fam, 'condition':mode, 'max_abs_replay_diff':err, 'model_sha256':digest(file)})
    return values, replay


def prediction_records(rows, fam, mode, method, probs):
    import math
    if len(rows) != len(probs):
        raise ValueError(f'Wrong score count: {method}/{fam}/{mode}')
    ans = []
    for row, p in zip(rows, probs):
        p = float(p)
        if not math.isfinite(p) or p < 0 or p > 1:
            raise ValueError(f'Invalid probability {p} in {method}/{fam}/{mode}')
        ans.append({
            **{k:row[k] for k in ('game','seed','family','player','round','cutoff','label')},
            'trained_fold':fam, 'method':method, 'condition':mode, 'p_werewolf':p,
        })
    return ans


def score_rbt3(root, external, old_tests, scrub, source_sha, batch_size):
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    torch.manual_seed(42)
    all_results, replay = [], []
    for fam in FAMILIES:
        for mode in MODES:
            run = root/'data/v05-transformer'/f'{"raw" if mode=="raw" else "masked"}-{fam}-full-v1'
            meta_file, sum_file, cp = run/'config.json', run/'training_summary.json', run/'best_model'
            if not meta_file.is_file() or not sum_file.is_file() or not (cp/'config.json').is_file():
                raise FileNotFoundError(f'Missing full RBT3 run: {run}')
            meta = load_json(meta_file)
            cfg, data = meta['config'], meta['data']
            if cfg['held_out'] != fam or cfg['max_train'] != 0 or cfg['max_val'] != 0:
                raise ValueError('Refusing non-full-data RBT3 checkpoint')
            if data['dataset_sha256'] != digest(root/'data/v04-generalization/dataset.jsonl') or data['split_sha256'] != digest(root/'data/v04-generalization/split.json'):
                raise ValueError('RBT3 training-data provenance mismatch')
            if mode == 'masked':
                if cfg.get('text_condition') != 'scrubbed_text' or cfg.get('scrub_source_sha256') != source_sha:
                    raise ValueError('RBT3 masked model was not trained with current canonical scrub()')
            elif cfg.get('text_condition') == 'scrubbed_text':
                raise ValueError('Expected RAW RBT3 checkpoint')
            if not (load_json(sum_file).get('test_evaluated') is False):
                raise ValueError('Unexpected training record')
            ref = read_jsonl(root/'data/v05-transformer'/f'eval-{mode}-{fam}-v1'/'predictions.jsonl')
            reference = {}
            for r in ref:
                k = key(r)
                if k in reference:
                    raise ValueError(f'Duplicate RBT3 reference: {k}')
                reference[k] = r['p_werewolf']
            tokenizer = AutoTokenizer.from_pretrained(str(cp), local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(str(cp), local_files_only=True).to(device)
            model.eval()
            length = int(cfg['max_length'])
            def infer(rows):
                scores = []
                with torch.no_grad():
                    for begin in range(0, len(rows), batch_size):
                        block = rows[begin:begin+batch_size]
                        inputs = [r['text'] if mode == 'raw' else scrub(r['text']) for r in block]
                        enc = tokenizer(inputs, padding=True, truncation=True,
                                        max_length=length, return_tensors='pt')
                        enc = {k:v.to(device) for k,v in enc.items()}
                        prob = torch.softmax(model(**enc).logits, dim=-1)[:, 1]
                        scores.extend(prob.cpu().tolist())
                return scores
            old_scores = infer(old_tests[fam])
            err = verify_replay(old_tests[fam], old_scores, reference, 5e-4, f'rbt3/{fam}/{mode}')
            new_scores = infer(external)
            all_results.extend(prediction_records(external, fam, mode, 'rbt3', new_scores))
            replay.append({'method':'rbt3','trained_fold':fam,'condition':mode,'max_abs_replay_diff':err,
                           'model_sha256':digest(cp/'model.safetensors'),'max_length':length,'device':device})
            del model
            if device == 'mps':
                torch.mps.empty_cache()
    return all_results, replay


def score_minilm(root, external, old_tests, scrub, encoder_path, batch_size):
    check_sk_version()
    import numpy as np
    import joblib
    import torch
    from sentence_transformers import SentenceTransformer
    encoder_path = Path(encoder_path).expanduser().resolve()
    if not (encoder_path/'model.safetensors').is_file():
        raise FileNotFoundError(f'Pinned local encoder not found at {encoder_path} (model.safetensors missing)')
    protocol = load_json(root/'data/v05-sentence-baseline/protocol.json')
    for name, expected in protocol.get('model_files_sha256', {}).items():
        path = encoder_path/name
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'MiniLM local snapshot hash mismatch/missing: {name}')
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    encoder = SentenceTransformer(str(encoder_path), device='cpu', local_files_only=True,
                                  trust_remote_code=False, model_kwargs={'use_safetensors':True})
    encoder.eval()
    for par in encoder.parameters():
        par.requires_grad_(False)
    if encoder.max_seq_length != 128 or encoder.get_sentence_embedding_dimension() != 384:
        raise ValueError('Unexpected MiniLM architecture; expected 128-token 384-dimension encoder')
    def encode(texts):
        # Original v0.5 sentence_baseline.py uses np.unique and inverse indices.
        unique, inverse = np.unique(np.asarray(texts, dtype=str), return_inverse=True)
        emb = np.asarray(encoder.encode(unique.tolist(), batch_size=batch_size,
                                       normalize_embeddings=True, convert_to_numpy=True,
                                       show_progress_bar=False), dtype=np.float32)
        if emb.ndim != 2 or emb.shape[1] != 384 or not np.isfinite(emb).all():
            raise ValueError('Invalid MiniLM embeddings')
        return emb[inverse]
    refs = reference_predictions(root, 'minilm')
    predictions, replay = [], []
    models_dir = root/'data/v05-sentence-baseline/models'
    # Encode external and original test once for each condition; all families reuse it.
    old_union = [r for fam in FAMILIES for r in old_tests[fam]]
    embeddings = {}
    for mode in MODES:
        filt = lambda rows:[r['text'] if mode=='raw' else scrub(r['text']) for r in rows]
        embeddings[(mode,'external')] = encode(filt(external))
        embeddings[(mode,'old')] = encode(filt(old_union))
    slices = {};offset = 0
    for fam in FAMILIES:
        slices[fam] = slice(offset, offset+len(old_tests[fam]))
        offset += len(old_tests[fam])
    for fam in FAMILIES:
        for mode in MODES:
            variant = 'sentence_text' if mode=='raw' else 'scrubbed_sentence_text'
            file = models_dir/f'{fam}-{variant}.joblib'
            if not file.is_file():
                raise FileNotFoundError(file)
            obj = joblib.load(file)  # TRUSTED locally generated files ONLY
            if obj['variant'] != variant or not obj['normalize_embeddings']:
                raise ValueError('Saved MiniLM model settings differ from training')
            if obj['encoder_id'] != protocol['model_id'] or obj['encoder_revision'] != protocol['model_revision']:
                raise ValueError('MiniLM encoder identity/revision mismatch')
            old_scores = obj['classifier'].predict_proba(embeddings[(mode,'old')][slices[fam]])[:, 1]
            err = verify_replay(old_tests[fam], old_scores, refs[(fam,mode)], 5e-4, f'minilm/{fam}/{mode}')
            new_scores = obj['classifier'].predict_proba(embeddings[(mode,'external')])[:, 1]
            predictions.extend(prediction_records(external, fam, mode, 'minilm', new_scores))
            replay.append({'method':'minilm','trained_fold':fam,'condition':mode,
                           'max_abs_replay_diff':err,'classifier_sha256':digest(file),
                           'encoder_revision':obj['encoder_revision']})
    return predictions, replay


def write_mode(root: Path, output: Path, mode: str, external: list[dict], meta: dict, rows, replay):
    if output.exists():
        raise FileExistsError(f'Will not overwrite: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f'.partial_{mode}_', dir=output.parent))
    try:
        with (tmp/'predictions.jsonl').open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False)+'\n')
        (tmp/'replay_audit.json').write_text(json.dumps(replay, ensure_ascii=False, indent=2))
        (tmp/'manifest.json').write_text(json.dumps({
            'method':mode, 'external_samples':len(external), 'games':len({r['game'] for r in external}),
            'conditions':len(replay), **meta,
            'note':'Frozen local models only; original v0.4/v0.5 held-out predictions reproduced first. No model fit or selection.'
        }, ensure_ascii=False, indent=2))
        tmp.rename(output)
    except Exception:
        shutil.rmtree(tmp)
        raise
    print(f'FINISHED {mode.upper()}: {len(replay)} original-fold conditions replayed; {len(rows)} external predictions saved -> {output}')


def calc_metrics(y, p):
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    import numpy as np
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if set(y.tolist()) != {0,1} or not np.isfinite(p).all() or min(p)<0 or max(p)>1:
        raise ValueError('Invalid probabilities or single-class labels')
    return {'roc_auc':float(roc_auc_score(y,p)),
            'average_precision':float(average_precision_score(y,p)),
            'brier':float(brier_score_loss(y,p))}


def evaluate_all(root, output:Path, n_boot:int):
    import numpy as np
    external, _, hashes = context(root)
    paths = {method:root/'data/v08-external-evaluation'/method/'predictions.jsonl' for method in METHODS}
    collected = defaultdict(dict)
    ext_keys = {key(r) for r in external}
    for method, path in paths.items():
        rows = read_jsonl(path)
        meta = load_json(path.parent/'manifest.json')
        if meta['external_dataset_sha256'] != hashes['external_dataset_sha256'] or meta['original_dataset_sha256'] != hashes['original_dataset_sha256']:
            raise ValueError(f'{method} was scored against a DIFFERENT dataset')
        for row in rows:
            if row['method'] != method or row['trained_fold'] not in FAMILIES or row['condition'] not in MODES:
                raise ValueError(f'Unexpected method/fold/condition in {path}')
            tag = (method, row['trained_fold'], row['condition'])
            k = key(row)
            if k not in ext_keys or k in collected[tag]:
                raise ValueError(f'Misaligned/duplicate external sample: {tag}/{k}')
            collected[tag][k] = float(row['p_werewolf'])
    wanted = {(method, fam, mode) for method in METHODS for fam in FAMILIES for mode in MODES}
    if set(collected) != wanted:
        raise ValueError(f'Expected 36 frozen-model conditions, got {len(collected)}')
    for tag, values in collected.items():
        if set(values) != ext_keys:
            raise ValueError(f'Incomplete {tag}: {len(values)} vs {len(ext_keys)}')
    order = [key(r) for r in external]
    y = np.asarray([r['label'] for r in external], dtype=int)
    scores = {tag:np.asarray([collected[tag][k] for k in order]) for tag in wanted}
    games = defaultdict(list)
    for i, row in enumerate(external):
        games[row['game']].append(i)
    group_ids = sorted(games)
    rng = np.random.default_rng(20261115)
    draws = []
    for _ in range(n_boot):
        ids = rng.choice(group_ids, size=len(group_ids), replace=True)
        ind = np.concatenate([games[g] for g in ids])
        if len(set(y[ind])) < 2:
            continue
        draws.append(ind)
    if len(draws) < min(30, n_boot):
        raise ValueError('Too few valid game bootstrap draws')
    def CI(vals):
        return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]
    results = []
    boots_cache = {}
    for tag in sorted(wanted):
        val = calc_metrics(y,scores[tag])
        boots = {m:[] for m in val}
        for ids in draws:
            stats = calc_metrics(y[ids],scores[tag][ids])
            for m,v in stats.items():boots[m].append(v)
        boots_cache[tag] = boots
        results.append({'method':tag[0],'trained_fold':tag[1],'condition':tag[2],
                        'samples':len(y),'games':len(group_ids), 'positive_prevalence':round(float(y.mean()),6),
                        **val,**{f'{m}_ci_low':CI(boots[m])[0] for m in val},
                        **{f'{m}_ci_high':CI(boots[m])[1] for m in val}})
    pairs=[]
    for fam in FAMILIES:
        for mode in MODES:
            for method_a,method_b in [('tfidf','rbt3'),('minilm','rbt3'),('tfidf','minilm')]:
                pairs.append(((method_a,fam,mode),(method_b,fam,mode), 'model'))
        for method in METHODS:
            pairs.append(((method,fam,'raw'),(method,fam,'masked'),'masking'))
    assert len(pairs) == 54
    differences = []
    for a,b,kind in pairs:
        ma, mb = calc_metrics(y,scores[a]), calc_metrics(y,scores[b])
        row = {'contrast':kind,'A':':'.join(a),'B':':'.join(b), 'delta_definition':'B minus A'}
        for metric in ('roc_auc','average_precision','brier'):
            diff = np.asarray(boots_cache[b][metric])-np.asarray(boots_cache[a][metric])
            row.update({f'{metric}_delta':mb[metric]-ma[metric],f'{metric}_delta_ci_low':CI(diff)[0],f'{metric}_delta_ci_high':CI(diff)[1]})
        differences.append(row)
    summary=[]
    for method in METHODS:
        for mode in MODES:
            group = [r for r in results if r['method']==method and r['condition']==mode]
            summary.append({'method':method,'condition':mode,
                            'auc_median_across_folds':float(np.median([r['roc_auc'] for r in group])),
                            'auc_min_across_folds':min(r['roc_auc'] for r in group),
                            'auc_max_across_folds':max(r['roc_auc'] for r in group),
                            'note':'Six correlated old-training folds, evaluated on same 100 external games; NOT independent replications'})
    if output.exists():
        raise FileExistsError(f'Will not overwrite: {output}')
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix='.partial_finalize_',dir=output.parent))
    try:
        for file, records in [('external_36_metrics.csv',results),('external_54_paired.csv',differences),('fold_summary.csv',summary)]:
            with (tmp/file).open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=records[0].keys())
                writer.writeheader();writer.writerows(records)
        (tmp/'manifest.json').write_text(json.dumps({**hashes, 'conditions':36,'comparisons':54,
           'n_test_samples':len(external),'n_test_games':len(group_ids),
           'bootstrap_requested':n_boot,'bootstrap_valid':len(draws),'bootstrap_unit':'whole_game',
           'versions':{'scikit_learn':__import__('sklearn').__version__},
           'limitations':['One independently implemented policy, same game engine and feature extractor',
             'No human data; no second independent strategy source',
             'Six old held-out training folds share many training games; not six independent replications',
             'Frozen-model test; game bootstrap excludes retraining and policy-source uncertainty',
             'No model chosen on external data; all 36 conditions reported',
             'Previous development inspected held-out v0.4/v0.5 test data repeatedly']},indent=2))
        tmp.rename(output)
    except Exception:
        shutil.rmtree(tmp)
        raise
    print(f'FINAL VERIFIED: 36 external metrics + 54 paired comparisons; {len(group_ids)} games / {len(y)} samples.')
    print('Files:',output)
    print('External AUC medians across SIX CORRELATED training folds:')
    for row in summary:
        print(f"  {row['method']:<7} {row['condition']:<6} {row['auc_median_across_folds']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('tfidf','minilm','rbt3','finalize'))
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--encoder-path',type=Path,help='Pinned local SentenceTransformer snapshot directory (minilm only)')
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--bootstrap',type=int,default=500)
    args=parser.parse_args()
    root=args.project.expanduser().resolve()
    if args.batch_size < 1 or args.bootstrap < 30:
        parser.error('batch-size >=1 and bootstrap >=30 required')
    if args.mode=='finalize':
        evaluate_all(root,root/'data/v08-external-evaluation/final-v1',args.bootstrap)
        return
    out=root/'data/v08-external-evaluation'/args.mode
    if out.exists():
        raise FileExistsError(f'Existing output, refusing overwrite: {out}')
    external, old_tests, hashes=context(root)
    scrub, scrub_sha = load_scrub(root)
    hashes['scrub_source_sha256']=scrub_sha
    print(f'External audit: {len(external)} rows / 100 games. Original held-out replay validation required.', flush=True)
    if args.mode=='tfidf':
        rows,replay=score_tfidf(root,external,old_tests,scrub)
    elif args.mode=='rbt3':
        rows,replay=score_rbt3(root,external,old_tests,scrub,scrub_sha,args.batch_size)
    else:
        if args.encoder_path is None:
            parser.error('minilm requires --encoder-path to the exact pinned local snapshot folder')
        rows,replay=score_minilm(root,external,old_tests,scrub,args.encoder_path,args.batch_size)
    if len(rows)!=12*len(external):
        raise AssertionError('Expected 12 per-model frozen conditions')
    write_mode(root,out,args.mode,external,hashes,rows,replay)

if __name__=='__main__':
    main()
