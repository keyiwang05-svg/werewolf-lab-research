# Research methodology and interpretation

This document supplements the [README](../README.md). It explains what was evaluated and which reproduction claims the selected-code showcase can support.

## Experimental unit and target

- Game: eight seats (two wolves, one seer, one guard, four villagers).
- Example: one living player at a daytime pre-vote snapshot. Multiple snapshots may come from the same game.
- Features: public speech, public mentions and earlier voting behavior. Private seer checks, hidden roles and future events are not predictor inputs.
- Supervised target: `label = 1` for a Werewolf player, `0` otherwise.

## Six-family original benchmark

The 1,200 games are 200 per family: random, rule, stealth, skeptical, stochastic and deceptive. Total samples: 19,728. In a fold withholding one family, the held-out family contributes 40 test games; the other five supply 600 training and 200 validation games across pre-defined seeds. Splitting by game and seed prevents within-game sample leakage. Families share simulator code and linguistic templates, so this is not an independent-source generalization benchmark.

## Models and interventions

- Baselines: Logistic Regression and Gradient Boosting; text, numeric, combined, and role-language masked feature conditions.
- NLP comparison: TF-IDF + LR, frozen multilingual MiniLM sentence embedding + LR, and full-parameter RBT3 fine-tuning.
- RBT3: roughly 38.5M trainable parameters; archived full runs used three epochs, batch size 4, learning rate 2e-5 and maximum sequence length 96, with validation-loss checkpoint selection. Raw and canonical role-language masked conditions were trained separately.
- Masking is an intervention on text and information, not a clean causal identification of a single 'spurious' feature.
- Models differ by backbone as well as training, so a frozen MiniLM-vs-fine-tuned RBT3 contrast cannot isolate fine-tuning effects.

## External-policy evaluation

A separately implemented policy generated 100 complete games (seeds 1100–1199) and 1,551 public pre-vote observations. This policy does not reuse the six-family policy module but does reuse the game engine and public feature extractor. The 36 configurations (6 old training folds × 3 NLP methods × 2 text conditions) were evaluated as saved, without refitting on the new 100 games. They share the same external test population; their scores are correlated.

`results/external_36_metrics.csv` contains one row per configuration and stores ROC-AUC, average precision, Brier score, sample and game counts, and pointwise game-cluster bootstrap intervals. `results/external_54_paired.csv` contains paired comparisons. `results/fold_summary.csv` summarizes external ROC-AUC across the six training folds for each method/condition.

Published-check results: 34/36 AUC point estimates < 0.5; 26/36 pointwise unadjusted bootstrap CI upper bounds < 0.5. These are not 26 family-wise-error-controlled discoveries. For fixed-model inference, resampling games does not reflect retraining stochasticity, model-selection uncertainty, alternative simulator policies, or human strategies.

## What is reproducible here?

**Reproducible with this showcase alone:** CSV summaries and counts via the included stdlib-only verifier; the self-contained unit tests for the separately implemented policy.

**Not end-to-end reproducible from this showcase alone:** generating the full games, reconstructing all 19,728 training observations, full RBT3 fine-tuning, replaying the original 36 model predictions, or reproducing live Kimi outputs. Selected scripts reference additional private project modules and locally archived model checkpoints by design. The private development repository contains a broader record; its availability is not implied by this public selection.

## Labels and safety boundaries

The reporting tables are internally consistent and the supplied evaluation scripts specify positive-class predictions. Original per-game private role audit logs are not redistributed, so this release does not independently revalidate every external example's label against private engine state. No raw private Kimi messages, live credentials or full privileged game logs are included.
