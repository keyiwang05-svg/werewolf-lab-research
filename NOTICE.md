# Source, attribution and scope of selected code

## Upstream game engine

**LLMWereWolf**: https://github.com/Mai0313/LLMWereWolf  
**Referenced baseline commit**: `76b23b49fd2055e8e1e31b1a38fa1514ed16705f`

The full local Werewolf Lab was built as a research derivative of LLMWereWolf. Its original game-engine architecture, roles, event framework and interfaces were reused. They are **not** claimed as original research code by Keyi Wang.

The upstream repository publishes the MIT text with the literal placeholder `Copyright (c) <year> <copyright holders>`. [UPSTREAM_LICENSE.txt](UPSTREAM_LICENSE.txt) retains the upstream license text **verbatim**, rather than inventing a copyright holder. This placeholder and its rights attribution should be reviewed with the upstream maintainer before public release.

## Selected research contributions

The showcase publishes selected research scripts associated with supervised pre-vote role prediction, Transformer fine-tuning, a separately implemented offline agent policy, and fixed-model external evaluation, plus aggregate synthetic results. These are excerpts of a larger local research codebase, not an independent implementation of the full game engine.

AI-assisted coding tools supported development. No independent new open-source license is asserted for the original research additions by this attribution notice; public visibility alone is not a license to reuse them. The upstream MIT notice applies to portions derived from that upstream software. Redistribution or reuse should respect applicable upstream terms and contributor rights.

## Exclusions and limitations

API tokens, private Kimi histories, raw privileged game logs, full large model checkpoints, and development environments are intentionally excluded. The selection's core result metrics can be checked from CSV, but its full training pipeline is not publicly runnable from this subset.
