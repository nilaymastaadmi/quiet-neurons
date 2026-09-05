# Source and licence record

| Component | Origin | Licence |
|---|---|---|
| `experiments/bdh.py` | [pathwaycom/bdh](https://github.com/pathwaycom/bdh), unmodified | MIT, Copyright 2025 Pathway Technology, Inc. |
| `experiments/sparsity_scan.py` | this repo, reproduces the paper's Section 6.4 protocol | see repo LICENSE |
| `experiments/export_weights.py` | this repo | see repo LICENSE |
| `web/bdh.js` | this repo, a port of `bdh.py` to JavaScript | see repo LICENSE |
| Trained checkpoints in `experiments/checkpoints/` | trained here from scratch on synthetic data | see repo LICENSE |

Primary sources:
- Kosowski, Uznański, Chorowski, Stamirowska, Bartoszkiewicz, *The Dragon Hatchling:
  The Missing Link between the Transformer and Models of the Brain*, arXiv:2509.26507.
  Section 6.4 and Figure 14 are what this project reproduces.
- *BDH-CQ: In-Context Learning with Recurrent Latent Reasoning*, arXiv:2608.09888.
  Cited only; its update rules are stated to be proprietary and are not modelled here.
