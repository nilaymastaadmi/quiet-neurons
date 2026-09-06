# Source and licence record

Complete record for every component, dataset and asset in this submission. The
AI-assistance, code, data and asset disclosure is in [`../AI_DISCLOSURE.md`](../AI_DISCLOSURE.md);
this repository's own licence text is in [`../LICENSE`](../LICENSE) (MIT).

## Code

| Component | Origin | Licence |
|---|---|---|
| `experiments/bdh.py` | [pathwaycom/bdh](https://github.com/pathwaycom/bdh), **used unmodified**; the BDH-GPU formulation that repository ships | MIT, Copyright 2025 Pathway Technology, Inc. |
| `experiments/sparsity_scan.py` | This repo. Trains one model and reproduces the paper's Section 6.4 protocol | MIT, `../LICENSE` |
| `experiments/measure.py` | This repo. Measures a checkpoint on a pinned sample; source of every number quoted in this project | MIT, `../LICENSE` |
| `experiments/export_weights.py` | This repo. Exports weights and a PyTorch reference trace for the browser | MIT, `../LICENSE` |
| `experiments/make_scaling.py` | This repo. Builds `web/data/scaling.json` from `results/measured.csv` | MIT, `../LICENSE` |
| `web/bdh.js` | This repo. A port of `bdh.py` to JavaScript, verified against it by `web/parity.html` | MIT, `../LICENSE` |
| `web/index.html`, `web/worker.js`, `web/parity.html`, `web/profile.html` | This repo | MIT, `../LICENSE` |
| `concept-summary.html` and the `web/concept-summary.pdf` it renders to | This repo. The one-page concept summary; the PDF is generated from the HTML by headless Chrome, so it is reproducible rather than hand-laid | MIT, `../LICENSE` |
| PyTorch | Dependency, installed by the reader, not vendored here | BSD-3-Clause |

## Weights and data

| Asset | Origin | Licence |
|---|---|---|
| `experiments/checkpoints/*.pt` | Trained here from scratch on synthetic data. **Not official BDH weights** and not presented as such | MIT, `../LICENSE` |
| `web/data/weights.bin`, `manifest.json`, `reference.json` | Exported from `checkpoints/bdh_n2048.pt` by `export_weights.py` | MIT, `../LICENSE` |
| `web/data/scaling.json`, `web/data/traces/*.json` | Produced by `measure.py` | MIT, `../LICENSE` |
| `experiments/results/*.csv`, `experiments/results/*.log` | Raw output of the scripts above. `results/measured.csv` is the source of record for every published number | MIT, `../LICENSE` |
| `experiments/stepmatched/*.log` | Logs of the step-matched retrains and the re-measurement. `sparsity_scan.py` also writes a single-sample scaling_results.csv at training time, deliberately **not** shipped, because its unaveraged ratios differ from the 5-sample values in `results/measured.csv` | MIT, `../LICENSE` |
| All training and evaluation text | **Synthetic**, generated in-process: a fixed random warm-up followed by random eight-letter words. No corpus, no third-party dataset | n/a |

## Fonts and graphics

| Asset | Origin | Licence |
|---|---|---|
| IBM Plex Sans, IBM Plex Sans Condensed, IBM Plex Mono | Served from Google Fonts by `web/index.html`; not vendored | SIL Open Font License 1.1 |
| Charts, neuron grid, attention heatmap, all page styling | Written for this submission: inline CSS, hand-written SVG, canvas drawing. No UI framework, no chart library, no icons, no images | MIT, `../LICENSE` |

No images, video, audio or third-party JavaScript are used anywhere in this project.

## Primary sources

Cited beside the claims they support, and not reused as code:

- Kosowski, Uznański, Chorowski, Stamirowska, Bartoszkiewicz, *The Dragon Hatchling: The
  Missing Link between the Transformer and Models of the Brain*, arXiv:2509.26507 (2025).
  Section 6.4 and Figure 14 are what this project reproduces.
- Engdahl, Kosowski, Chorowski, Stamirowska, Uznański et al., *BDH-CQ: In-Context Learning
  with Recurrent Latent Reasoning*, arXiv:2608.09888 (2026). **Cited only**; its dimensions
  and update rules are stated to be proprietary and are not modelled here.
- Li, You, Bhojanapalli et al., *The Lazy Neuron Phenomenon*, arXiv:2210.06313 (2022).
- Mirzadeh, Alizadeh, Mehta et al., *ReLU Strikes Back*, arXiv:2310.04564 (2023).
- Liu, Wang, Dao et al., *Deja Vu*, arXiv:2310.17157 (2023).
- Vaswani et al., *Attention Is All You Need*, arXiv:1706.03762 (2017).
- Gu and Dao, *Mamba*, arXiv:2312.00752 (2023).
