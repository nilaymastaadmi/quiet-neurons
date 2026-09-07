# AI assistance, code, data, asset and licence disclosure

Required by the DataForge 2026 Pathway track problem statement, which asks that "all
AI-generated, reused, or forked work must be disclosed in the README" and that the package
carry "an AI assistance, code, data, asset, and license disclosure". The README summarises
this; the full record is here.

## AI assistance

**This project was built with AI assistance (Claude) throughout: code, prose, analysis and
review.** That covers the training and measurement scripts, the JavaScript port of the
forward pass, the parity and profiling pages, the explainer page, and this documentation.

What that does and does not mean here:

- **No claim, number or figure in this project came from a model's memory.** Every measured
  number is printed by a committed script from a committed checkpoint, and is reproducible
  with the commands in the README. Where a number is quoted from someone else's work it is
  attributed and labelled as not reproduced here.
- **The parity test exists because of AI assistance, not in spite of it.** The JavaScript
  port had to be proved equivalent to PyTorch rather than assumed equivalent, so
  `web/parity.html` diffs the browser forward pass against a PyTorch reference trace on every
  load. It is the first thing a sceptical reader should open.
- **Assumptions made during the build were wrong and were caught by measurement.** The pinned
  evaluation sample in `measure.py` exists because an earlier measurement drew its evaluation
  words after training had consumed the random number stream, so its numbers were not
  checkable. The per-phase profiler exists because a performance assumption about the browser
  forward pass turned out to be wrong. The central claim itself was narrowed after the data
  refused a stronger version. These corrections are visible in the git history.
- **The author understands and can defend every component**, which is the standard the problem
  statement sets. The architecture is roughly 150 lines of PyTorch and the port is 265 lines
  of annotated JavaScript, each step naming the line of `bdh.py` it reproduces.

## Code

| Component | Origin | Licence |
|---|---|---|
| `experiments/bdh.py` | [pathwaycom/bdh](https://github.com/pathwaycom/bdh), **used unmodified**. This is the BDH-GPU formulation, the one that repository ships. | MIT, Copyright 2025 Pathway Technology, Inc. |
| `experiments/sparsity_scan.py`, `measure.py`, `export_weights.py`, `make_scaling.py` | Written for this submission | MIT, see `LICENSE` |
| `web/bdh.js` | Written for this submission: a port of `bdh.py` to JavaScript, verified against it by `web/parity.html` | MIT, see `LICENSE` |
| `web/index.html`, `worker.js`, `parity.html`, `profile.html` | Written for this submission | MIT, see `LICENSE` |
| `concept-summary.html`, and `web/concept-summary.pdf` rendered from it | Written for this submission. Text is ours; the PDF is produced from the HTML by headless Chrome so it can be regenerated | MIT, see `LICENSE` |
| PyTorch | Dependency, not vendored | BSD-3-Clause |

Nothing in this repository is a fork. `bdh.py` is a vendored dependency, unmodified, and the
contribution sits alongside it rather than inside it.

## Data and weights

| Asset | Origin | Licence |
|---|---|---|
| `experiments/checkpoints/bdh_n2048.pt`, `bdh_n8192.pt`, `bdh_n16384.pt` | **Trained here from scratch** on synthetic data, on a laptop CPU. Not official BDH weights and not presented as such. | MIT, see `LICENSE` |
| `web/data/weights.bin`, `manifest.json`, `reference.json` | Exported from `bdh_n2048.pt` by `experiments/export_weights.py` | MIT, see `LICENSE` |
| `web/data/scaling.json`, `web/data/traces/*.json` | Produced by `experiments/measure.py` from the checkpoints above | MIT, see `LICENSE` |
| `experiments/results/contrast.md`, `experiments/results/latency.md`, `experiments/results/word_scatter.README.md`, `experiments/results/pool_ladder.README.md`, `experiments/results/onepager_test.md` | Method notes for the three measurements taken by driving the running page rather than by `measure.py`: WCAG contrast, interaction latency, and the per-word surprise-injection scatter, each carrying the console script that reproduces it; plus the pre-registered prediction for the provenance ladder, committed before its data existed | MIT, see `LICENSE` |
| `experiments/results/*.csv`, `experiments/results/*.log` | Raw output of the scripts above. `results/measured.csv` is the source of record for every published number | MIT, see `LICENSE` |
| `experiments/stepmatched/*.log` | Logs of the step-matched retrains and the re-measurement. The n=2,048 retrain has no separate training log; its measurement output is inside `rebuild_all.log`. `sparsity_scan.py` also writes a single-sample scaling_results.csv at training time, deliberately **not** included in this package. Its unaveraged ratios differ from the 5-sample values in `measured.csv`, and two numbers for one quantity is worse than one | MIT, see `LICENSE` |
| Training and evaluation text | **Synthetic**, generated in-process by the scripts: a fixed random warm-up sequence followed by random eight-letter words. No corpus, no scraped text, no third-party dataset. | n/a |

## Assets

| Asset | Origin | Licence |
|---|---|---|
| IBM Plex Sans, IBM Plex Sans Condensed, IBM Plex Mono | Served from Google Fonts by `web/index.html`; not vendored into this repository | SIL Open Font License 1.1 |
| Colours, layout, charts, the neuron grid, the attention heatmap | Written for this submission as inline CSS and hand-written SVG and canvas drawing; no UI framework, no chart library, no icon set, no images | MIT, see `LICENSE` |

There are no images, no video, no audio, no icons and no third-party JavaScript in this
project. The page loads two things over the network: its own files, and the fonts above.

## Prior work this builds on

Cited in the README beside the claims they support, and not reused as code:

- Kosowski, Uznański, Chorowski, Stamirowska, Bartoszkiewicz, *The Dragon Hatchling*,
  arXiv:2509.26507 (2025). Section 6.4 and Figure 14 are what this project reproduces.
- *BDH-CQ: In-Context Learning with Recurrent Latent Reasoning*, arXiv:2608.09888 (2026).
  Cited only. Its dimensions and update rules are stated to be proprietary and are not
  modelled, simulated or written out anywhere in this project.
- Li et al., arXiv:2210.06313; Mirzadeh et al., arXiv:2310.04564; Liu et al.,
  arXiv:2310.17157; Vaswani et al., arXiv:1706.03762; Gu and Dao, arXiv:2312.00752.

## Mentorship

None. This is a solo submission with no mentor involvement to disclose.
