<!-- Committed per the problem statement's own acceptance test for the one-page summary
     (p.11): "give the summary, without additional context, to a person or capable AI system.
     It should enable them to explain your central claim, the underlying mechanism, the roles
     of BDH and BDH-CQ, the evidence supporting your explanation, and its limitations without
     introducing major inaccuracies."

     Run 2026-09-07 against web/concept-summary.pdf, PDF only, no README, no repository, no
     live page. Raw output, unedited. What it changed is recorded at the bottom. -->

# Quiet Neurons: document-only review

Source: `C:\Users\toshn\Downloads\concept-summary.pdf`

## Part 1

### Central claim

In the document's reproduced BDH-GPU experiments, predictable text is quieter in aggregate, but predictability itself does not fully explain activation. The proposed distinction is whether the relevant knowledge is stored in the model's weights or must be supplied by causal context, while the document explicitly limits this to an observed signature, not a proven cause.

### Mechanism and difference from a conventional Transformer

BDH-GPU structurally produces sparse non-negative activations by elementwise multiplying two ReLU vectors, making their product sparser than either input. Its attention is described as a causally masked Gram matrix of activations with no softmax; the document contrasts this with a dense Transformer feed-forward layer that evaluates every unit for every token, where sparsity is emergent and exact zeros require a hard rectifier.

### Roles of BDH and BDH-CQ

BDH-GPU is central: it is the formulation described mechanistically and measured in this document. BDH-CQ is not central: its internals are "not public," and its reported benchmark result is explicitly developer-reported rather than reproduced here.

### Evidence and strength

- **Measured here, moderate strength:** the authors reproduce the aggregate quieting trend in 3 locally trained models at 2,048, 8,192, and 16,384 neurons, reporting 1.47x, 2.12x, and 1.87x ratios across 5 samples. This supports a narrow, small-scale replication, not the original 65,536-neuron result.
- **Measured here, stronger against the simple predictability explanation:** equally well-predicted sequence segments differ by 2.65x in activity, and the surprise-injection check raises activity at the injected character while leaving the preceding character unchanged. This shows the aggregate relationship is insufficient by itself.
- **Measured here, weak causal support:** fixing the word collapses a 2.35x gap to 1.04x, but the document calls the control degenerate and says it cannot separate knowledge provenance from difficulty.
- **Measured here, implementation check only:** browser-model and PyTorch parity, including a 6.1e-5 maximum logit difference and 12 of 12 bit-identical sparsity series. This supports consistency of the implementation, not the architectural claim.
- **Cited from others, not independently established here:** the original BDH paper's Fig. 14 range; Pathway's Sudoku, scaling, and BDH-CQ ARC-AGI-1 results. The document labels these developer-reported, and says the external grounding is a commercial Amazon SageMaker partnership rather than independent evaluation.

### Limitations

The document has no non-degenerate causal experiment that separates "knowledge in weights versus context" from task difficulty. That is the most serious limitation because it leaves the headline explanation stronger than the demonstrated result.

### Follow-up question

Can you run a preregistered, non-degenerate intervention that independently varies where knowledge is available, in weights or context, while holding token difficulty, prediction error, position, and training exposure constant?

### Where to go next, in order

1. Reference [1], section 6.4 and Fig. 14, for the result replicated here.
2. Reference [1], section 13.2 and Eq. (8), for the mechanism and quantity counted.
3. Reference [3], for why sparsity alone does not distinguish architectures.
4. The linked artifact and its counterexample, then reproduce the supplied command.

## Part 2

### Contradiction or tension

Yes, there is an epistemic contradiction. The document says, "What separates them is not predictability but where the knowledge came from," but later says, "we establish the signature, not its cause." The former asserts the causal distinction; the latter correctly says it has not established one.

### Claim stronger than its evidence

"In aggregate the paper is right: predictable text is quieter" is broader than the evidence presented: 3 small models trained for 309 steps, with none completing the 4,000-step schedule. To support it as a general result, the document would need independent replications at the reported scale, completed schedules, and a predefined sampling and statistical protocol.

The stronger claim is again the provenance claim. It would need a non-degenerate causal intervention that holds difficulty and predictability fixed while relocating the relevant knowledge between weights and context.

### Numbers, terms, or references lacking context

- "Layer 0 runs backwards (0.80, 0.80, 0.86)" does not define the ratio direction or what "backwards" quantitatively means.
- "Five pinned samples" is not defined.
- "Factifies" is not explained and appears malformed in context.
- BDH-CQ is explicitly non-public, so its mechanism and reported benchmark cannot be independently interpreted from this document.

### Initially misleading point

The bold aggregate conclusion, "predictable text is quieter," initially reads like predictability is the driver. The next paragraph corrects that impression by showing similarly predictable inputs with a 2.65x activity difference and reframing the issue as knowledge provenance.

### Confidence

**7/10.** The single change that would raise this most is a public, non-degenerate causal experiment with code, data, and a protocol that separates provenance from difficulty.


---

## What this run changed in the document

Committed unedited, including the parts that went well and the parts where the reader was
wrong. Three of its complaints were **reader transcription errors, not document defects**, and
were checked before being dismissed: the PDF's text layer extracts `2,309` (not "309"), `§3.2`
(not "13.2") and `falsifies` (not "factifies"). Those three are why the confidence rating below
should be read as a floor rather than a verdict.

**Acted on:**

1. **The contradiction it found in question 8 was real and was fixed.** "What separates them is
   not predictability but where the knowledge came from" asserts a separator, while the same
   document says it establishes the signature and not the cause. After the degenerate control
   was retracted earlier the same day, the assertion had even less behind it. It now reads
   "what *covaries* with the difference", and says explicitly that the control which tried to
   establish causation failed. This is the **third** time a blind reader has caught this same
   class of overclaim in this document; the first two rounds caught "refuted the paper" and
   "the variable is where the knowledge lives".
2. **"In aggregate the paper is right"** was flagged as broader than three toy models support.
   Now "in aggregate our three models agree with the paper".
3. **"Layer 0 runs backwards (0.80, 0.80, 0.86)"** gave no direction. Now states that ratios
   below 1 mean more neurons fire while repeating.
4. **"Five pinned samples"** was undefined. Now "five independently seeded evaluation samples".
5. The reader worked out unprompted that the parity check supports implementation consistency
   and not the architectural claim. That inference is now stated in the document rather than
   left to be made.

**Deliberately not changed:** the reader called the bold aggregate line initially misleading and
then said the next paragraph corrects it. That is the intended structure, state the paper's
result then sharpen it, and the correction landing is the test passing rather than failing.

**What it got right that mattered most:** asked which of BDH and BDH-CQ is central, it answered
that BDH-GPU is central and BDH-CQ is not, without inventing a role for BDH-CQ. The problem
statement warns specifically against a BDH mention "added only to satisfy the rules", and this
is the only direct evidence available that the document does not read that way.
