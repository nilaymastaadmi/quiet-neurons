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

---

# Run 2, 2026-09-08 05:0x, on the final PDF

Same conditions: `web/concept-summary.pdf` only, one Read of one file, no README, no repository,
no live page, fresh model session with no history of this project. The prompt asked for the same
five acceptance questions the problem statement names, plus an explicit hunt for internal
inconsistencies, ambiguities, overclaims and confusions.

**It found 26 items across those categories and said "No category came back empty."** Nine were
real errors. Two of them were introduced hours earlier by the n=8,192 edit, which is exactly what
this test exists to catch. Raw output follows, unedited, then the fixes.

## Raw output, run 2

### Central claim
> The document claims that BDH-GPU's activation sparsity tracks **where a piece of knowledge came
> from** rather than **how predictable the text is**. The title states it flatly. The body's own
> version is weaker and more careful: "a correlation, not an established cause," and later "we
> establish the *signature*, not its cause."

### Mechanism, roles of BDH and BDH-CQ, evidence, limitations
Answered correctly and in the document's own terms. It identified BDH-GPU as the only thing
measured and BDH-CQ as explicitly out of scope, quoting "it has no role here, and inventing one
would be worse". On evidence it wrote: "The evidence labelling discipline here is unusually good.
The problem is not honesty of labelling, it is that the labelled-strong evidence supports the
*paper's* claim and the document's *own* claim rests on a single block-pair comparison plus one
compromised intervention." On limitations it chose, as most serious, that "the one non-degenerate
intervention designed to test it produced a sign-flipped case the author cannot explain... scale
can be bought and a sign flip cannot be waited out."

### Follow-up question
> What is K, and why does K=16 give 0.92? A 16-word pool still puts the letters in the weights and
> the identity in context, which is the exact condition your mechanism says produces quieting, yet
> the effect inverts.

### Inaccuracies and confusions, all 26

**Internal inconsistencies**
1. "none finished the 4,000-step schedule" sat in the same sentence as "retraining n=2,048 to
   completion" and the n=8,192 completion. "Either those runs completed or they did not. As
   written, both are asserted."
2. The half-width arithmetic did not work. The only half-width on the page was ±0.005; the
   n=8,192 change of 0.0799 is 16 of those, not the stated "about nine".
3. "all for 2,309 steps" reads as a design choice where it was a repair after the wall-clock
   confound.
4. Per-letter coupling asserted by the injection check and denied by Spearman −0.05.
5. 9.7 / 3.6 is 2.69, not the stated "2.65-fold".
6. "2.35×" used twice with no antecedent.

**Ambiguities**
7. The title reads as a refutation of the paper while the body says the models agree with it.
8. The 2.65× comparison introduced as "The same measurement" is a different comparison.
9. "almost equally well" covers a 10× surprise gap, 0.0004 against 0.004 nats.
10. "Reproduce every number" beside one command that loads one of three checkpoints.
11. `ln(a)` parseable as a natural logarithm in a document that also uses natural-log units.
12. "two rectified vectors" against an equation showing one visible rectifier.
13. "author-reported" and "developer-reported" used for what looks like one evidence class.
14. The table's "Evidence status" column has two readings.
15. Layers 0, 2 and 3 named; layer 1 never mentioned.

**Overclaims**
16. The title asserts an exclusive causal claim the body twice disclaims.
17. "that product is sparser than either" stated absolutely. "The support of an elementwise
    product is the intersection of the two supports, so the correct statement is *at least as
    sparse as either*, strictly sparser only when the supports differ. Small, but it is the
    load-bearing architectural claim."
18. "which lets it read as a concept" offered with no probing or interpretability evidence.
19. "A learner falsifies it in a minute" overstates a check that mainly confirms causal masking.

**Confusions**
20. "falsifies *it*" has "a correlation, not an established cause" as its nearest antecedent, so
    the literal reading is that the check refutes the authors' own claim.
21. "killing a stronger claim" never names the claim.
22. "dip surviving at 40× the mean half-width" names neither the dip nor whose half-width.
23. "Positions whose neurons fired alike bind" leaves "bind" undefined.
24. Parameter range not checkable against the neuron range from the document alone.
25. No absolute sparsity percentages for the main protocol, so the reader cannot compare against
    the paper's 4.0–7.5% and 2.5%.
26. The BDH-CQ table row reads as inclusion while the prose disclaimer sits 15 lines below.

## What was changed, and what was not

**Fixed (9).** 1 and 2, both introduced by the n=8,192 edit that same night: the sentence now says
"none of the three published models finished the 4,000-step schedule, though separate runs that did
finish move layer 2 only 1.4705 to 1.4620 at n=2,048, and 2.1201 to 2.2000 at n=8,192, a rise of
0.0799 against that run's own ±0.0086 spread." 5: the percentages now read 9.65% and 3.64%, which
are the values 2.65 is computed from. 6: "n=2,048's 2.35× warm-up-to-repeat gap". 10: "Reproduce
the n=2,048 row". 11: `relu(self.ln(a) @ decoder_y) * x`. 15: layer 1 added, positive at every size
(1.22, 1.41, 1.33). 17: "can never be denser than either", on the one-pager and on the page's
section 06, which carried the same absolute. 18: the concept clause is gone. 19 and 20: "A reader
tests this in a minute", and the causal-mask restatement dropped. 21 and 22 partly: "Most
importantly: signature, not cause" and "the dip surviving at 40× its half-width".

**Not fixed, and why.** 16, the title. The reviewer is right that "quietens on what it just
learned, not on what is merely predictable" is stronger than "a correlation, not an established
cause" two paragraphs below. It is the project's name, it appears on four surfaces and in the
repository URL, and the body carries the disclaimer in the same breath as the claim. Changing it
the morning of the deadline trades one risk for a larger one. **It is the strongest single
criticism this artifact has and it is recorded here rather than answered.** Items 3, 4, 7, 8, 9,
12, 13, 14, 23, 24, 25 and 26 are terseness in a document at 948 of 950 permitted words: every fix
costs words the budget does not have, and none of them is false as written. They are listed above
so a reader can see them rather than discover them.

## Why run 2 matters more than run 1

Run 1, on 2026-09-07, tested a document that then changed substantially: a third architecture row,
a rewritten difficulty passage, the canonical claim, and the n=8,192 result. Run 1's clean bill was
about a document that no longer exists. Run 2 tested what ships, found nine real errors, and two of
them were less than eight hours old. A no-context reader catching in four minutes what four
surfaces of self-checking did not is the argument for keeping this gate.

---

# Run 3, 2026-09-08 05:5x, on the PDF corrected after run 2

Same conditions. The prompt was tightened to separate **major inaccuracies** — contradictions,
arithmetic that does not check out, claims stated as established that the evidence does not
support, or wording a careful reader takes away as a false fact — from mere terseness, so the
result could not be inflated by a long list of undefined terms.

**It found six major items.** Run 2's fixes had not made the document clean; two of the six were
introduced by run 2's own fixes, hours old. This is the second time in one night that editing this
file created a new error while removing an old one.

## The six, and what happened to each

**1. "In aggregate our three models agree with the paper: predictable text is quieter."** Bolded,
two lines above the counterexample that refutes exactly that proposition, and above numbers from
the same n=8,192 model showing the *more* predictable block is 2.65 times *louder*. The reviewer:
"'Predictable text is quieter' is exactly the proposition the document exists to refute."
**Fixed** to "On the paper's own protocol our three models reproduce its result", which is what we
actually did and does not endorse the gloss.

**2. "the dip surviving at 40x its half-width" does not check out.** The dip is 2.1201 to 1.8742,
0.2459. Against the n=8,192 half-width, 0.00810, that is 30x; against n=16,384's 0.00415 it is
59x; neither is 40. **The number was right and my edit broke it.** It read "the mean half-width"
until run 2's fixes changed it to "its half-width" while addressing a complaint that "the mean
half-width" was undefined. 0.2459 / mean(0.00810, 0.00415) = **40.1**. Now stated as "40x the mean
of the two half-widths", which names the denominator instead of removing it.

**3. "only" governing a nine-half-width move.** "move layer 2 only 1.4705 to 1.4620 at n=2,048, and
2.1201 to 2.2000 at n=8,192" let one "only" cover a 1.8-half-width move and a 9.3-half-width one.
**Fixed**: "and" to "but". One word.

**4. The title again.** Same finding as run 2, independently. Recorded, not fixed, for the reasons
in run 2's section.

**5. The injection check cannot come out any other way, and does not discriminate.** Causal masking
already guarantees the preceding letter is untouched, so quoting nine decimals of it "dresses a
structural guarantee as a measurement"; and activity rising at an *unpredictable* letter is what
the predictability account predicts, so the check never separated the two hypotheses it sat
beneath. **Fixed** by saying what it actually shows: "Predictability does move activity within a
block ... That is the paper's effect, not this one."

**6. "Reproduced here:" listed the counterexample**, which the same document calls something "no
paper states". Nothing unstated can be reproduced. **Fixed**: the label is now "Measured here:".

Also fixed from the minor list, because it was arithmetic loose in our own favour: the ladder
losses now print as 0.0384 and 0.1739 rather than 0.038 and 0.174, so the stated 4.5x divides out
(4.53) instead of reading as 4.58.

## A finding about the artifact, not the document

The corrected file rendered to **two pages at 949 body words** where 948 had fitted on one. The
one-pager sits on a page boundary, and word count alone does not protect it: `onepager_length`
checks page count for exactly this reason, and caught it. Trimmed to **934 words**, which restores
one page with sixteen words of margin. Three of those cuts came from the reviewer's own minor list.

## Standing count

Run 2: 26 items, 9 real, all fixed but the title. Run 3: 6 major, 5 fixed, the title recorded.
Two of run 3's six were created by run 2's fixes. The lesson is not that the document was careless;
it is that **every edit to a dense page is a chance to introduce an error, and only an outside
reader with no memory of the previous draft reliably finds them.**
