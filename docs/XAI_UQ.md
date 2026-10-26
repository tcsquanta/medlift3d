# Uncertainty quantification (step 16) and explainability (step 17)

Two analyses on top of the trained models. Both run **on Kaggle** from `medlift_kaggle.zip` (the RunPod
run also runs them once at the end, so their outputs are already in the bundle).

| step | question | needs | Kaggle time (estimate) |
|---|---|---|---|
| 16 `16_uncertainty.py` | How sure are we about each nodule's volume / mass, and about its clinical category? | `work/results/pipeline.jsonl` (step 9) | seconds, CPU |
| 17 `17_explain.py` | What did the X-rays measure, what did the diffusion prior add, what does the detector look at, which nodules are hard? | both models + the 15 bundled test patients + `pipeline.jsonl` | ~30-50 min on a T4 for 12 nodules (part B); parts A and C: minutes |

---

## Step 16: uncertainty quantification with conformal predictive systems (`crepes`)

### What it does, in plain words

Step 9 measures every nodule 8 times (8 diffusion samples, the "ensemble"). Their spread is a hint of
how unsure the model is, but a hint is not a guarantee. **Conformal prediction** turns it into numbers
with a guarantee: we look at how wrong the method actually was on the **cal** patients (truth known) and
use exactly that to size the uncertainty for new patients. Step 10 already does this for 90% intervals.

Step 16 uses a **conformal predictive system** (CPS, from the `crepes` library). Instead of one interval it
gives each test nodule a whole calibrated distribution: "the true volume is below v with probability p",
for every v. From that we get:

1. **Intervals at any level**, in four variants:
   * *standard*: the same relative width for every nodule;
   * *normalised by the ensemble spread*: wider where the 8 samples disagree (this is step 10's method);
   * *Mondrian size groups*: calibrated separately for small / medium / large nodules (by the *estimated*
     size, known before the truth). Plain conformal intervals are right **on average**; they can be too
     narrow for small nodules and too wide for large ones. Mondrian makes them right **inside each group**;
   * *Mondrian + normalised*: both.
2. **Calibrated probabilities of clinical decisions.** The European Position Statement on lung cancer
   screening (EUPS, Oudkerk et al. 2017; built on NELSON volumetry, the categories the demo already shows)
   manages solid nodules by volume: < 100 mm3 (low risk), 100-300 mm3 (indeterminate, 3-month follow-up),
   >= 300 mm3 (refer). For each nodule: P(volume >= 100 mm3) and P(volume >= 300 mm3),
   the probability of each category, and the **set of categories** that contains the true one 90% of the
   time (e.g. "100-300 or >= 300 mm3" when the nodule is close to a threshold). Compared with the
   uncalibrated alternatives: the share of the 8 samples above the threshold ("ensemble vote"), the spread
   read as a log-normal error, and the point estimate alone.

### Outputs (`outputs/16_uncertainty/`) and how to read them

| file | what to look at |
|---|---|
| `table_uq_intervals.md` | coverage close to 90% (the guarantee); **width**: if "normalised" is narrower than "standard" at the same coverage, the diffusion ensemble's spread carries real information about the error. CRPS: quality of the whole distribution (lower = better). The last row reproduces step 10 with our own code (cross-check: should match "normalised") |
| `table_uq_groups.md`, `group_coverage.png` | coverage per size group: does plain conformal under-cover small nodules, and does Mondrian fix it? |
| `table_uq_risk.md`, `reliability.png` | Brier score and calibration error (ECE) of P(V >= 100), P(V >= 300): lower is better; calibrated probabilities should lie on the diagonal of the reliability plot. Category sets: how often they contain the true category, and how often they are a single category |
| `table_uq_methods.md` | the same for every method (ours vs physics-only) |
| `cpd_examples.png` | the calibrated distribution of 4 test nodules with the truth and the EUPS thresholds |
| `uq_test_nodules.csv` | per test nodule: estimate, 90% interval, P(>= 100), P(>= 300), category set, truth |

**In the demo:** every report now shows the calibrated category probabilities, e.g.
`EUPS category, calibrated probability (step 16): < 100 mm3: 4% | 100-300 mm3: 71% | >= 300 mm3: 25%`
(read from `outputs/16_uncertainty/cps_calibration.json`; computed with plain numpy, tested to be identical to
`crepes`). This is what a doctor can act on: not only "about 180 mm3" but "71% likely indeterminate, 25% likely
refer".

**Technical choices** (fixed in advance, not tuned on the test patients): log space (errors are
multiplicative); sigma = ensemble log-std + 0.02; 3 equal-count size groups on the cal nodules (fewer if a
group would have < 19 nodules, the minimum for a finite 90% interval); probabilities from the pooled
normalised CPS (all cal nodules -> finest resolution), the Mondrian version shown as a second row; the
CPS value is the expected smoothed conformal p-value (tau = 1/2): deterministic and valid on average.

Install on Kaggle: `pip install crepes scikit-learn` (scikit-learn is usually preinstalled).

---

## Step 17: explainability

### A. Error drivers (CPU, seconds)

Which nodule properties go with large volume / mass errors of our method: size, solidity (ground glass
-> solid), density, the radiologists' own disagreement, malignancy rating, the ensemble spread, the physics
data misfit. Spearman correlation with a bootstrap 95% interval, plus the error per size and solidity
group. Output: `table_error_drivers.md`. Read: a correlation whose interval excludes 0 is a real pattern
("small ground-glass nodules are hardest"); this tells a doctor when to trust the number less.

### B. Seen / unseen split: what the X-rays measured vs what the prior filled in (the main part)

**The idea.** Fifteen X-ray views from a 30 deg sweep cannot see everything in 3D. Every 3D volume can be
split exactly into two parts:

* **seen** = the part the 15 views measure; it is fully determined by the X-ray data;
* **unseen** = the part the views cannot see at all: adding or removing it does not change a single
  measured pixel. Mathematically: the null space of the projection operator.

A physics-only reconstruction (SIRT) contains only the seen part. Whatever unseen content our diffusion
reconstruction contains was supplied by the **learned prior**: the data cannot have put it there. We
compute the split for each nodule box (40 x 50 x 40 mm) of four volumes: the **true CT**, ROI-SIRT,
diffusion, diffusion + Gaussians, using the exact projector of the simulation (CGLS, checked by
|A unseen| / |A x| close to 0).

**What it shows** (`table_explain.md`, `<nodule>_seen_unseen.png`):

| number | meaning |
|---|---|
| unseen share of the **true** nodule's signal | how much of a real nodule is invisible to 15-view DTS: the part no physics method can ever recover |
| unseen share of the **true** nodule's **mass** | close to 0: the mass is seen by every view. **This is why H1 holds** (mass error < volume error), now shown on each nodule, not only as a formula |
| volume from the seen part only / full | how much the depth smear inflates the volume when the unseen part is missing |
| unseen share of **ours** | how much the prior added |
| **unseen skill** vs truth | 1 = the prior filled in exactly the true invisible part; 0 = no better than adding nothing (physics); **< 0 = the prior invented structure** (hallucination). This is the honest test of the generative prior |
| seen skill vs truth | how well the measured part matches the truth (noise, ROI-SIRT data consistency) |
| Spearman(unseen share of the truth, our volume error) | do less visible nodules get larger errors? |

The figure shows a side view (depth -> horizontal, the direction DTS resolves worst): the true nodule is a
compact blob; its seen part is smeared in depth; its unseen part is the sharp core with negative side lobes
-- exactly what the prior has to supply. Below: depth profiles through the nodule centre, and the
disagreement of the diffusion members (where the model itself is unsure).

### C. Detector saliency

For each suggestion of the learned detector on the test scans: which voxels of the DTS volume made it fire
(|input x gradient| of the heatmap at the suggestion, 64^3 window). `table_saliency.md`: share of the
saliency within 10 mm of the suggestion for true vs false positives, and on the nodule itself for true
positives. `saliency_examples.png`: 3 true and 3 false positives (coronal and side view). Read: a detector
that "looks at" the nodule concentrates its saliency there; false positives that look at vessels or ribs
explain the typical mistakes.

---

## Run on Kaggle

After Cells 1-2 of the Kaggle notebook (RUNPOD_FULL_RUN.md section 10 / README Part B):

```bash
!pip install -q crepes scikit-learn
!python scripts/16_uncertainty.py
!python scripts/17_explain.py --limit 12 --members 4 --outputs /kaggle/working/outputs_kaggle
!cat outputs/16_uncertainty/table_uq_*.md /kaggle/working/outputs_kaggle/17_explain/table_*.md
```

The bundle already holds step 17 from RunPod (30 nodules, 8 members): `--outputs` keeps the Kaggle run
(12 nodules of the 15 bundled patients) separate. `--skip-maps` runs only parts A and C (minutes).

---

## Does this add novelty? (honest answer, literature checked Oct 2026)

| addition | what already exists | what is new here |
|---|---|---|
| Conformal predictive distributions for nodule volume / mass with **calibrated EUPS category probabilities** and category sets | conformal *intervals* for CT measurements; conformal risk control for nodule *detection* (Hulsman et al., MLR Press 2025); CPS in general (Vovk et al.; `crepes`, Boström 2022) | decision-level calibrated uncertainty for DTS nodule volumetry; test of whether the generative ensemble's spread is informative (normalised vs standard) and of size-conditional coverage (Mondrian) |
| Seen / unseen (range / null space) analysis of the reconstruction | null-space "hallucination maps" (Bhadra, Kelkar, Brooks, Anastasio, IEEE TMI 2021); range-null-space *reconstruction* methods (DDNM, RN-SDEs 2024) | used as an **explanation of a clinical measurement** in DTS: per-nodule proof that the mass is seen and the depth shape is not (H1), how much of a real nodule is invisible to 15-view DTS, and an "unseen skill" score of the diffusion prior against the true CT |
| Detector saliency, error drivers | standard XAI | supporting analysis, not a claim |

**What you can claim:** calibrated, decision-level uncertainty for DTS nodule measurement; a physics-grounded
explanation of what the X-rays determine and what the learned prior contributes, quantified against the
truth. **Do not claim** the first use of conformal prediction in lung nodules, the first null-space
hallucination analysis, or that saliency maps are new; cite Bhadra et al. 2021 for the null-space idea.

**What makes it a contribution and not decoration:** each part answers a question the main results raise
-- *Can a doctor act on the number?* (category probabilities), *Why is mass better than volume?* (mass is
seen), *Is the diffusion model inventing the nodule?* (unseen skill), *When should I not trust it?*
(error drivers, size-group coverage). Report the numbers whichever way they come out; a negative unseen
skill would be an important, honest finding about generative priors in DTS.

## Limitations

* The seen / unseen split is local (nodule box, the rest of the prism treated as known) and uses the
  simulation's own projector; in a real scanner the operator is only approximately known.
* Finite iterations: components the data barely constrain count as unseen (practically invisible at this
  noise level); the reported residual quantifies it.
* Category probabilities use the radiologist consensus volume as truth, which itself varies between readers
  (step 10 reports the reader CV). The EUPS thresholds are for solid nodules; we apply them to all for
  illustration.
* With ~75 cal / ~75 test nodules, group coverages and reliability curves are coarse (a few nodules per bin).
