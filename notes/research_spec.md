# Research Spec

## Project goal

This repository is a clean reimplementation of two related research projects from our group:

1. "Deep Networks Learn Features from Local Discontinuities in the Label Function"
2. "Layers as Lenses: A Narrative of Feature Learning in Deep Networks"

The goal is to convert messy supplementary notebook code into a clear, reproducible, modular Python project. The first objective is faithful reimplementation, not novelty. Extensions should come only after the baseline pipeline is correct and reproducible.

## Source of truth

There are no meaningful external web references for this project.
The source of truth is:

- the attached papers,
- the old supplementary notebooks in `notebooks/old notebooks/`,
- any notes added in this repository.

If there is a conflict:

1. trust the paper over the notebook,
2. trust explicit experiment details over implied behavior,
3. document ambiguities instead of guessing silently.

## First milestone

Build a minimal clean implementation that can:

- generate or load the required data setting,
- instantiate the core model cleanly,
- run training from a config file,
- evaluate and save outputs to `results/`,
- reproduce one baseline result from each paper or a clearly defined subset.

## Non-goals for the first phase

- No premature abstraction for many experiment types.
- No large framework before one successful baseline run.
- No extensions until baseline behavior is documented and reproducible.
- No notebook-only logic that is required for rerunning experiments.

## Code organization principles

- `src/layer_lenses/` contains stable importable code.
- `notebooks/` is for exploration and temporary analysis only.
- `notes/` contains paper notes, experiment specs, and design decisions.
- `configs/` contains experiment configs.
- `results/` contains metrics, plots, tables, checkpoints, and artifacts.

## Notebook migration rule

Old notebooks are reference material, not final implementation.
Any logic needed for reproducibility should be migrated into clean Python modules.
Useful notebook content should be extracted into:

- data generation/loading code,
- model definitions,
- training/evaluation utilities,
- analysis utilities.

## Implementation priorities

1. Understand and summarize the old notebooks.
2. Identify reusable components and duplicated logic.
3. Build a minimal baseline implementation for one paper first.
4. Add tests and config-driven runs.
5. Only then add extensions or generalization.

## Open questions

- Which parts of the notebook code are essential versus exploratory?
- What exact experiments are required for a faithful baseline?
- Which hyperparameters and data settings are truly part of the claimed result?

---

# Extension Plan: Layers-as-Lenses Narrative for ReLU MLPs

Math-rendered copy: `notes/research_spec_math.ipynb` (use this in Cursor when equations do not render in `.md` preview).

This section captures the high-level research plan for extending the
"Layers as Lenses" narrative from DLGNs to standard fully connected ReLU MLPs
on COB-ODT data. It is *intentionally conceptual*: hypotheses, tests,
visualisations, and sequencing only — no code yet. It is the working
research charter for the ReLU analysis phase.

## A. Recap of the Paper's Core Narrative

The paper makes three durable claims that we want to extend to ReLU MLPs:

- **Lens–Blinder–Residue decomposition.** For DLGN, the gradient on neuron
$\mathbf{w}*k$ factors as Residue × Lens × Blinder × $y^* a \mathbf{x}$.
The Lens $\Lambda_k(\mathbf{x})=\prod*{\kappa\ne k}\phi(\mathbf{w}_\kappa^\top\mathbf{x})$ comes
from *other* neurons; the Blinder $\phi'(\mathbf{w}_k^\top\mathbf{x})$ comes from the
neuron itself.
- **Dual role of a neuron and positive feedback.** $\mathbf{w}_k$ is simultaneously
(a) a linear separator on data importance-weighted by its lens and
(b) a component of lens for everyone else. This dual role is the engine
of feature learning.
- **Hierarchy-seeking + self-opposing second-order dynamics.** These give
"leaf-first, then root, then descendants" feature emergence, but also
produce backlash along spurious directions, motivating the L2-then-prune
two-phase training in DLGN.

The narrative-level claim of the paper is broader than DLGN: this should
be a generic phenomenon for any deep non-linear architecture. The goal of
the ReLU analysis phase is to make this claim *legible* in a standard
ReLU MLP.

## B. Conceptual Translation: DLGN → ReLU MLP

This is the crux. We need clean ReLU analogs of three DLGN objects.

### B.1 Gates

- **DLGN-SF gate at layer $\ell$:** $\phi(\mathbf{w}_\ell^\top \mathbf{x})$ — a *shallow*
halfspace in input space (always reads $\mathbf{x}$ directly).
- **ReLU gate at layer $\ell$:** $\mathbb{1}[(W_\ell h_{\ell-1}(\mathbf{x}))*i > 0]$ — a
deep halfspace. For a fixed input $\mathbf{x}$ inside a single linear region,
it is still a halfspace in input space, but its normal depends on the
activation pattern at $\mathbf{x}$:
$$\tilde\mathbf{w}*{\ell,i}(\mathbf{x}) = \bigl((W_\ell)*{i,:}\text{diag}(\mathbf{m}*{\ell-1}(\mathbf{x}))W_{\ell-1}\cdots \text{diag}(\mathbf{m}_1(\mathbf{x}))W_1\bigr)^\top$$
  - For first-layer neurons, $\tilde\mathbf{w}*{1,i} = (W_1)*{i,:}$.
  - For deeper neurons, $\tilde\mathbf{w}_{\ell,i}(\mathbf{x})$ is *input-dependent*. The
  "gate" is not a single vector but a family of vectors over linear
  regions and must be aggregated sensibly (e.g., averaged over the
  neuron's active subset of inputs).

### B.2 Paths

- **DLGN path:** $\pi = (m_1,\ldots,m_L) \in [M]^L$ with multiplicative
gate product across layers.
- **ReLU path:** Same index $\pi$, but the path's contribution is
multilinear and gated:
$$f(\mathbf{x}) = \sum_{\pi} \Bigl[\prod_{\ell=1}^{L-1} m_{\ell,i_\ell}(\mathbf{x})\Bigr]\cdot a_\pi \cdot (W_1)*{i_1,:}^\top \mathbf{x}$$
with $a*\pi = w_{L,i_{L-1}}\prod_{\ell=2}^{L-1}(W_\ell)*{i*\ell,i_{\ell-1}}$.
So a ReLU path naturally has:
  - an **activation indicator** (subset of inputs that turn it on),
  - a **scalar coefficient** $a_\pi$,
  - a **linear separator** $(W_1)_{i_1,:}$ (shared across all paths
  sharing $i_1$).

### B.3 Lens–Blinder for ReLU

The ReLU gradient w.r.t. $(W_k)*{i,:}$ takes the same factorised form,
with the lens being a back-propagated sensitivity through the active
sub-network for $\mathbf{x}$:
$$\nabla*{(W_k)*{i,:}} \mathcal{L} = \mathbb{E}\bigl[\underbrace{\sigma(-y^*\hat y)}*{\text{Residue}}\cdot \underbrace{\Lambda^{\text{ReLU}}*{k,i}(\mathbf{x})}*{\text{Lens}}\cdot \underbrace{\mathbb{1}[(W_k h_{k-1})*i > 0]}*{\text{Blinder}}\cdot y^*h_{k-1}(\mathbf{x})^\top\bigr]$$
The lens for ReLU is a *path-summed* quantity: contributions from every
active path passing through neuron $(k,i)$. This is the natural ReLU
analog of $\Lambda_k(\mathbf{x})$ in DLGN.

## C. Hypotheses (numbered, each independently testable)

Grouped by the level of ambition.

### Tier A — Direct DLGN-style claims for first-layer ReLU

**H1 (First-layer alignment).** First-layer ReLU neurons in a successful
run align (in direction) with ODT decision vectors. Most large-norm
neurons have a high $|\cos|$ with some $\mathbf{u}_i$.

- *Test:* Track norm vs $|\cos|$ scatter over training; expect mass to
migrate from origin toward axis-aligned clusters at high $|\cos|$.

**H2 (Hierarchical emergence in first layer).** Neurons capture leaves
first, then root, then root's children, etc.

- *Test:* For each ODT level, plot fraction of first-layer neurons whose
argmax-$|\cos|$ lies at that level vs epoch. Expect leaf curves to rise
first, then deeper levels with delays.

**H3 (Order-of-emergence quantification).** Even within a level, more
"important" ODT directions (root before its siblings, etc.) are captured
earlier.

- *Test:* For each $\mathbf{u}_i$, define "capture epoch" as the first epoch any
neuron achieves $|\cos|\ge\tau$ with it. Plot capture epoch vs ODT
level / scope of discontinuity.

### Tier B — Path-level structure

**H4 (Path specialisation).** Over training, the active paths $\pi$ tend
to align with ODT root-to-leaf paths. The set of inputs activating a
given $\pi$ becomes (approximately) a subset of one ODT leaf.

- *Test:* For each path $\pi$, measure the entropy of its activation's
distribution over ODT leaves. Track this entropy over epochs; expect
it to drop. At convergence, top paths should map nearly 1-to-1 with
ODT leaves.

**H5 (Path coefficient sparsity / lottery-ticket parallel).** A small
number of paths carry most of the output magnitude.

- *Test:* Plot Lorenz curve / Gini of $|a_\pi|$ over training. Track
effective number of paths $\sim \exp(H(|a_\pi|/\sum|a_{\pi'}|))$.

**H6 (Path-level coverage of input space).** The union of top-$k$ paths'
activation regions covers nearly the entire input distribution and
aligns with ODT leaf partition.

- *Test:* Color test points by their dominant active path; compare
against true ODT leaf assignments via clustering ARI / NMI.

### Tier C — Lens–Blinder–Residue empirics

**H7 (Lens dominates direction near init).** Early in training, the
gradient direction for $\mathbf{w}_k$ is well-predicted by the lens-weighted
average of $y^*\mathbf{x}$. The blinder mostly modulates magnitude, not
direction.

- *Test:* For each first-layer neuron, decompose the empirical gradient
into the ratio of (lens × Residue × signed input) vs (blinder
modulation). Track this ratio over time.

**H8 (Lens reorientation).** As training progresses, the lens of each
neuron shifts from "diffuse" to "concentrated on one ODT leaf" — i.e.
each neuron's lens-weighted data subset becomes sharply localised.

- *Test:* For each neuron, compute the ODT-leaf distribution of its
lens-weighted (or for ReLU, gradient-weighted) training data. Track
entropy and dominant-leaf identity over time.

**H9 (Hierarchy-seeking analog).** Perturbing one neuron along a leaf
direction increases the gradient component of other neurons toward an
ancestor direction. (Direct ReLU analog of the paper's Theorem 1.)

- *Test:* Empirical second-order probe: pick a checkpoint, perturb
$W_1[i,:]\leftarrow W_1[i,:]+\epsilon\mathbf{u}_j$, recompute gradient on
another neuron, project on ancestor $\mathbf{u}_i$. Average over many input
batches. Compare against unperturbed gradient.

**H10 (Self-opposing analog).** Perturbing one neuron along a spurious
direction $\h \perp \mathbf{u}_i$ pushes other neurons along $-\h$ —
magnifying the spurious component for the population.

- *Test:* Same perturbation framework as H9 but with random spurious
$\h$. Aggregate second-order responses.

### Tier D — Dynamics, interventions, predictions

**H11 (Positive feedback ablation).** Freezing deeper layers (no lens
evolution) drastically slows or breaks first-layer ODT alignment.

- *Test:* Compare full training vs frozen-deep-layer training (only
first layer trains). Track first-layer alignment trajectories.

**H12 (Width and lottery tickets).** With wider first layer, the
fraction of neurons that ever align is roughly constant; absolute count
grows. Pruning to aligned neurons preserves accuracy.

- *Test:* Vary first-layer width. After training, prune all first-layer
neurons whose $|\cos|<\tau$ to all $\mathbf{u}_i$; retrain or fine-tune the
rest. Compare to original accuracy.

**H13 (Init-to-solution prediction).** A heuristic score (proximity +
hierarchy-seeking + self-opposition support) at initialization predicts
which ODT direction each first-layer neuron will eventually lock onto.

- *Test:* Direct port of the paper's Section 3.4 heuristic, adapted to
first-layer ReLU. Even if accuracy is modest, beating proximity-only
baseline is a strong signal.

### Tier E — Deeper layers (hardest)

**H14 (Effective gate alignment for deeper layers).** When you compute
effective gating vectors $\tilde\mathbf{w}_{\ell,i}(\mathbf{x})$ averaged across the
active subset of neuron $(\ell,i)$, they too align with ODT directions,
with later layers tending to pick *coarser* ODT levels (root/internal
nodes) rather than leaves.

- *Test:* For each $(\ell,i)$, compute mean effective gate over its
active inputs. Run the same scatter (norm vs $|\cos|$) and ODT-level
coloring. Compare across layers.

**H15 (Layer-as-lens specialisation).** Earlier layers act more as
"separators" (their gates are sharp halfspaces in $\mathbf{x}$); later layers
act more as "lens modulators" (they boost/suppress contributions from
earlier paths).

- *Test:* Quantify per-layer contributions to gradient lens vs separator
role; show monotone trend with depth.

## D. Visualisation Ideas (matched to hypotheses)

- **V1: First-layer norm vs $|\cos|$ scatter, level-coloured.** Already
in place. Add: animate over the quadratic snapshot epochs to see
emergence narrative; overlay the "capture epoch" of each ODT direction.
- **V2: Per-neuron trajectory traces.** For each first-layer neuron, a
curve $|\cos|$ vs epoch with a separate panel for norm vs epoch.
Optionally, mark which $\mathbf{u}_i$ is the argmax over time (and if it
switches). Helps test H1, H3, H13.
- **V3: ODT capture timeline (Gantt-style).** y-axis: ODT nodes ordered
by level; x-axis: epoch; bar starts when first neuron captures $\mathbf{u}_i$
above $\tau$. Helps test H2, H3.
- **V4: Path-level summaries.** For each epoch: path-frequency bar plot
(top 50 by activation count); per-path "ODT leaf entropy" over epochs.
Helps test H4, H5, H6.
- **V5: Lens-weighted data projection.** For a chosen neuron, compute
its lens (ReLU back-prop sensitivity through that neuron); project
lens-weighted training data onto $(\mathbf{u}_i, \mathbf{u}_j)$ planes for various
$(i,j)$. Track how the projection concentrates over time. Helps test
H7, H8.
- **V6: Effective gate scatter for deeper layers.** For each $(\ell,i)$,
compute mean effective gate $\bar{\tilde\mathbf{w}}_{\ell,i}$. Plot the same
norm vs $|\cos|$ scatter, layer by layer. Helps test H14, H15.
- **V7: Second-order probe heatmap.** Matrix indexed by (perturbation
direction $\h \in \mathbf{u}_j$, measurement direction $\z\in\mathbf{u}_i$);
entry = empirical second-order push (averaged over neurons). Compare
to the theoretical sign pattern from Theorems 1–2. Helps test H9, H10.
- **V8: Lottery-ticket pruning curves.** After training, sweep a
norm/cos threshold and prune first-layer neurons. Plot test accuracy
vs pruning fraction. Helps test H12.
- **V9: Init-to-solution prediction confusion matrix.** For multiple
seeds: predicted vs actual eventual ODT lock-in for each first-layer
neuron. Helps test H13.

## E. Suggested Sequencing

Given that we currently have one successful single-seed ReLU run, the
next conversations should sequence like this (each can be a small
focused next step):

1. **Solidify first-layer narrative (H1–H3) with V1–V3.** Make sure the
  "leaves-then-root-then-children" story holds. Closest port of the
   paper's Figs 2/3 to ReLU and confirms that the framework transfers.
2. **Path-level objects (H4–H6) with V4 + V8.** Define and instrument
  paths as first-class entities. Conceptually new step beyond DLGN
   tooling.
3. **Lens–Blinder empirics (H7–H8) with V5.** Pick one or two
  illustrative neurons; show the lens reorientation story.
4. **Second-order probes (H9–H10) with V7.** Direct empirical validation
  of the paper's theorems in the ReLU setting — very compelling if it
   works.
5. **Interventions (H11–H13).** Freezing, pruning, init-prediction —
  these test the explanatory/predictive power of the framework, which
   is what the paper emphasises ("explain, predict, understand").
6. **Deeper layers (H14–H15).** The hardest. Best to attempt only after
  the first five steps stabilise the narrative.

## F. Pitfalls to Watch

- **Linear-region explosion:** ReLU has $O(2^{LM})$ linear regions in
principle. We can't enumerate per region; aggregate quantities
(per-neuron averages over its active set, per-path means over active
inputs) are the right granularity.
- **Sign ambiguity:** ReLU is not sign-symmetric like the gating in
DLGN; so $|\cos|$ may need to be replaced by signed cosine, or
analyzed with care.
- **Bias terms:** Current `ReLUMLP` uses `bias=True`. This shifts gate
boundaries off the origin. The COB-ODT data is on the unit sphere
with zero bias, so to get clean alignment we may want to also study
the `bias=False` variant.
- **Permutation invariance:** Like DLGN, ReLU layers are
permutation-invariant. Per-neuron tracking must allow re-identification
across epochs (Hungarian matching by $|\cos|$, or by parameter
trajectory continuity).
- **Identifiability of paths:** Many paths share the same $i_1$ and may
be degenerate. Path-level analysis should focus on "effectively
distinct" paths via clustering on $a_\pi$ and activation-region.
- **Successful vs unsuccessful runs:** Some seeds may fail (saddle
points). Studying both, and what distinguishes them, is itself
informative — this is what the paper does in Section 3.4.

## G. Open Decisions for Next Conversations

- Whether to formally define ReLU paths as first-class objects in the
codebase, and how to compute them efficiently.
- What subset of H1–H6 to prioritise for a first deep dive.
- Whether to add a `bias=False` parallel run for cleanly comparing
first-layer alignment.





## H. ReLU extension plan:

---
name: ReLU Lens Theorem Elicitation
overview: Experimental program to generalize the single-path DLGN construction and the heuristic solution predictor to ReLU MLPs, with the concrete target of predicting the final convergent hyperplane of each first-layer neuron in a 3x40 ReLU net trained on COB-ODT data.
todos:
  - id: label-harness
    content: Build per-neuron convergence labelling + data-invariance filter (Track B1) on multi-seed ReLU runs
    status: pending
  - id: trajectories
    content: Add per-neuron signed-cos trajectory extraction and lock-in epoch detection over checkpoints
    status: pending
  - id: cell-census
    content: Run Track A ReLU-cell ([D,1,1]) convergence census and bespoke-init trajectory replication, D=4
    status: pending
  - id: predictor-p0-p3
    content: Implement proximity (P0) and gradient-probe (P3) predictors with commitment curves
    status: pending
  - id: effective-gates
    content: Implement effective-gate computation for layers 2-3 (mean input-space normals over active sets)
    status: pending
  - id: predictor-full
    content: Implement full heuristic score (SO + HS terms) with c_SO/c_HS grid search and confusion matrices
    status: pending
  - id: second-order-probes
    content: Implement finite-difference second-order probes and sign-pattern heatmap vs DLGN theorems
    status: pending
  - id: session-log
    content: Log this planning session in notes/session_log.md and link durable decisions
    status: pending
isProject: false
---

# ReLU Layers-as-Lenses: Experimental Setup to Elicit Candidate Theorems

## Framing

The paper's two artifacts to generalize:

- **Single-path DLGN (paper Sec. 3):** a product of D input-space halfspaces converges to one of \(2^{D-2}D!\) solutions, each aligning layer neurons with ODT directions along one root-to-leaf path.
- **Heuristic solution predictor (paper Sec. 3.4 / App. C):** score \(s_{i,\ell}(W) = w_\ell\cdot u_i + c_{SO}\sum_{k\ne\ell}(-w_k\cdot u_i) + c_{HS}A_{i,\ell}(W)\), predicting the gradient-descent solution from initialization at 49% vs 13% proximity-only baseline.

Target question: **for a trained ReLU net (3 hidden layers, 40 wide, on d=100 ODT data), predict which ODT hyperplane \(\pm u_i\) a given first-layer neuron converges to.**

Key structural facts that shape the design (from `notes/research_spec.md` Sec. B):

- First-layer ReLU rows are honest input-space hyperplanes; deeper neurons only have *effective* input-dependent normals \(\tilde w_{\ell,i}(x)\).
- The lens on first-layer neuron \(m\) is path-summed: \(\Lambda_m(x)=\sum_{\pi\ni m} a_\pi \prod_{\ell\ge2} \mathbb{1}[\cdot]\), with path coefficients \(a_\pi\) products of downstream weights.
- A width-1 ReLU chain is degenerate (single-index function, no lens) — the minimal non-degenerate "cell" needs multiple first-layer neurons feeding a narrow downstream chain.

## Track A — Generalized single-path construction: the "ReLU cell"

**Architecture:** `ReLUMLP(hidden_dims=[D, 1, 1])` (and `[D, 2, 2]` as a robustness variant) on depth-D ODT data, D=4, d=100, bias=False. The D first-layer neurons play the role of the D DLGN layers; the width-1 downstream chain provides two deep gates whose input-space regions are shaped by all D first-layer halfspaces — the ReLU lens.

**Experiments:**

1. **Convergence census.** Many seeds (start ~200 inits x 3–5 dataset resamples, mirroring the paper's data-invariance filter). For each run, record the \(\upsilon\)-tuple: for each first-layer neuron, argmax signed-cos ODT direction (or "unaligned" below threshold \(\tau\approx0.85\)). Question: do runs converge to a discrete solution family analogous to the DLGN's \(2^{D-2}D!\) set (one neuron per node along an ODT path, possibly a two-adjacent-path superposition)? Also record saddle/failure runs (train loss stuck between \(\frac{15}{16}\log 2\) and \(\log 2\)).
2. **Bespoke-init trajectory replication** (paper Sec. 3.3 analog): initialize near zero with small perturbations \(-\epsilon u_j\) on chosen neurons; verify the perturbation steers the solution as the hierarchy-seeking narrative predicts. Plot per-neuron signed-cos trajectories against all \(u_i\) (analog of paper Fig. parameter trajectories).
3. **Controls:** width-1 first layer `[1, k, k]` (expected: collapses to single-index, likely \(\pm u_0\)-ish direction — demonstrates the lens requires multiple first-layer features); frozen-downstream variant (no lens evolution, tests H11 causally at cell scale).

**Candidate theorem A:** at (near-)orthogonal init, the cross-block second-order structure of the ReLU cell has the same sign pattern as DLGN Theorems 1–2, with each term weighted by the path coefficient signs \(a_\pi\); consequently the cell converges to per-path solutions. Track A's census + probes give the empirical shape of this statement before attempting proof.

## Track B — Per-neuron fate prediction in the full-width net (the concrete target)

Setting: current scratch config (d=100, D=5, `[40,40,40]`, or D=4 for a cleaner first pass), multi-seed.

### B1. Ground-truth labelling protocol

- Label of first-layer neuron m = signed argmax-cos ODT node at convergence if \(|\cos|\ge\tau\) and stable over the last k checkpoints; else "unaligned". Label set: \(\{\pm u_0,\dots,\pm u_N,\varnothing\}\) (63 classes at D=5, 31 at D=4).
- Port the paper's *data-invariance filter*: fixed init seed, 3–5 dataset/order resamples; keep only neurons whose label is invariant (neuron identity is preserved because init is shared). This defines the clean prediction benchmark, as in the paper's 368/2000.
- Sweep to check label robustness against: weight decay (0 vs 0.005/0.01 — WD may play the role of the DLGN's L2-then-prune cleanup), cosine LR schedule vs constant.

### B2. Predictor ladder (each rung is a testable claim)

1. **P0 proximity:** signed cos of \(W_1[m,:]\) with each \(\pm u_i\) at init. This is the 13%-analog baseline.
2. **P1 + self-opposition (reinterpreted):** in the multipath net, the competitors of neuron (1,m) are *other first-layer neurons*, weighted by lens overlap (shared downstream path mass), not "other layers" as in the single path. Term: \(-c_{SO}\sum_{m'\ne m} \rho_{mm'}\, w_{m'}\cdot u_i\) with \(\rho_{mm'}\) = downstream-path-coefficient overlap.
3. **P2 + hierarchy-seeking:** port \(A_{i,\ell}\) using *effective gates* of downstream neurons: compute \(\bar{\tilde w}_{\ell,j}\) = mean input-space normal of neuron \((\ell,j)\) over its active set (research-spec H14 machinery), take alignments with ancestors/descendants of \(u_i\) along ODT paths, weight by connection strength \(|a_{\pi}|\) between (1,m) and \((\ell,j)\). Grid-search \(c_{SO}, c_{HS}\) as in the paper (0–0.9 x 0–0.9).
4. **P3 gradient probe:** one population-gradient step at init; predict by the gradient's dominant ODT component. Cheap, and separates "information present at init" from "score design".
5. **Commitment curves:** run every predictor from checkpoints at epochs \(t=0,1,2,\dots\) (quadratic snapshots exist). Plot accuracy vs t. The epoch where proximity-only becomes near-perfect defines the **lock-in time**; the gap between P2-at-init and P0-at-init measures how much the lens terms explain.

**Metrics:** per-neuron top-1 accuracy vs P0 and chance; stratified by ODT level (leaves vs root vs internal); confusion matrices (spec V9); survival analysis of label switches.

### B3. Mechanism validation (justifies the score terms)

- **Second-order probes (H9/H10):** at init and early checkpoints, finite-difference Hessian-vector probes: perturb \(W_2[j,:]\) (or a first-layer competitor) along directions that map to \(\pm u_{leaf}\) in effective-gate space, measure the induced change in \(\nabla_{W_1[m,:]}\mathcal{L}\cdot u_{ancestor}\). Produce the sign-pattern heatmap (spec V7) and compare against DLGN Theorems 1–2 predictions. Agreement here is the empirical core of any ReLU theorem.
- **Frozen-layer ablation (H11):** freeze layers 2–3 at init; expect first-layer alignment to stall or lose hierarchy ordering.

## Candidate theorems this setup is designed to elicit

- **CT1 (leaf-first first-order dynamics):** for orthogonal-to-ODT init on the sphere, the expected first-layer gradient has components only along leaf directions, weighted by path-coefficient sums — ReLU analog of the DLGN saddle \(W^*\).
- **CT2 (path-weighted hierarchy-seeking lens):** the DLGN Theorem 1 inequality holds for ReLU cross-neuron Hessian blocks with each term multiplied by \(\mathrm{sign}(a_\pi)\) of the connecting path — verified/falsified by B3 heatmaps.
- **CT3 (lock-in / commitment):** there exists an early epoch \(t^*\) (empirically: around root capture) after which the eventual label of aligned neurons is determined; before \(t^*\) init-proximity is weakly informative and the lens terms carry the predictive signal.
- **CT4 (winner-take-most competition):** per (node, sign) at most a few first-layer neurons grow norm; competition is mediated by residue shrinkage, and the SO term with lens-overlap weighting captures it.

## Implementation mapping (when we build)

- `src/layer_lenses/relu_analysis.py`: add effective-gate computation for deeper layers, per-neuron trajectory extraction (signed cos over checkpoints), lock-in epoch detection. Existing `first_layer_odt_alignment`, ablation and leaf-activity tools are reused as-is.
- New module `src/layer_lenses/relu_predictor.py`: label extraction + data-invariance filter, P0–P3 scores, \(c_{SO}/c_{HS}\) grid, evaluation/confusion-matrix helpers.
- New module or extension for second-order probes (finite-difference HVPs on a checkpoint).
- Multi-seed runner extending `run_single_relu_seed` (grid over init seeds x dataset seeds); artifacts under `results/`.
- Track A cell experiments live in a new notebook first (per repo policy), with stable logic migrated to `src/`.
- Config choices (D, widths, WD, \(\tau\), seed counts) go to `configs/` per reproducibility rules.

## Suggested sequencing

1. Track B1 labelling harness + trajectories (cheap, uses existing runs) — establishes the benchmark and lock-in phenomenology.
2. Track A cell census (small models, fast) — the direct single-path generalization; likely the clearest theorem material.
3. B2 predictor ladder P0/P3, then effective gates + P1/P2.
4. B3 second-order probes — connects empirics back to provable statements.

## Decision points to settle before coding

- Primary depth: D=4 (cleaner, matches the paper's 96-solution setting, smaller label space) vs current D=5 scratch setting. Recommendation: D=4 primary, D=5 replication.
- Unaligned threshold \(\tau\) and whether "unaligned" is a predicted class or excluded from the benchmark (paper excludes non-convergent runs; recommend excluding unaligned neurons from accuracy but reporting their fraction).
- Weight decay: current scratch uses 0.01; run a WD=0 arm since WD may itself do the alignment cleanup and confound the lens story.

## Pitfalls carried over from the research spec

Sign asymmetry of ReLU (use signed cos, not \(|\cos|\)), permutation identity preserved via shared init seeds, linear-region explosion handled by active-set averages only, saddle/failed seeds analyzed separately rather than dropped silently.