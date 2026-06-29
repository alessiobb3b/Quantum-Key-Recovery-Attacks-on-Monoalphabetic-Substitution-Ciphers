# Corpus-Driven Oracle Construction for Quantum Key-Recovery Attacks on Monoalphabetic Substitution Ciphers

> **Di Santo, A. & Lanziani, G. (2026)**

---

## Overview

This repository contains the full experimental framework accompanying the paper. The toolkit implements a complete pipeline for:

1. Normalising Renaissance Italian plaintext corpora to a unified 21-letter alphabet
2. Building character $n$-gram language models with add-$\alpha$ smoothing
3. Estimating the marked key fraction `p_good` — the fraction of substitution-cipher permutations whose decryption satisfies a score-anchored plausibility threshold
4. Deriving analytical Grover oracle costs and classical exhaustive-search baselines
5. Running exact Grover statevector simulations over fully enumerated key spaces (k ≤ 8)
6. Running QUBO-style permutation annealing
7. Projecting oracle costs analytically across alphabet sizes k = 7–21
8. Reproducing all publication figures from saved results

The central finding is that `p_good`, measured empirically from character trigram language models under the **S_max/τ threshold**, varies genuinely with both the stringency parameter τ and the cipher length L. The Boyer formula provides a reliable near-optimal stopping criterion across all tested configurations.

---

## Corpora

The framework was applied to four Renaissance Italian works:

| Work | Author | Genre |
|---|---|---|
| *Il Principe* | Machiavelli | Political prose |
| *Il Cortegiano* | Castiglione | Courtly dialogue |
| *I Ricordi* | Guicciardini | Aphoristic reflection |
| *Orlando Furioso* | Ariosto | Epic poetry |

---

## Requirements

```
python >= 3.9
numpy
matplotlib
```

Install dependencies:

```bash
pip install numpy matplotlib
```

No additional quantum simulation libraries are required. The statevector simulation uses NumPy dense linear algebra over the full k!-element permutation space.

---

## Oracle Threshold

All Grover commands use the **sign-consistent S_max/τ threshold**:

```
threshold = S_max / τ
```

where `S_max = max over all evaluated permutations of S(π)` and `τ ∈ (0, 1)` is the stringency parameter. A key π is *marked* (considered plausible) if `S(π) ≥ S_max / τ`.

**Why this form, not a quantile?**

Trigram log-probability scores are negative real numbers. For `S_max < 0` and `τ < 1`, the quantity `S_max / τ` is more negative than `S_max`, placing the threshold strictly below the best observed score — so a non-trivial fraction of keys is marked. Higher τ → threshold closer to S_max (stricter, fewer marked keys). Lower τ → threshold further below (more permissive, more marked keys).

Using `np.quantile(scores, τ)` as a threshold instead fixes `p_good ≈ 1 − τ` by construction, removing all dependence on L and the language statistics.

---

## Pipeline

The tool is a single-file CLI with eight subcommands. Run them in order:

```
normalize → build-model → classical → grover-sweep → quantum-grover → qubo → grover-project → figures
```

---

### Step 0 — Normalize

Converts raw UTF-8 text files to the Renaissance 21-letter working alphabet. The following historically grounded orthographic substitutions are applied before stripping all non-letter characters:

| Raw | Mapped | Rationale |
|---|---|---|
| `J` | `I` | Renaissance i/j digraph |
| `V` | `U` | Renaissance u/v digraph |
| `K` | *(removed)* | Absent from standard Italian |
| `W` | *(removed)* | Absent from standard Italian |
| `Y` | *(removed)* | Absent from standard Italian |

All diacritics are stripped via Unicode NFD decomposition prior to mapping. The result is a 21-letter alphabet: `A,B,C,D,E,F,G,H,I,L,M,N,O,P,Q,R,S,T,U,X,Z` (including `Q` and `X`, which survive normally since they are not removed by the K/W/Y filter).

```bash
python renaissance_cipher_suite.py normalize \
  --in_files Il_Principe.txt Il_libro_del_Cortegiano.txt \
             Orlando_Furioso.txt Ricordi__Guicciardini__Serie_seconda.txt \
  --out_dir corpus/ \
  --merge merged_renaissance.txt
```

**Outputs** (written to `--out_dir`):

| File | Description |
|---|---|
| `{stem}_norm.txt` | Normalised version of each input file |
| `merged_renaissance.txt` | Concatenation of all normalised texts |

---

### Step 1 — Build Language Model

Trains unigram, bigram, and trigram character models with add-α smoothing from one or more normalised text files. All models are serialised to a single JSON file for use by downstream commands.

```bash
python renaissance_cipher_suite.py build-model \
  --corpus corpus/merged_renaissance.txt \
  --out models_renaissance.json \
  --max_order 3 \
  --smoothing 0.001
```

| Argument | Default | Description |
|---|---|---|
| `--max_order` | `3` | Highest n-gram order to build (1–3) |
| `--min_count` | `1` | Minimum count for an n-gram to be retained |
| `--smoothing` | `0.001` | Add-α smoothing constant applied to all counts |

**Scoring.** At inference time the scorer uses a backoff chain: trigram → bigram → unigram. This is the `score_text()` function called by all downstream commands. Scores are in log₁₀ units and are always negative; higher (less negative) values indicate more plausible decryptions.

**Output:** a single `models_renaissance.json` file containing:

```json
{
  "models": { "1": {...}, "2": {...}, "3": {...} },
  "alphabet": "ABCDEFGHILMNOPQRSTUXZ",
  "normalize": "renaissance_21",
  "metadata": { "sources": [...], "max_order": 3, "smoothing": 0.001 }
}
```

`"alphabet"` is stored in alphabetical order (21 letters). Several downstream commands (Steps 4–6) restrict this to a smaller k-letter subset — see **"Frequency-ranked sub-alphabet selection"** under Key Concepts for how that subset is chosen.

---

### Step 2 — Classical Baselines

Runs hill climbing and/or simulated annealing key-recovery attacks on a synthesised ciphertext.

```bash
python renaissance_cipher_suite.py classical \
  --models models_renaissance.json \
  --plaintext corpus/merged_renaissance.txt \
  --synthesize \
  --out_dir results/baselines/ \
  --method both \
  --seed 1337
```

| Algorithm | Implementation details |
|---|---|
| **Hill climbing** | 20 independent restarts × 2 000 swap steps; accepts only score-improving moves |
| **Simulated annealing** | 10 000 steps; geometric cooling `T(t) = T₀ × (T_end/T₀)^(t/(steps−1))`; `T₀ = 3.0`, `T_end = 0.1`; Metropolis acceptance |

**Outputs** (written to `--out_dir`):

| File | Description |
|---|---|
| `results.csv` | Per-method: score, runtime (s), key accuracy |
| `best_decryptions/hill_decryption.txt` | Best plaintext recovered by hill climbing |
| `best_decryptions/hill_key.json` | Corresponding key mapping |
| `best_decryptions/anneal_decryption.txt` | Best plaintext recovered by simulated annealing |
| `best_decryptions/anneal_key.json` | Corresponding key mapping |
| `plots/runtime_vs_score.png` | Bar chart of runtime per method |

---

### Step 3 — Grover Oracle Sweep

Monte Carlo estimation of `p_good` and analytical Grover oracle costs across the full **21-letter alphabet**. For each combination of cipher length L and stringency τ:

1. Draw a ciphertext of length L under a random key.
2. Sample `--score_samples` random permutations; estimate `S_max` as the maximum observed score, augmented by the simulated annealing best to reduce underestimation bias.
3. Compute `threshold = S_max / τ`.
4. Sample `--pgood_samples` additional permutations; count those with `S(π) ≥ threshold` to estimate `p_good`.
5. Apply the Boyer formula: `r* = floor(π / (4·arcsin(√p_good)) − 0.5)`, `N_oracle = 2r*+1`.

```bash
python renaissance_cipher_suite.py grover-sweep \
  --models models_renaissance.json \
  --plaintext corpus/merged_renaissance.txt \
  --alphabet_size 21 \
  --cipher_lengths 200,400,600,800,1000 \
  --tau_list 0.90,0.92,0.95,0.98,0.99,0.995 \
  --out_dir results/grover/ \
  --seed 42 \
  --score_samples 5000 \
  --pgood_samples 20000
```

| Argument | Default | Description |
|---|---|---|
| `--alphabet_size` | required | Working alphabet size; must be ≤ model alphabet size (21). For this command the alphabet is taken in alphabetical order, since the main analytical sweep is expected to use the full 21-letter alphabet rather than a reduced subset. |
| `--cipher_lengths` | required | Comma-separated list of L values |
| `--tau_list` | required | Comma-separated τ values ∈ (0, 1) |
| `--score_samples` | `5000` | Permutations used to estimate S_max |
| `--pgood_samples` | `20000` | Permutations used to estimate p_good |
| `--seed` | `42` | RNG seed for reproducibility |

**Outputs** (written to `--out_dir`):

| File | Description |
|---|---|
| `grover_sweep_results.csv` | Full results table: one row per (L, τ) pair |
| `plots/score_hist_L{L}.png` | Score distribution with S_max/τ threshold bands |
| `plots/pgood_vs_tau_L{L}.png` | p_good vs τ at each cipher length |
| `plots/fig_pgood_vs_tau_k{alphabet_size}.png` | Combined p_good vs τ for all L (filename reflects the actual `--alphabet_size` used, e.g. `fig_pgood_vs_tau_k21.png`) |

**CSV columns:**

| Column | Description |
|---|---|
| `alphabet_size` | Alphabet size actually used (≤ requested value) |
| `cipher_length` | L |
| `tau` | Stringency parameter τ |
| `tau_threshold` | Absolute threshold S_max/τ |
| `S_max_est` | Estimated maximum score over all evaluated permutations |
| `p_good` | Estimated marked key fraction |
| `grover_iters_opt` | Boyer optimal iteration count r* |
| `grover_oracle_calls` | Oracle calls 2r*+1 |
| `grover_success_prob` | Success probability at r* |
| `classical_expected_trials` | 1/p_good |
| `speedup_vs_random` | Classical trials / oracle calls |

---

### Step 4 — QUBO Annealing

Permutation annealing under the objective `E(π) = −S(π) + λ·C(π)`, where `C(π)` penalises deviations from a valid permutation. The annealer operates directly on a permutation index array using random transpositions as proposals.

**Reduced-alphabet selection.** The k-letter alphabet used here is the **k most frequent letters** of the corpus, ranked by unigram probability under the trained language model — not an alphabetically-sorted prefix of `lm["alphabet"]`. See *"Frequency-ranked sub-alphabet selection"* under Key Concepts.

```bash
python renaissance_cipher_suite.py qubo \
  --models models_renaissance.json \
  --plaintext corpus/merged_renaissance.txt \
  --alphabet_size 7 \
  --cipher_length 400 \
  --out_dir results/qubo/ \
  --seed 7
```

| Parameter | Value | Description |
|---|---|---|
| Steps | 25 000 | Total annealing sweeps |
| `T₀` | 3.0 | Initial temperature |
| `T_end` | 0.05 | Final temperature |
| Schedule | Geometric | `T(t) = T₀ × (T_end/T₀)^(t/(steps−1))` |

**Outputs** (written to `--out_dir`):

| File | Description |
|---|---|
| `best_mapping.json` | Best substitution mapping found |
| `best_decryption.txt` | Plaintext recovered under the best mapping |
| `summary.csv` | Alphabet size, cipher length, final energy, key accuracy, and the actual `alphabet_letters` string used (so runs are auditable) |
| `energy_curve.png` | Energy trajectory over all 25 000 annealing steps |

---

### Step 5 — Exact Quantum Statevector Simulation

The primary validation phase. All k! permutations are enumerated exactly and scored; a NumPy statevector simulation runs for `--max_iters` Grover iterations, recording `P_t` at each step.

**Oracle:** key π is marked if `S(π) ≥ S_max / τ`, where `S_max` is computed exactly from the full enumeration.

**Reduced-alphabet selection.** As in Step 4, the k-letter alphabet is the **k most frequent letters** of the corpus, ranked by unigram probability — not an alphabetically-sorted prefix.

**Validation:** every simulated `t* = argmax P_t` is cross-checked against the analytical formula `P_t = sin²((2t+1)·θ)`. The two must agree; a mismatch is flagged as `MISMATCH` in the output and indicates a numerical issue.

**Recommended:** `--alphabet_size ≤ 8` (8! = 40 320 states; all fit in RAM with a small NumPy footprint).

```bash
python renaissance_cipher_suite.py quantum-grover \
  --models models_renaissance.json \
  --plaintext corpus/merged_renaissance.txt \
  --alphabet_size 7 \
  --cipher_lengths 200,400,600,800,1000 \
  --tau_list 0.90,0.92,0.95,0.98,0.99,0.995 \
  --max_iters 50 \
  --out_dir results/quantum/ \
  --seed 42
```

| Argument | Default | Description |
|---|---|---|
| `--alphabet_size` | required | k; the simulation runs over all k! permutations of the k most frequent letters |
| `--cipher_lengths` | required | Comma-separated L values |
| `--tau_list` | required | Comma-separated τ values |
| `--max_iters` | `50` | Number of Grover iterations to simulate |
| `--seed` | `42` | RNG seed |

**Outputs** (written to `--out_dir`):

| File | Description |
|---|---|
| `exact_k{k}.csv` | Full results: one row per (L, τ) pair |
| `plots/envelope_k{k}_L{L}_tau{τ}.png` | Probability envelope: sim vs analytical |
| `plots/pgood_vs_L_k{k}.png` | p_good vs L for all τ (confirms length-dependence) |
| `plots/speedup_k{k}.png` | Boyer-formula quantum speedup vs τ |

**CSV columns include:**

| Column | Description |
|---|---|
| `k` | Alphabet size |
| `alphabet_letters` | The actual k most frequent letters used (auditability) |
| `S_max` | Exact maximum score over all k! permutations |
| `S_true` | Score of the correct key π* |
| `S_max_gap` | S_true − S_max (→ 0 as L increases) |
| `threshold` | S_max / τ |
| `M_marked` | Number of marked permutations |
| `p_good` | Exact marked fraction M / k! |
| `r_star` | Boyer stopping iteration |
| `P_r_star` | Success probability at r* |
| `t_star_sim` | Simulated argmax P_t |
| `t_star_analytic` | Analytical argmax P_t |
| `sim_matches_analytic` | True/False — simulation integrity check |
| `speedup_boyer` | Classical trials / (2r*+1) — operationally relevant |
| `speedup_tstar` | Classical trials / (2t*+1) — global peak cost |

---

### Step 6 — Analytical Scalability Projection

Traces how Boyer oracle call counts evolve across alphabet sizes **k = 7–21** using Monte Carlo `p_good` estimates and the S_max/τ threshold. Results are indicative for k > 8 because `S_max` is estimated from a finite sample (rather than exact enumeration) at those scales.

**Reduced-alphabet selection.** As in Steps 4–5, each alphabet size k restricts the corpus to the **k most frequent letters**, ranked by unigram probability.

**S_max estimation.** As in Step 3, `S_max` is estimated from a reference sample of up to 5 000 permutations, **augmented by the best score returned by a 5 000-step simulated-annealing run** — the same procedure used in `grover-sweep`, applied here per-k.

**`--k_max` bound.** The corpus alphabet has exactly 21 distinct letters; `--k_max` cannot exceed this. Passing a larger value raises an error rather than silently duplicating the k=21 row under a higher-k label (Python string slicing does not error past the end of a string, so this guard is enforced explicitly in code).

```bash
python renaissance_cipher_suite.py grover-project \
  --models models_renaissance.json \
  --plaintext corpus/merged_renaissance.txt \
  --k_min 7 --k_max 21 \
  --cipher_length 400 \
  --tau_list 0.90,0.95,0.98,0.995 \
  --pgood_samples 20000 \
  --out_dir results/projection/ \
  --seed 42
```

| Argument | Default | Description |
|---|---|---|
| `--k_min` | `7` | Smallest alphabet size in the sweep |
| `--k_max` | `21` | Largest alphabet size in the sweep; must not exceed the corpus's true alphabet size (21) |
| `--cipher_length` | `400` | Ciphertext length used at every k |
| `--tau_list` | required | Comma-separated τ values |
| `--pgood_samples` | `20000` | Permutations sampled to estimate p_good at each (k, τ) |

**Outputs** (written to `--out_dir`):

| File | Description |
|---|---|
| `grover_project_results.csv` | One row per (k, τ) pair, including the `alphabet_letters` actually used at each k |
| `plots/combined_scaling_k7_k21.png` | Three-panel: oracle calls, classical trials, speedup vs k |

---

### Step 7 — Figures

Generates publication-ready summary figures from previously saved result CSVs. All three source directories are optional.

```bash
python renaissance_cipher_suite.py figures \
  --sweep_dir results/grover/ \
  --quantum_dir results/quantum/ \
  --baselines_dir results/baselines/ \
  --out_dir results/figures/
```

**Outputs:**

| File | Description |
|---|---|
| `baselines_runtime_bar.png` | Bar chart of classical method runtimes |
| `fig_pgood_vs_tau_summary.png` | p_good vs τ for all cipher lengths (sweep results) |
| `grover_calls_vs_length_tau098.png` | Grover oracle calls vs L at τ ≈ 0.98 |
| `classical_trials_vs_length_tau098.png` | Classical expected trials vs L at τ ≈ 0.98 |
| `fig_pgood_vs_L_k{k}.png` | Exact p_good vs L for each simulated k |
| `fig_speedup_k{k}.png` | Boyer-formula speedup vs τ for each simulated k |

---

## Output Directory Structure

After running the full pipeline:

```
corpus/
├── Il_Principe_norm.txt
├── Il_libro_del_Cortegiano_norm.txt
├── Orlando_Furioso_norm.txt
├── Ricordi__Guicciardini__Serie_seconda_norm.txt
└── merged_renaissance.txt

models_renaissance.json

results/
├── baselines/
│   ├── results.csv
│   ├── synthetic_ciphertext.txt
│   ├── best_decryptions/
│   │   ├── hill_decryption.txt
│   │   ├── hill_key.json
│   │   ├── anneal_decryption.txt
│   │   └── anneal_key.json
│   └── plots/
│       └── runtime_vs_score.png
├── grover/
│   ├── grover_sweep_results.csv
│   └── plots/
│       ├── score_hist_L{200..1000}.png
│       ├── pgood_vs_tau_L{200..1000}.png
│       └── fig_pgood_vs_tau_k21.png
├── qubo/
│   ├── best_mapping.json
│   ├── best_decryption.txt
│   ├── summary.csv
│   └── energy_curve.png
├── quantum/
│   ├── exact_k7.csv
│   ├── exact_k8.csv
│   └── plots/
│       ├── envelope_k7_L{L}_tau{τ}.png
│       ├── pgood_vs_L_k7.png
│       ├── speedup_k7.png
│       └── (k8 equivalents)
├── projection/
│   ├── grover_project_results.csv
│   └── plots/
│       └── combined_scaling_k7_k21.png
└── figures/
    ├── baselines_runtime_bar.png
    ├── fig_pgood_vs_tau_summary.png
    ├── fig_pgood_vs_L_k7.png
    ├── fig_speedup_k7.png
    ├── grover_calls_vs_length_tau098.png
    └── classical_trials_vs_length_tau098.png
```

---

## Reproducibility

All commands accept a `--seed` argument. The seeds used in the paper are:

| Experiment | Seed |
|---|---|
| Classical baselines | `1337` |
| Grover sweep (21-letter) | `42` |
| Quantum statevector simulation (k=7, k=8) | `42` |
| QUBO annealing | `7` |
| Scalability projection | `42` |

All intermediate data is written to disk before figures are generated, so figures can be regenerated from saved CSVs without re-running experiments.

---

## Key Concepts

### Marked key fraction `p_good`

The fraction of all substitution keys π whose decryption satisfies the score-anchored plausibility criterion:

```
p_good = Pr[ S(π) ≥ S_max / τ ]
```

where `S(π)` is the trigram log-likelihood of the decryption, `S_max` is the highest score observed across all evaluated permutations (exact in the quantum-grover phase; estimated by sampling, augmented by simulated annealing, in the grover-sweep and grover-project phases), and `τ ∈ (0, 1)` controls stringency.

`p_good` varies with both `τ` and the cipher length `L` — as `L` increases, `S_max` grows roughly proportionally (scores are sums of L log-probability terms), while the score distribution concentrates (standard deviation ∝ √L), admitting a growing fraction of permutations above the threshold. This length-dependence is confirmed by exact enumeration at k=7 and k=8 in the `quantum-grover` phase.

### Frequency-ranked sub-alphabet selection

Whenever an experiment restricts the corpus to a k-letter subset (`qubo`, `quantum-grover`, and every k in the `grover-project` sweep), the k letters used are the **k most frequent letters in the corpus**, ranked by unigram log-probability under the trained language model — not an alphabetically-sorted prefix of `lm["alphabet"]`.

This matters: `lm["alphabet"]` is stored as `sorted(set(...))`, so a naive `lm["alphabet"][:7]` would pick whichever 7 letters happen to sort first alphabetically (e.g. `A,B,C,D,E,F,G`) regardless of how common they actually are in Renaissance Italian. Selecting by frequency instead ensures the reduced-alphabet experiments operate on a representative slice of the language's character distribution, consistent with how the full-alphabet results are interpreted. Every command that performs this selection records the actual letters used (`alphabet_letters` in the relevant CSV) so runs are auditable.

The `grover-sweep` command is the one exception: it is intended to run over the full 21-letter alphabet, so it truncates `lm["alphabet"]` directly (a no-op when `--alphabet_size` equals the full alphabet size).

### Grover oracle cost (Boyer formula)

```
θ = arcsin(√p_good)
r* = floor(π / (4θ) − 0.5)
N_oracle = 2r* + 1
```

`r*` identifies the first near-optimal probability peak; success probability at `r*` is at least 0.88 across all configurations tested in the paper. The global probability maximum `t*` (recorded in `exact_k{k}.csv` as `t_star_sim`) is reported for completeness; a practitioner should stop at `r*`, not `t*`.

### Classical expected trials

```
classical_trials = 1 / p_good
```

Expected number of uniformly random key draws required to land on a marked state. This is the weakest classical adversary and serves as the denominator for the quantum speedup ratio.

### Self-consistency: S_max convergence

In the `quantum-grover` phase, the CSV column `S_max_gap = S_true − S_max` tracks how close the correct key's score is to the globally best score. As L increases, this gap shrinks toward zero — at L ≥ 600 for k = 7, `S_max = S_true` exactly, meaning the oracle's highest-scoring permutation is the correct decryption.

---

## Changes from the Original Code

The key algorithmic change from the previous version is the replacement of the quantile-based threshold with the S_max/τ threshold throughout all three Grover commands:

| Command | Old threshold | New threshold |
|---|---|---|
| `grover-sweep` | `np.quantile(scores, τ)` | `S_max / τ` |
| `quantum-grover` | `np.quantile(all_scores, τ)` | `all_scores.max() / τ` |
| `grover-project` | `np.quantile(sample_scores, τ)` | `S_max_est / τ` |

**Further fixes applied to align the code with the alphabet-size correction (21 letters, not 23) and the paper's stated Methods (§3.4):**

| Issue | Old behavior | Fixed behavior |
|---|---|---|
| Sub-alphabet selection (`qubo`, `quantum-grover`, `grover-project`) | `lm["alphabet"][:k]` — an arbitrary alphabetically-sorted prefix (e.g. k=7 → `A,B,C,D,E,F,G`) | `frequency_ranked_alphabet(lm)[:k]` — the k most frequent letters by unigram probability, as specified in the paper's Methods §3.4 |
| `grover-project` S_max estimation | Reference-sample maximum only | Reference-sample maximum **augmented by a 5 000-step simulated-annealing best**, matching `grover-sweep` and the paper's description |
| `grover-project --k_max` | Default `23`; silently duplicated the k=21 row under k=22/23 labels (string slicing past the end of a 21-character string is a no-op, not an error) | Default `21`; raises `ValueError` if `--k_max` exceeds the corpus's true alphabet size |
| Output filenames | Hardcoded `fig_pgood_vs_tau_k23.png` / `combined_scaling_k7_k23.png` regardless of the alphabet size actually used | Filenames now reflect the true alphabet size (`fig_pgood_vs_tau_k{k}.png`, `combined_scaling_k7_k21.png`); the `figures` summary plot, which aggregates across heterogeneous runs, uses a size-agnostic name (`fig_pgood_vs_tau_summary.png`) |

Additional earlier changes (carried over):
- Normalisation: `V→U` (not `W→V`), `J→I`, remove `K,W,Y` — no `X→SS` substitution (not attested in the paper's §3.1)
- Default smoothing changed from `0.5` to `0.001` to match the paper
- `quantum-grover` reports two speedup columns (`speedup_boyer`, `speedup_tstar`) and logs the simulation integrity check result for every case
- `figures` subcommand accepts a `--quantum_dir` argument to generate p_good and speedup figures from exact simulation CSVs
- `grover-sweep` augments S_max estimation with a 5 000-step simulated annealing run to reduce the downward bias inherent in random sampling

---

## Extending the Framework

- **Higher-order models:** pass `--max_order 4` to `build-model`; the scorer backsoff automatically.
- **Different alphabets:** any `--alphabet_size` ≤ the model alphabet size (21) works for all commands; sizes below 21 select the k most frequent letters (see "Frequency-ranked sub-alphabet selection" above) for every command except `grover-sweep`.
- **Additional classical methods:** add a function following the `hill_climb` / `simulated_annealing` pattern inside `cmd_classical`.
- **Polyalphabetic ciphers:** replace `random_key` and `apply_key` with a Vigenère-style generator; the scoring and sweep logic is cipher-agnostic.
- **Quantum hardware:** `boyer_r_star(p_good)` returns `(r*, P_{r*})` directly; these can be passed to Qiskit, PennyLane, or Cirq to instantiate the oracle with the correct iteration count.

---

## Citation

If you use this code or data, please cite:

```
Di Santo, A. & Lanziani, G. (2026). Corpus-Driven Oracle Construction for
Quantum Key-Recovery Attacks on Monoalphabetic Substitution Ciphers.
```

---

## Contact

**Alessio Di Santo** — alessio.disanto@graduate.univaq.it  
*Department of Information Engineering, Computer Science, and Mathematics, University of L'Aquila*

**Gabriella Lanziani** — gabriella.lanziani@byteguardian.it  
*Independent Researcher*

---

## License

MIT License. See `LICENSE` for details.
