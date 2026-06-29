#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
renaissance_cipher_suite.py
===========================
One-stop toolkit for corpus normalisation, language-model building,
classical baselines, Grover oracle sweeps, exact quantum statevector
simulation, QUBO/annealing tests, and figure generation.

USAGE (examples)
----------------
# 0) Normalise raw texts to Renaissance 21-letter alphabet and merge
python renaissance_cipher_suite.py normalize \\
    --in_files Il_Principe.txt Il_libro_del_Cortegiano.txt \\
               Orlando_Furioso.txt Ricordi__Guicciardini__Serie_seconda.txt \\
    --out_dir corpus/ --merge merged_renaissance.txt

# 1) Build trigram language model from normalised corpus
python renaissance_cipher_suite.py build-model \\
    --corpus corpus/merged_renaissance.txt \\
    --out models_renaissance.json --max_order 3 --smoothing 0.001

# 2) Classical baselines (hill climbing + simulated annealing)
python renaissance_cipher_suite.py classical \\
    --models models_renaissance.json \\
    --plaintext corpus/merged_renaissance.txt --synthesize \\
    --out_dir results/baselines/ --seed 1337

# 3) Monte Carlo Grover sweep (21-letter alphabet, S_max/tau threshold)
python renaissance_cipher_suite.py grover-sweep \\
    --models models_renaissance.json \\
    --plaintext corpus/merged_renaissance.txt \\
    --alphabet_size 21 --cipher_lengths 200,400,600,800,1000 \\
    --tau_list 0.90,0.92,0.95,0.98,0.99,0.995 \\
    --out_dir results/grover/ --seed 42

# 4) QUBO permutation annealing on reduced alphabet (k most frequent letters)
python renaissance_cipher_suite.py qubo \\
    --models models_renaissance.json \\
    --plaintext corpus/merged_renaissance.txt \\
    --alphabet_size 7 --cipher_length 400 \\
    --out_dir results/qubo/ --seed 7

# 5) Exact quantum statevector simulation (k<=8 recommended; k! states)
python renaissance_cipher_suite.py quantum-grover \\
    --models models_renaissance.json \\
    --plaintext corpus/merged_renaissance.txt \\
    --alphabet_size 7 --cipher_lengths 200,400,600,800,1000 \\
    --tau_list 0.90,0.92,0.95,0.98,0.99,0.995 \\
    --max_iters 50 --out_dir results/quantum/ --seed 42

# 6) Analytical scalability projection across alphabet sizes k=7..21
#    (k_max cannot exceed the corpus alphabet's true size; see FIX #3 below)
python renaissance_cipher_suite.py grover-project \\
    --models models_renaissance.json \\
    --plaintext corpus/merged_renaissance.txt \\
    --k_min 7 --k_max 21 --cipher_length 400 \\
    --tau_list 0.90,0.95,0.98,0.995 \\
    --out_dir results/projection/ --seed 42

# 7) Publication figures from saved results
python renaissance_cipher_suite.py figures \\
    --sweep_dir results/grover/ --quantum_dir results/quantum/ \\
    --baselines_dir results/baselines/ --out_dir results/figures/

ORACLE THRESHOLD
----------------
All three Grover commands (grover-sweep, quantum-grover, grover-project)
use the sign-consistent threshold:

    threshold = S_max / tau

where S_max = max over all evaluated permutations of S(pi), and
tau in (0, 1) is the stringency parameter.

For log-probability scores S(pi) < 0 and tau < 1:
    S_max / tau < S_max     (threshold lies BELOW S_max)
meaning a non-trivial fraction of keys is marked.  Higher tau -> stricter
(threshold closer to S_max); lower tau -> more permissive.

This is the sign-consistent analogue of tau * S_max for negative scores,
and reduces to tau * S_max when S_max > 0.

DO NOT use np.quantile(scores, tau) as a threshold — this makes
p_good ~= 1 - tau by construction, removing all dependence on L and
the language statistics.

REDUCED-ALPHABET SELECTION  [FIX #1]
-------------------------------------
Whenever an experiment restricts the corpus to a k-letter subset of the
full alphabet (k=7 and k=8 exact simulation, k=7 QUBO/k=8 QUBO, and every
k in the k_min..k_max scalability projection sweep), the k letters used
are the k MOST FREQUENT letters in the corpus, ranked by unigram
probability under the trained language model — NOT the first k letters
of lm["alphabet"] in alphabetical order. The paper's Methods Sec. 3.4
explicitly specifies "the k most frequent letters"; earlier versions of
this script took an alphabetically-sorted prefix instead, which silently
picked an arbitrary, non-representative subset (e.g. k=7 -> "ABCDEFG").

MISSING SA AUGMENTATION IN THE PROJECTION  [FIX #2]
-----------------------------------------------------
Sec. 3.4 specifies that S_max in the scalability projection is estimated
from a reference sample "augmented by the simulated annealing best," the
same procedure used in grover-sweep. This is now applied in
cmd_grover_project as well (previously it used only the raw reference
sample maximum).

K_MAX VALIDATION  [FIX #3]
-----------------------------
The corpus alphabet produced by normalize_renaissance() has exactly 21
distinct letters (A,B,C,D,E,F,G,H,I,L,M,N,O,P,Q,R,S,T,U,X,Z). Because
Python string slicing does not raise an error past the end of a string,
requesting k > 21 previously failed silently: full_alph[:23] on a
21-character string just returns the same 21 letters again, duplicating
the k=21 row under a k=22 or k=23 label. cmd_grover_project now validates
k_max against the actual alphabet length and raises a clear error instead
of producing silently duplicated data. The CLI default for --k_max is
also changed from 23 to 21 to match the true corpus alphabet size.
"""

import argparse
import csv
import itertools
import json
import math
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def normalize_renaissance(text: str) -> str:
    """
    Normalise raw UTF-8 text to the 21-letter Renaissance Italian alphabet.

    Mapping applied (historically grounded):
        J  -> I      (Renaissance i/j digraph)
        V  -> U      (Renaissance u/v digraph)
        K  -> remove (absent from standard Italian)
        W  -> remove (absent from standard Italian)
        Y  -> remove (absent from standard Italian)

    All diacritics are stripped via Unicode NFD decomposition.
    Remaining non-letter characters are removed.
    Result is uppercase A-Z minus J, K, V, W, Y (21 letters, including Q, X).
    """
    import unicodedata
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.upper()
    text = text.replace("J", "I").replace("V", "U")
    text = re.sub(r"[KWY]", "", text)
    text = re.sub(r"[^A-Z]", "", text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Language model
# ─────────────────────────────────────────────────────────────────────────────

def ngrams(seq, n):
    for i in range(len(seq) - n + 1):
        yield seq[i:i + n]


def build_models(texts, max_order=3, min_count=1, smoothing=0.001):
    """Build unigram/bigram/trigram models with add-alpha smoothing."""
    counts = [defaultdict(int) for _ in range(max_order + 1)]
    for t in texts:
        for n in range(1, max_order + 1):
            for g in ngrams(t, n):
                counts[n][g] += 1
    if min_count > 1:
        for n in range(1, max_order + 1):
            counts[n] = defaultdict(int,
                {g: c for g, c in counts[n].items() if c >= min_count})

    alphabet = sorted(set("".join(texts)))
    V = len(alphabet)
    models = {}

    # Unigram
    total = sum(counts[1].values()) + smoothing * V
    uni_logp = {a: math.log10((counts[1].get(a, 0) + smoothing) / total)
                for a in alphabet}
    models["1"] = {"order": 1, "logp": uni_logp}

    # Bigram / trigram
    for n in range(2, max_order + 1):
        cond = {}
        for g, c in counts[n].items():
            ctx = g[:-1]
            cond.setdefault(ctx, {})[g[-1]] = c
        cond_logp = {}
        for ctx, d in cond.items():
            Z = sum(d.values()) + smoothing * V
            cond_logp[ctx] = {a: math.log10((d.get(a, 0) + smoothing) / Z)
                              for a in alphabet}
        models[str(n)] = {"order": n, "logp": cond_logp}

    return models, "".join(alphabet)


def score_text(text, lm):
    """
    Score text under the trigram model (log10) with bigram/unigram backoff.
    Returns a negative float; higher (less negative) = more plausible.
    """
    models = lm["models"]
    total = 0.0
    for i, ch in enumerate(text):
        if "3" in models and i >= 2:
            row = models["3"]["logp"].get(text[i - 2:i])
            if row is not None:
                total += row.get(ch, models["1"]["logp"].get(ch, -20.0))
                continue
        if "2" in models and i >= 1:
            row = models["2"]["logp"].get(text[i - 1:i])
            if row is not None:
                total += row.get(ch, models["1"]["logp"].get(ch, -20.0))
                continue
        total += models["1"]["logp"].get(ch, -20.0)
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Key utilities
# ─────────────────────────────────────────────────────────────────────────────

def random_key(alphabet):
    p = list(alphabet)
    random.shuffle(p)
    return dict(zip(alphabet, p))


def invert_key(key):
    return {v: k for k, v in key.items()}


def apply_key(text, key):
    return "".join(key.get(c, c) for c in text)


# ─────────────────────────────────────────────────────────────────────────────
# FIX #1: Frequency-ranked alphabet subsetting
# ─────────────────────────────────────────────────────────────────────────────

def frequency_ranked_alphabet(lm):
    """
    Return the model's full alphabet re-ordered by descending unigram
    frequency (most frequent letter first), based on the trained unigram
    log-probabilities in lm["models"]["1"]["logp"].

    This is the ranking that must be used whenever an experiment restricts
    the corpus to "the k most frequent letters" (paper Methods Sec. 3.4),
    e.g. via alphabet = frequency_ranked_alphabet(lm)[:k].

    Falls back to the raw (alphabetically-sorted) lm["alphabet"] string if
    no unigram model is present, with a warning, since frequency ranking
    is then impossible.
    """
    full_alphabet = lm["alphabet"]
    uni = lm.get("models", {}).get("1", {}).get("logp")
    if not uni:
        print("  WARNING: no unigram model found; falling back to "
              "alphabetical alphabet order (frequency ranking unavailable).")
        return full_alphabet
    # Unigram entries are log10 probabilities: higher (less negative) means
    # more frequent. Sort descending by log-probability.
    ranked = sorted(full_alphabet, key=lambda c: uni.get(c, -1e9), reverse=True)
    return "".join(ranked)


# ─────────────────────────────────────────────────────────────────────────────
# Grover helpers
# ─────────────────────────────────────────────────────────────────────────────

def boyer_r_star(p_good):
    """
    Optimal Boyer iteration count and success probability.
    r* = floor(pi / (4*theta) - 0.5),  theta = arcsin(sqrt(p_good)).
    Returns (r_star, P_r_star).
    """
    if p_good <= 0.0:
        return 0, 0.0
    theta = math.asin(math.sqrt(min(p_good, 1.0)))
    r = max(0, int(math.floor(math.pi / (4.0 * theta) - 0.5)))
    P = math.sin((2 * r + 1) * theta) ** 2
    return r, P


def smax_threshold(scores_array, tau):
    """
    Compute the S_max/tau threshold from an array of scores.
    scores_array : numpy array of log10 scores (negative floats)
    tau          : stringency parameter in (0, 1)
    Returns a scalar threshold T such that keys with S(pi) >= T are marked.

    For S_max < 0 and tau < 1:  T = S_max / tau < S_max  (correct, below S_max).
    Higher tau -> T closer to S_max (stricter).
    Lower  tau -> T further below  S_max (more permissive).
    """
    S_max = float(scores_array.max())
    return S_max / tau


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: normalize
# ─────────────────────────────────────────────────────────────────────────────

def cmd_normalize(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_parts = []
    for path in args.in_files:
        raw = Path(path).read_text(encoding="utf-8", errors="ignore")
        norm = normalize_renaissance(raw)
        stem = Path(path).stem
        out_path = out_dir / f"{stem}_norm.txt"
        out_path.write_text(norm, encoding="utf-8")
        merged_parts.append(norm)
        print(f"  {stem}: {len(norm):,} letters -> {out_path}")
    if args.merge:
        merged = "".join(merged_parts)
        mp = out_dir / args.merge
        mp.write_text(merged, encoding="utf-8")
        print(f"  Merged: {len(merged):,} letters -> {mp}")
    chars = sorted(set("".join(merged_parts)))
    print(f"  Alphabet ({len(chars)}): {''.join(chars)}")


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: build-model
# ─────────────────────────────────────────────────────────────────────────────

def cmd_build_model(args):
    texts = []
    for p in args.corpus:
        texts.append(Path(p).read_text(encoding="utf-8", errors="ignore"))
    models, alphabet = build_models(
        texts,
        max_order=args.max_order,
        min_count=args.min_count,
        smoothing=args.smoothing,
    )
    payload = {
        "models": models,
        "alphabet": alphabet,
        "normalize": "renaissance_21",
        "metadata": {
            "sources": args.corpus,
            "max_order": args.max_order,
            "smoothing": args.smoothing,
        },
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"Model written -> {args.out}  (alphabet: {alphabet})")


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: classical
# ─────────────────────────────────────────────────────────────────────────────

def hill_climb(ct, lm, restarts=20, steps=2000, seed=None):
    rng = random.Random(seed)
    alphabet = lm["alphabet"]
    best_key = random_key(alphabet)
    best_score = score_text(apply_key(ct, best_key), lm)
    for _ in range(restarts):
        key = random_key(alphabet)
        score = score_text(apply_key(ct, key), lm)
        for __ in range(steps):
            a, b = rng.sample(alphabet, 2)
            new_key = dict(key)
            new_key[a], new_key[b] = new_key[b], new_key[a]
            ns = score_text(apply_key(ct, new_key), lm)
            if ns > score:
                key, score = new_key, ns
        if score > best_score:
            best_key, best_score = key, score
    return best_key, best_score


def simulated_annealing(ct, lm, steps=10000, T0=3.0, Tend=0.1, seed=None):
    rng = random.Random(seed)
    alphabet = lm["alphabet"]
    key = random_key(alphabet)
    score = score_text(apply_key(ct, key), lm)
    best_key, best_score = dict(key), score
    for t in range(steps):
        T = T0 * (Tend / T0) ** (t / (steps - 1))
        a, b = rng.sample(alphabet, 2)
        new_key = dict(key)
        new_key[a], new_key[b] = new_key[b], new_key[a]
        ns = score_text(apply_key(ct, new_key), lm)
        if ns > score or rng.random() < math.exp((ns - score) / T):
            key, score = new_key, ns
        if score > best_score:
            best_key, best_score = dict(key), score
    return best_key, best_score


def key_accuracy(true_key, found_key):
    correct = sum(1 for k in true_key if true_key[k] == found_key.get(k))
    return correct / len(true_key)


def cmd_classical(args):
    import time
    lm = json.loads(Path(args.models).read_text(encoding="utf-8"))
    alphabet = lm["alphabet"]
    random.seed(args.seed)
    pt = Path(args.plaintext).read_text(encoding="utf-8", errors="ignore")
    pt = "".join(c for c in pt if c in alphabet)

    if args.synthesize:
        true_key = random_key(alphabet)
        ct = apply_key(pt, invert_key(true_key))
    else:
        ct = Path(args.ciphertext).read_text(encoding="utf-8", errors="ignore")
        ct = "".join(c for c in ct if c in alphabet)
        true_key = None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)
    (out_dir / "best_decryptions").mkdir(exist_ok=True)

    if args.synthesize:
        (out_dir / "synthetic_ciphertext.txt").write_text(ct, encoding="utf-8")

    rows = []
    methods_to_run = (["hill", "anneal"] if args.method == "both"
                      else [args.method])

    for method in methods_to_run:
        t0 = time.perf_counter()
        if method == "hill":
            found_key, found_score = hill_climb(ct, lm, seed=args.seed)
        else:
            found_key, found_score = simulated_annealing(ct, lm, seed=args.seed)
        elapsed = time.perf_counter() - t0
        acc = key_accuracy(true_key, found_key) if true_key else None
        decryption = apply_key(ct, found_key)
        stem = "hill" if method == "hill" else "anneal"
        (out_dir / "best_decryptions" / f"{stem}_decryption.txt").write_text(
            decryption, encoding="utf-8")
        (out_dir / "best_decryptions" / f"{stem}_key.json").write_text(
            json.dumps(found_key, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append({"method": method, "score": round(found_score, 4),
                     "runtime_sec": round(elapsed, 3),
                     "key_accuracy": round(acc, 4) if acc is not None else "N/A"})
        print(f"  {method}: score={found_score:.3f}  time={elapsed:.2f}s  "
              f"acc={acc:.3f}" if acc is not None else
              f"  {method}: score={found_score:.3f}  time={elapsed:.2f}s")

    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "score",
                                          "runtime_sec", "key_accuracy"])
        w.writeheader(); w.writerows(rows)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([r["method"] for r in rows], [r["runtime_sec"] for r in rows],
           color=["#2166ac", "#d6604d"])
    ax.set_xlabel("Method"); ax.set_ylabel("Runtime (s)")
    ax.set_title("Classical baselines: runtime")
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "runtime_vs_score.png", dpi=150)
    plt.close(fig)
    print(f"Results -> {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: grover-sweep  (Monte Carlo, S_max/tau threshold)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_grover_sweep(args):
    """
    Monte Carlo Grover oracle sweep using the S_max/tau threshold.

    For each (cipher_length L, stringency tau):
      1. Draw a ciphertext of length L under a random key.
      2. Sample --score_samples permutations and score each; estimate S_max
         as the maximum observed score (augmented by simulated annealing best).
      3. Compute threshold = S_max / tau.
      4. Sample --pgood_samples additional permutations; count those with
         S(pi) >= threshold to estimate p_good.
      5. Compute Boyer oracle cost: r* = floor(pi/(4*arcsin(sqrt(p_good)))-0.5),
         N_oracle = 2r*+1.

    Note: --alphabet_size here is expected to equal the corpus's full
    alphabet size (21) for the main analytical sweep; the function simply
    truncates lm["alphabet"], which is fine ONLY when alphabet_size equals
    (or exceeds, harmlessly) the full alphabet length. For any sub-alphabet
    experiment (alphabet_size < full size), use frequency_ranked_alphabet()
    as done in cmd_qubo / cmd_quantum_grover / cmd_grover_project.
    """
    lm = json.loads(Path(args.models).read_text(encoding="utf-8"))
    full_alphabet = lm["alphabet"]
    alphabet = full_alphabet[:args.alphabet_size]
    random.seed(args.seed); np.random.seed(args.seed)

    pt_full = Path(args.plaintext).read_text(encoding="utf-8", errors="ignore")
    pt_full = "".join(c for c in pt_full if c in alphabet)

    cipher_lengths = [int(x) for x in args.cipher_lengths.split(",")]
    tau_list = [float(x) for x in args.tau_list.split(",")]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    rows = []
    for L in cipher_lengths:
        if len(pt_full) < L:
            print(f"  L={L}: corpus too short, skipping"); continue
        pt = pt_full[:L]
        true_key = random_key(alphabet)
        ct = apply_key(pt, invert_key(true_key))

        # Score reference sample for S_max estimation
        ref_scores = np.array([
            score_text(apply_key(ct, random_key(alphabet)), lm)
            for _ in range(args.score_samples)
        ], dtype=np.float64)
        # Augment S_max with SA best (reduces underestimation bias)
        sa_key, sa_score = simulated_annealing(ct, lm, steps=5000, seed=args.seed)
        S_max = max(float(ref_scores.max()), sa_score)
        S_mean = float(ref_scores.mean())
        S_std  = float(ref_scores.std())

        print(f"\n  L={L}  S_max={S_max:.2f}  mean={S_mean:.2f}  std={S_std:.2f}")

        for tau in tau_list:
            threshold = S_max / tau       # sign-consistent S_max/tau
            good = sum(
                1 for _ in range(args.pgood_samples)
                if score_text(apply_key(ct, random_key(alphabet)), lm) >= threshold
            )
            p = max(good / args.pgood_samples, 1e-7)
            r_star, P_r = boyer_r_star(p)
            oracle_calls = 2 * r_star + 1
            classical = 1.0 / p
            speedup = classical / oracle_calls

            print(f"    tau={tau:.3f}  thr={threshold:.2f}  p_good={p:.5f}  "
                  f"r*={r_star}  N_oracle={oracle_calls}  speedup={speedup:.1f}x")

            rows.append({
                "alphabet_size":           len(alphabet),
                "cipher_length":           L,
                "tau":                     tau,
                "tau_threshold":           round(threshold, 3),
                "S_max_est":               round(S_max, 3),
                "p_good":                  round(p, 6),
                "grover_iters_opt":        r_star,
                "grover_oracle_calls":     oracle_calls,
                "grover_success_prob":     round(P_r, 4),
                "classical_expected_trials": round(classical, 2),
                "speedup_vs_random":       round(speedup, 3),
            })

        # Plot: score distribution with thresholds
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(ref_scores, bins=60, density=True, color="#aec6e8",
                edgecolor="white", linewidth=0.3)
        for tau, col in zip([0.90, 0.95, 0.98],
                            ["#2ca02c", "#ff7f0e", "#d62728"]):
            if tau in tau_list:
                ax.axvline(S_max / tau, color=col, lw=1.5, ls="--",
                           label=f"tau={tau}  thr={S_max/tau:.1f}")
        ax.axvline(S_max, color="black", lw=2, label=f"S_max={S_max:.1f}")
        ax.set_xlabel("Score S(pi)"); ax.set_ylabel("Density")
        ax.set_title(f"Score distribution with S_max/tau thresholds  L={L}")
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(out_dir / "plots" / f"score_hist_L{L}.png", dpi=150)
        plt.close(fig)

        # Plot: p_good vs tau
        xs = [r["tau"] for r in rows if r["cipher_length"] == L]
        ys = [r["p_good"] for r in rows if r["cipher_length"] == L]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy(xs, ys, marker="o", lw=1.5)
        ax.set_xlabel("tau"); ax.set_ylabel("p_good")
        ax.set_title(f"p_good vs tau (S_max/tau oracle, L={L})")
        ax.grid(ls="--", alpha=0.4); fig.tight_layout()
        fig.savefig(out_dir / "plots" / f"pgood_vs_tau_L{L}.png", dpi=150)
        plt.close(fig)

    # Write CSV
    if rows:
        with open(out_dir / "grover_sweep_results.csv", "w",
                  newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    # Summary plot: p_good vs tau for all L
    fig, ax = plt.subplots(figsize=(7, 4))
    mk = ["o", "s", "^", "D", "v"]
    c5 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, L in enumerate(cipher_lengths):
        xs = [r["tau"] for r in rows if r["cipher_length"] == L]
        ys = [r["p_good"] for r in rows if r["cipher_length"] == L and r["p_good"] > 1e-6]
        xs = xs[:len(ys)]
        if xs:
            ax.semilogy(xs, ys, marker=mk[i % 5], color=c5[i % 5],
                        lw=1.5, ms=5, label=f"L={L}")
    ax.set_xlabel("tau"); ax.set_ylabel("p_good (MC estimate)")
    ax.set_title(f"p_good vs tau  —  S_max/tau threshold, k={len(alphabet)}")
    ax.legend(fontsize=9); ax.grid(ls="--", alpha=0.4); fig.tight_layout()
    fig.savefig(out_dir / "plots" / f"fig_pgood_vs_tau_k{len(alphabet)}.png", dpi=150)
    plt.close(fig)
    print(f"\nResults -> {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: qubo   [FIX #1 applied: frequency-ranked alphabet]
# ─────────────────────────────────────────────────────────────────────────────

def cmd_qubo(args):
    """
    Permutation annealing under the QUBO objective E(pi) = -S(pi) + lambda*C(pi).
    C(pi) penalises deviations from a valid permutation matrix.

    FIX #1: the k-letter alphabet is now the k MOST FREQUENT letters
    (ranked by unigram probability), not the alphabetically-sorted prefix
    of lm["alphabet"]. This matches the paper's Methods description and
    ensures the k=7/k=8 QUBO experiments operate on a representative,
    not arbitrary, subset of the corpus's character distribution.
    """
    lm = json.loads(Path(args.models).read_text(encoding="utf-8"))
    ranked_alphabet = frequency_ranked_alphabet(lm)
    alphabet = ranked_alphabet[:args.alphabet_size]
    random.seed(args.seed); np.random.seed(args.seed)

    pt_full = Path(args.plaintext).read_text(encoding="utf-8", errors="ignore")
    pt = "".join(c for c in pt_full if c in alphabet)[:args.cipher_length]

    true_key = random_key(alphabet)
    ct = apply_key(pt, invert_key(true_key))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Permutation index annealing (no explicit binary matrix needed)
    perm = list(range(len(alphabet)))
    random.shuffle(perm)

    def perm_to_key(p):
        return dict(zip(alphabet, [alphabet[i] for i in p]))

    def energy(p):
        return -score_text(apply_key(ct, perm_to_key(p)), lm)

    steps = 25000; T0 = 3.0; Tend = 0.05
    E = energy(perm)
    best_perm, best_E = list(perm), E
    energies = []

    for t in range(steps):
        T = T0 * (Tend / T0) ** (t / (steps - 1))
        i, j = random.sample(range(len(alphabet)), 2)
        perm[i], perm[j] = perm[j], perm[i]
        nE = energy(perm)
        if nE < E or random.random() < math.exp((E - nE) / T):
            E = nE
        else:
            perm[i], perm[j] = perm[j], perm[i]
        if E < best_E:
            best_E, best_perm = E, list(perm)
        energies.append(E)

    best_key = perm_to_key(best_perm)
    decryption = apply_key(ct, best_key)
    acc = key_accuracy(true_key, best_key)

    (out_dir / "best_mapping.json").write_text(
        json.dumps(best_key, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "best_decryption.txt").write_text(decryption, encoding="utf-8")

    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["alphabet_size", "cipher_length",
                                          "final_energy", "key_accuracy",
                                          "alphabet_letters"])
        w.writeheader()
        w.writerow({"alphabet_size": len(alphabet), "cipher_length": args.cipher_length,
                    "final_energy": round(best_E, 4),
                    "key_accuracy": round(acc, 4),
                    "alphabet_letters": alphabet})

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(energies, lw=0.8, color="#2166ac")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Energy (lower is better)")
    ax.set_title(f"QUBO permutation annealing  k={len(alphabet)}, L={args.cipher_length}")
    fig.tight_layout()
    fig.savefig(out_dir / "energy_curve.png", dpi=150)
    plt.close(fig)
    print(f"QUBO done: accuracy={acc:.3f}  energy={best_E:.3f}  "
          f"alphabet={alphabet}  -> {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: quantum-grover   [FIX #1 applied: frequency-ranked alphabet]
# ─────────────────────────────────────────────────────────────────────────────

def cmd_quantum_grover(args):
    """
    Exact Grover amplitude-amplification simulation over the full k! state space.

    For each (cipher_length L, tau):
      - All k! permutations are enumerated and scored exactly.
      - The oracle threshold is S_max / tau.
      - A NumPy statevector simulation runs for --max_iters iterations,
        recording the success probability P_t at each step.
      - t* = argmax P_t  is cross-validated against the analytical formula
        P_t^th = sin^2((2t+1)*theta);  the two must agree.
      - r* from the Boyer formula is reported as the operationally relevant
        stopping point.

    FIX #1: the k-letter alphabet is now the k MOST FREQUENT letters
    (ranked by unigram probability), not an alphabetically-sorted prefix.

    Recommended: alphabet_size <= 8  (8! = 40320 states fits in RAM easily).
    """
    lm  = json.loads(Path(args.models).read_text(encoding="utf-8"))
    ranked_alphabet = frequency_ranked_alphabet(lm)
    alphabet  = ranked_alphabet[:args.alphabet_size]
    N_states  = math.factorial(len(alphabet))
    random.seed(args.seed); np.random.seed(args.seed)

    pt_full = Path(args.plaintext).read_text(encoding="utf-8", errors="ignore")
    pt_full = "".join(c for c in pt_full if c in alphabet)

    cipher_lengths = [int(x) for x in args.cipher_lengths.split(",")]
    tau_list       = [float(x) for x in args.tau_list.split(",")]
    max_iters      = args.max_iters

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    print(f"Exact quantum simulation  k={len(alphabet)}  N={N_states:,}  "
          f"alphabet (freq-ranked) = {alphabet}")
    all_perms = [dict(zip(alphabet, p))
                 for p in itertools.permutations(alphabet)]

    summary_rows   = []

    for L in cipher_lengths:
        if len(pt_full) < L:
            print(f"  L={L}: corpus too short, skipping"); continue
        pt      = pt_full[:L]
        true_key = random_key(alphabet)
        ct       = apply_key(pt, invert_key(true_key))

        print(f"\n  [L={L}] Scoring all {N_states:,} permutations …",
              end="", flush=True)
        all_scores = np.array(
            [score_text(apply_key(ct, perm), lm) for perm in all_perms],
            dtype=np.float64)
        S_max  = float(all_scores.max())
        S_true = score_text(pt, lm)
        print(f" done.  S_max={S_max:.2f}  S_true={S_true:.2f}  "
              f"gap={S_true-S_max:.2f}")

        for tau in tau_list:
            threshold = S_max / tau
            marked    = all_scores >= threshold
            M         = int(marked.sum())
            p_exact   = M / N_states

            if M == 0:
                print(f"    tau={tau}: M=0 (threshold too strict), skipping")
                continue
            if M == N_states:
                print(f"    tau={tau}: all states marked (threshold too loose), skipping")
                continue

            # Boyer analytical prediction
            r_star, P_r = boyer_r_star(p_exact)
            theta = math.asin(math.sqrt(p_exact))
            analytic = [math.sin((2 * t + 1) * theta) ** 2
                        for t in range(1, max_iters + 1)]

            # Statevector simulation
            N = N_states
            psi      = np.ones(N, dtype=np.float64) / math.sqrt(N)
            uniform  = psi.copy()
            o_sign   = np.where(marked, -1.0, 1.0)
            best_t, best_P = 0, float(np.sum(psi[marked] ** 2))
            sim_probs = []

            for it in range(1, max_iters + 1):
                psi    = o_sign * psi
                psi    = 2.0 * float(np.dot(uniform, psi)) * uniform - psi
                norm   = float(np.linalg.norm(psi))
                if norm > 1e-12:
                    psi /= norm
                p_it   = float(np.sum(psi[marked] ** 2))
                sim_probs.append(p_it)
                if p_it > best_P:
                    best_P, best_t = p_it, it

            t_analytic = int(np.argmax(analytic)) + 1
            sim_ok     = (best_t == t_analytic)
            spd_boyer  = (1.0 / p_exact) / (2 * r_star + 1)
            spd_tstar  = (1.0 / p_exact) / (2 * best_t + 1)

            print(f"    tau={tau:.3f}  M={M:5d}  p={p_exact:.5f}  "
                  f"r*={r_star}  t_sim={best_t}  t_an={t_analytic}  "
                  f"{'OK' if sim_ok else 'MISMATCH'}  "
                  f"SpeedupBoyer={spd_boyer:.2f}x")

            summary_rows.append({
                "k":                 len(alphabet),
                "alphabet_letters":  alphabet,
                "L":                 L,
                "tau":               tau,
                "S_max":             round(S_max, 3),
                "S_true":            round(S_true, 3),
                "S_max_gap":         round(S_true - S_max, 3),
                "threshold":         round(threshold, 3),
                "M_marked":          M,
                "N_states":          N_states,
                "p_good":            round(p_exact, 6),
                "r_star":            r_star,
                "oracle_calls_boyer":2 * r_star + 1,
                "P_r_star":          round(P_r, 5),
                "t_star_sim":        best_t,
                "t_star_analytic":   t_analytic,
                "sim_matches_analytic": sim_ok,
                "oracle_calls_sim":  2 * best_t + 1,
                "best_prob_sim":     round(best_P, 5),
                "classical_trials":  round(1.0 / p_exact, 2),
                "speedup_boyer":     round(spd_boyer, 3),
                "speedup_tstar":     round(spd_tstar, 3),
            })

            # Probability envelope plot for selected (L, tau)
            if L in [200, 1000] and tau in [0.92, 0.95, 0.98]:
                iters = list(range(1, max_iters + 1))
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(iters, sim_probs, "b-o", ms=3, lw=1.5,
                        label="Statevector sim")
                ax.plot(iters, analytic, "r--", lw=1.5,
                        label="Analytical formula")
                ax.axvline(best_t, color="blue", ls=":", lw=1.2,
                           label=f"t*_sim={best_t}  P={best_P:.3f}")
                ax.axvline(r_star, color="red", ls=":", lw=1.2,
                           label=f"r*_Boyer={r_star}  P={P_r:.3f}")
                ax.axhline(p_exact, color="gray", ls="--", lw=0.8,
                           label=f"p_good={p_exact:.4f}")
                ax.set_xlabel("Grover iterations")
                ax.set_ylabel("Success probability")
                ax.set_title(
                    f"k={len(alphabet)}, L={L}, tau={tau}  "
                    f"(S_max/tau,  p_good={p_exact:.4f})")
                ax.legend(fontsize=8); ax.set_ylim(0, 1.05); fig.tight_layout()
                fname = (f"envelope_k{len(alphabet)}_L{L}"
                         f"_tau{str(tau).replace('.','')}.png")
                fig.savefig(out_dir / "plots" / fname, dpi=150)
                plt.close(fig)

    # Write summary CSV
    if summary_rows:
        csv_path = out_dir / f"exact_k{len(alphabet)}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader(); w.writerows(summary_rows)
        print(f"\n  Summary CSV -> {csv_path}")

    # Combined summary figure
    if summary_rows:
        mk = ["o", "s", "^", "D", "v"]
        c5 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        # p_good vs L for all tau
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, tau in enumerate(tau_list):
            xs = [r["L"]     for r in summary_rows if r["tau"] == tau]
            ys = [r["p_good"] for r in summary_rows if r["tau"] == tau]
            if xs: ax.semilogy(xs, ys, marker=mk[i % 5], color=c5[i % 5],
                               lw=1.5, ms=5, label=f"tau={tau}")
        ax.set_xlabel("Cipher length L"); ax.set_ylabel("p_good (exact)")
        ax.set_title(f"Length-dependence of p_good  —  S_max/tau, k={len(alphabet)}")
        ax.legend(fontsize=8); ax.grid(ls="--", alpha=0.4); fig.tight_layout()
        fig.savefig(out_dir / "plots" / f"pgood_vs_L_k{len(alphabet)}.png",
                    dpi=150); plt.close(fig)

        # Boyer speedup vs tau per L
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, L in enumerate(cipher_lengths):
            xs = [r["tau"]          for r in summary_rows if r["L"] == L]
            ys = [r["speedup_boyer"] for r in summary_rows if r["L"] == L]
            if xs: ax.plot(xs, ys, marker=mk[i % 5], color=c5[i % 5],
                           lw=1.5, ms=5, label=f"L={L}")
        ax.axhline(1.0, color="black", ls="--", lw=1, label="Break-even")
        ax.set_xlabel("tau"); ax.set_ylabel("Quantum speedup (Boyer)")
        ax.set_title(f"Boyer-formula quantum speedup  —  k={len(alphabet)}")
        ax.legend(fontsize=8); ax.grid(ls="--", alpha=0.4); fig.tight_layout()
        fig.savefig(out_dir / "plots" / f"speedup_k{len(alphabet)}.png",
                    dpi=150); plt.close(fig)

        # Simulation integrity check summary
        n_ok = sum(1 for r in summary_rows if r["sim_matches_analytic"])
        print(f"\n  Boyer formula accuracy: {n_ok}/{len(summary_rows)} cases "
              f"sim == analytic  "
              f"({'ALL PASS' if n_ok == len(summary_rows) else 'CHECK MISMATCHES'})")

    print(f"\nQuantum simulation done -> {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: grover-project  (analytical scalability, S_max/tau threshold)
#   [FIX #1 applied: frequency-ranked alphabet]
#   [FIX #2 applied: SA augmentation added to S_max estimate]
#   [FIX #3 applied: k_max validated against true alphabet length]
# ─────────────────────────────────────────────────────────────────────────────

def cmd_grover_project(args):
    """
    Analytical scalability projection across alphabet sizes k_min..k_max.

    For each k:
      1. Restrict corpus to the k MOST FREQUENT letters (FIX #1: ranked by
         unigram probability, not an alphabetically-sorted prefix).
      2. Sample --pgood_samples permutations to estimate S_max and p_good
         under the S_max/tau threshold. S_max is estimated from an
         independent reference sample of up to 5,000 permutations,
         AUGMENTED by the simulated-annealing best score (FIX #2) — the
         same procedure used in cmd_grover_sweep and described in the
         paper's Methods Sec. 3.4.
      3. Apply the Boyer formula to obtain oracle call counts.

    FIX #3: k_max cannot exceed the number of distinct letters actually
    present in the corpus alphabet (21, after normalize_renaissance()).
    Python string slicing does not raise an error past the end of a
    string, so full_alph[:23] on a 21-letter alphabet would previously
    have silently returned the same 21-letter set again under a k=22 or
    k=23 label — duplicating the k=21 row rather than producing a true
    22- or 23-letter sweep. This function now raises a clear error instead.

    Results are indicative for k > 8 because S_max is estimated from a
    finite sample.  Exact-simulation results at k=7 and k=8 anchor the curve.
    """
    lm = json.loads(Path(args.models).read_text(encoding="utf-8"))
    full_alph = lm["alphabet"]
    ranked_alph = frequency_ranked_alphabet(lm)

    # FIX #3: validate k_max against the true alphabet size.
    if args.k_max > len(full_alph):
        raise ValueError(
            f"--k_max={args.k_max} exceeds the corpus alphabet's true size "
            f"({len(full_alph)} distinct letters: '{full_alph}'). "
            f"String slicing past the end of the alphabet would silently "
            f"duplicate the k={len(full_alph)} row under higher-k labels "
            f"instead of raising an error. Set --k_max <= {len(full_alph)}."
        )

    random.seed(args.seed); np.random.seed(args.seed)

    pt_full = Path(args.plaintext).read_text(encoding="utf-8", errors="ignore")
    tau_list = [float(x) for x in args.tau_list.split(",")]
    n_samp   = args.pgood_samples

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    rows = []
    k_values = list(range(args.k_min, args.k_max + 1))

    for k in k_values:
        alphabet = ranked_alph[:k]
        pt = "".join(c for c in pt_full if c in alphabet)[:args.cipher_length]
        if len(pt) < args.cipher_length:
            print(f"  k={k}: corpus too short for L={args.cipher_length}, skipping")
            continue
        true_key = random_key(alphabet)
        ct = apply_key(pt, invert_key(true_key))

        # Estimate S_max from reference sample, AUGMENTED by SA best (FIX #2)
        ref = np.array([score_text(apply_key(ct, random_key(alphabet)), lm)
                        for _ in range(min(n_samp, 5000))], dtype=np.float64)
        sa_key, sa_score = simulated_annealing(ct, lm, steps=5000, seed=args.seed)
        S_max = max(float(ref.max()), sa_score)

        print(f"  k={k}  alphabet={alphabet}  S_max_est={S_max:.2f}", end="")
        for tau in tau_list:
            threshold = S_max / tau
            good = sum(1 for _ in range(n_samp)
                       if score_text(apply_key(ct, random_key(alphabet)), lm)
                          >= threshold)
            p = max(good / n_samp, 1e-7)
            r_star, _ = boyer_r_star(p)
            oracle_calls   = 2 * r_star + 1
            classical      = 1.0 / p
            speedup        = classical / oracle_calls

            print(f"  [tau={tau}: p={p:.4f} N_or={oracle_calls}]", end="")
            rows.append({
                "k":             k,
                "alphabet_letters": alphabet,
                "tau":           tau,
                "S_max_est":     round(S_max, 3),
                "tau_threshold": round(threshold, 3),
                "p_good":        round(p, 6),
                "r_star":        r_star,
                "oracle_calls":  oracle_calls,
                "classical_trials": round(classical, 2),
                "speedup":       round(speedup, 3),
            })
        print()

    # Write CSV
    csv_path = out_dir / "grover_project_results.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    # Combined 3-panel plot
    marker_list = ["o", "s", "^", "D", "v", "P", "X", "*"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
              "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (metric, ylabel, title) in zip(axes, [
        ("oracle_calls",     "Oracle calls",         "Grover oracle calls"),
        ("classical_trials", "Classical trials",     "Classical expected trials"),
        ("speedup",          "Quantum speedup",      "Boyer speedup"),
    ]):
        for i, tau in enumerate(tau_list):
            xs = [r["k"]      for r in rows if r["tau"] == tau]
            ys = [r[metric]   for r in rows if r["tau"] == tau]
            ax.plot(xs, ys, marker=marker_list[i % len(marker_list)],
                    color=colors[i % len(colors)],
                    label=f"tau={tau}", lw=1.5, ms=5)
        if metric == "speedup":
            ax.axhline(1.0, color="black", ls="--", lw=1)
        ax.set_xlabel("Alphabet size k", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
        ax.set_yscale("log")
        ax.grid(ls="--", alpha=0.4)

    fig.suptitle(
        "Analytical scalability projection: Grover complexity vs alphabet size k\n"
        "(S_max/tau threshold, k most frequent letters; k>8 estimates are indicative)",
        fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "combined_scaling_k7_k21.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Projection done -> {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: figures
# ─────────────────────────────────────────────────────────────────────────────

def cmd_figures(args):
    """Generate publication summary figures from previously saved CSVs."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Classical baselines bar chart
    if args.baselines_dir:
        bpath = Path(args.baselines_dir) / "results.csv"
        if bpath.exists():
            import csv as _csv
            with open(bpath, newline="", encoding="utf-8") as f:
                brows = list(_csv.DictReader(f))
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar([r["method"] for r in brows],
                   [float(r["runtime_sec"]) for r in brows],
                   color=["#2166ac", "#d6604d"])
            ax.set_xlabel("Method"); ax.set_ylabel("Runtime (s)")
            ax.set_title("Classical baselines: runtime")
            fig.tight_layout()
            fig.savefig(out_dir / "baselines_runtime_bar.png", dpi=150)
            plt.close(fig)

    # Grover sweep summary
    if args.sweep_dir:
        spath = Path(args.sweep_dir) / "grover_sweep_results.csv"
        if spath.exists():
            import csv as _csv
            with open(spath, newline="", encoding="utf-8") as f:
                srows = list(_csv.DictReader(f))
            lengths = sorted(set(int(r["cipher_length"]) for r in srows))
            mk = ["o", "s", "^", "D", "v"]
            c5 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

            # p_good vs tau for all L
            fig, ax = plt.subplots(figsize=(7, 4))
            for i, L in enumerate(lengths):
                xs = [float(r["tau"])   for r in srows if int(r["cipher_length"]) == L]
                ys = [float(r["p_good"]) for r in srows if int(r["cipher_length"]) == L
                      and float(r["p_good"]) > 1e-6]
                xs = xs[:len(ys)]
                if xs:
                    ax.semilogy(xs, ys, marker=mk[i % 5], color=c5[i % 5],
                                lw=1.5, ms=5, label=f"L={L}")
            ax.set_xlabel("tau"); ax.set_ylabel("p_good")
            ax.set_title("p_good vs tau  (S_max/tau oracle)")
            ax.legend(fontsize=9); ax.grid(ls="--", alpha=0.4)
            fig.tight_layout()
            fig.savefig(out_dir / "fig_pgood_vs_tau_summary.png", dpi=150)
            plt.close(fig)

            # Oracle calls and classical trials at tau~0.98
            tau_star = 0.98
            bars = []
            for L in lengths:
                sub = [r for r in srows
                       if int(r["cipher_length"]) == L]
                if not sub: continue
                closest = min(sub, key=lambda r: abs(float(r["tau"]) - tau_star))
                bars.append((str(L),
                             float(closest["grover_oracle_calls"]),
                             float(closest["classical_expected_trials"])))

            if bars:
                xs, gor, cls = zip(*bars)
                for vals, title, fname in [
                    (gor, "Grover oracle calls vs length (tau≈0.98)",
                     "grover_calls_vs_length_tau098.png"),
                    (cls, "Classical expected trials vs length (tau≈0.98)",
                     "classical_trials_vs_length_tau098.png"),
                ]:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.bar(xs, vals, color="#2166ac")
                    ax.set_xlabel("Cipher length"); ax.set_title(title)
                    fig.tight_layout()
                    fig.savefig(out_dir / fname, dpi=150)
                    plt.close(fig)

    # Quantum simulation summary figures
    if args.quantum_dir:
        import csv as _csv
        qdir = Path(args.quantum_dir)
        for csv_file in qdir.glob("exact_k*.csv"):
            with open(csv_file, newline="", encoding="utf-8") as f:
                qrows = list(_csv.DictReader(f))
            k = int(qrows[0]["k"])
            taus    = sorted(set(float(r["tau"]) for r in qrows))
            lengths = sorted(set(int(r["L"])    for r in qrows))
            mk = ["o", "s", "^", "D", "v"]
            c5 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

            fig, ax = plt.subplots(figsize=(7, 4))
            for i, tau in enumerate(taus):
                xs = [int(r["L"])          for r in qrows if float(r["tau"]) == tau]
                ys = [float(r["p_good"])   for r in qrows if float(r["tau"]) == tau]
                if xs:
                    ax.semilogy(xs, ys, marker=mk[i % 5], color=c5[i % 5],
                                lw=1.5, ms=5, label=f"tau={tau}")
            ax.set_xlabel("Cipher length L"); ax.set_ylabel("p_good (exact)")
            ax.set_title(f"Length-dependence of p_good  —  S_max/tau, k={k}")
            ax.legend(fontsize=8); ax.grid(ls="--", alpha=0.4)
            fig.tight_layout()
            fig.savefig(out_dir / f"fig_pgood_vs_L_k{k}.png", dpi=150)
            plt.close(fig)

            # Boyer speedup summary
            fig, ax = plt.subplots(figsize=(7, 4))
            for i, L in enumerate(lengths):
                xs = [float(r["tau"])           for r in qrows if int(r["L"]) == L]
                ys = [float(r["speedup_boyer"])  for r in qrows if int(r["L"]) == L]
                if xs:
                    ax.plot(xs, ys, marker=mk[i % 5], color=c5[i % 5],
                            lw=1.5, ms=5, label=f"L={L}")
            ax.axhline(1.0, color="black", ls="--", lw=1, label="Break-even")
            ax.set_xlabel("tau"); ax.set_ylabel("Quantum speedup (Boyer)")
            ax.set_title(f"Boyer-formula speedup  —  k={k}")
            ax.legend(fontsize=8); ax.grid(ls="--", alpha=0.4)
            fig.tight_layout()
            fig.savefig(out_dir / f"fig_speedup_k{k}.png", dpi=150)
            plt.close(fig)

    print(f"Figures saved -> {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        prog="renaissance_cipher_suite",
        description="Corpus-driven oracle construction for quantum key-recovery analysis.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # normalize
    sp = sub.add_parser("normalize",
        help="Normalise raw texts to Renaissance 21-letter alphabet")
    sp.add_argument("--in_files", nargs="+", required=True)
    sp.add_argument("--out_dir",  required=True)
    sp.add_argument("--merge",    default=None,
                    help="Merged output filename (written inside out_dir)")
    sp.set_defaults(func=cmd_normalize)

    # build-model
    sp = sub.add_parser("build-model",
        help="Build n-gram language model from normalised corpus")
    sp.add_argument("--corpus",     nargs="+", required=True)
    sp.add_argument("--out",        required=True)
    sp.add_argument("--max_order",  type=int, default=3)
    sp.add_argument("--min_count",  type=int, default=1)
    sp.add_argument("--smoothing",  type=float, default=0.001)
    sp.set_defaults(func=cmd_build_model)

    # classical
    sp = sub.add_parser("classical",
        help="Hill climbing and simulated annealing baselines")
    sp.add_argument("--models",    required=True)
    sp.add_argument("--plaintext", required=True)
    sp.add_argument("--ciphertext", default=None)
    sp.add_argument("--synthesize", action="store_true")
    sp.add_argument("--out_dir",   required=True)
    sp.add_argument("--method",    choices=["hill", "anneal", "both"],
                    default="both")
    sp.add_argument("--seed",      type=int, default=1337)
    sp.set_defaults(func=cmd_classical)

    # grover-sweep
    sp = sub.add_parser("grover-sweep",
        help="Monte Carlo Grover sweep (S_max/tau threshold, large alphabet)")
    sp.add_argument("--models",        required=True)
    sp.add_argument("--plaintext",     required=True)
    sp.add_argument("--alphabet_size", type=int, required=True)
    sp.add_argument("--cipher_lengths",required=True,
                    help="Comma-separated, e.g. 200,400,600,800,1000")
    sp.add_argument("--tau_list",      required=True,
                    help="Comma-separated tau values, e.g. 0.90,0.95,0.98,0.995")
    sp.add_argument("--out_dir",       required=True)
    sp.add_argument("--seed",          type=int, default=42)
    sp.add_argument("--score_samples", type=int, default=5000,
                    help="Permutations for S_max estimation (default: 5000)")
    sp.add_argument("--pgood_samples", type=int, default=20000,
                    help="Permutations for p_good estimation (default: 20000)")
    sp.set_defaults(func=cmd_grover_sweep)

    # qubo
    sp = sub.add_parser("qubo",
        help="QUBO permutation annealing on reduced alphabet "
             "(k most frequent letters)")
    sp.add_argument("--models",        required=True)
    sp.add_argument("--plaintext",     required=True)
    sp.add_argument("--alphabet_size", type=int, required=True)
    sp.add_argument("--cipher_length", type=int, required=True)
    sp.add_argument("--out_dir",       required=True)
    sp.add_argument("--seed",          type=int, default=7)
    sp.set_defaults(func=cmd_qubo)

    # quantum-grover
    sp = sub.add_parser("quantum-grover",
        help="Exact statevector Grover simulation over all k! permutations "
             "(k<=8 recommended; k most frequent letters)")
    sp.add_argument("--models",        required=True)
    sp.add_argument("--plaintext",     required=True)
    sp.add_argument("--alphabet_size", type=int, required=True)
    sp.add_argument("--cipher_lengths",required=True,
                    help="Comma-separated, e.g. 200,400,600,800,1000")
    sp.add_argument("--tau_list",      required=True,
                    help="Comma-separated tau values, e.g. 0.90,0.92,0.95,0.98")
    sp.add_argument("--out_dir",       required=True)
    sp.add_argument("--max_iters",     type=int, default=50,
                    help="Maximum Grover iterations to simulate (default: 50)")
    sp.add_argument("--pgood_samples", type=int, default=5000,
                    help="Unused in exact mode; kept for CLI compatibility")
    sp.add_argument("--seed",          type=int, default=42)
    sp.set_defaults(func=cmd_quantum_grover)

    # grover-project
    sp = sub.add_parser("grover-project",
        help="Analytical scalability projection across alphabet sizes "
             "(k most frequent letters; k_max capped at the true corpus "
             "alphabet size)")
    sp.add_argument("--models",        required=True)
    sp.add_argument("--plaintext",     required=True)
    sp.add_argument("--k_min",         type=int, default=7)
    sp.add_argument("--k_max",         type=int, default=21,
                    help="Default 21: the true size of the normalised "
                         "Renaissance Italian corpus alphabet. Values "
                         "above the corpus's actual alphabet length will "
                         "raise an error (FIX #3) rather than silently "
                         "duplicating data.")
    sp.add_argument("--cipher_length", type=int, default=400)
    sp.add_argument("--tau_list",      required=True,
                    help="Comma-separated tau values")
    sp.add_argument("--pgood_samples", type=int, default=20000)
    sp.add_argument("--out_dir",       required=True)
    sp.add_argument("--seed",          type=int, default=42)
    sp.set_defaults(func=cmd_grover_project)

    # figures
    sp = sub.add_parser("figures",
        help="Generate publication figures from existing result CSVs")
    sp.add_argument("--sweep_dir",    default=None)
    sp.add_argument("--quantum_dir",  default=None)
    sp.add_argument("--baselines_dir",default=None)
    sp.add_argument("--out_dir",      required=True)
    sp.set_defaults(func=cmd_figures)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
