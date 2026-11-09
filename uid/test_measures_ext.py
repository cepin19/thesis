#!/usr/bin/env python3
"""Checks of measures_ext.py on inputs with known answers. Run: python test_measures_ext.py"""
import numpy as np

import measures as M
import measures_ext as E

rng = np.random.default_rng(0)


def w_from(s, ent=None, null=None, words=None, line=None, sent=None):
    n = len(s)
    return {"s": np.asarray(s, float), "ent": ent, "null": null, "words": words or ["slovo"] * n,
            "line": np.zeros(n, int) if line is None else line, "sent": np.zeros(n, int) if sent is None else sent}


# i.i.d. noise: von Neumann ratio 2, ACF 0, skew 0, variance of window sums linear in the window (slope 1)
iid = rng.normal(50, 5, 200000)
r = E.extended(w_from(iid))
assert abs(r["vn_ratio"] - 2) < 0.02 and abs(r["ac1"]) < 0.01 and abs(r["skew"]) < 0.03 and abs(r["kurt"]) < 0.05
assert abs(r["vt_slope"] - 1) < 0.03 and abs(r["fano_w4"] - 1) < 0.05
assert abs(r["spec_high"] - 0.5) < 0.01 and abs(r["spec_entropy"] - 0.93) < 0.1

# shuffling leaves the distribution unchanged but changes the order measures
ar = np.zeros(100000)
for i in range(1, len(ar)):
    ar[i] = 0.8 * ar[i - 1] + rng.normal()
ar += 50
a, b = E.extended(w_from(ar)), E.extended(w_from(rng.permutation(ar)))
assert abs(a["cv"] - b["cv"]) < 1e-3 and abs(a["ac1"] - 0.8) < 0.02 and abs(b["ac1"]) < 0.02
assert a["vn_ratio"] < 0.5 and abs(b["vn_ratio"] - 2) < 0.03 and a["vt_slope"] > b["vt_slope"] + 0.3
assert abs(a["vn_ratio"] - 2 * (1 - a["ac1"])) < 0.01
assert a["spec_high"] < b["spec_high"]

# a pure sinusoid is one spectral peak at its frequency
t = np.arange(512)
sine = 20 + 5 * np.sin(2 * np.pi * t / 16)
r = E.extended(w_from(sine))
assert r["fisher_g"] > 0.95 and abs(r["spec_centroid"] - 1 / 16) < 0.01

# time scaling finds a sinusoid in unit position, also when units differ in length
lens = rng.integers(5, 15, 60)
pos = np.concatenate([np.arange(k) / k for k in lens])
unit = np.repeat(np.arange(len(lens)), lens)
s = 20 + 5 * np.sin(2 * np.pi * pos) + rng.normal(0, 0.5, len(pos))
assert E.extended(w_from(s, sent=unit))["harm_gain"] > 0.9
assert E.extended(w_from(rng.permutation(s), sent=unit))["harm_gain"] < 0.05

# typicality: zero when surprisal equals entropy
s = rng.uniform(1, 20, 100)
r = E.extended(w_from(s, ent=s.copy()))
assert r["typ"] == 0 and r["excess"] == 0
r = E.extended(w_from(s, ent=s - 2))
assert abs(r["typ"] - 2) < 1e-9 and abs(r["excess"] - 2) < 1e-9

# calibrated CV: about 0 for a text drawn from the null, strongly negative for a flat one
null = rng.exponential(5, (300, 200)) + 0.1
r = E.extended(w_from(rng.exponential(5, 300) + 0.1, null=null))
assert abs(r["z_cv"]) < 3
assert E.extended(w_from(np.full(300, 5.0) + rng.normal(0, 0.01, 300), null=null))["z_cv"] < -5

# per-unit cost: surprisal proportional to characters has CV 0 per character
words = ["a" * k for k in rng.integers(1, 12, 200)]
s = np.array([3.0 * len(x) for x in words])
r = E.extended(w_from(s, words=words))
assert r["bpc_cv"] < 1e-9 and r["cv"] > 0.3 and abs(r["bpc_mean"] - 3) < 1e-9
assert list(E.n_syll(["strom", "kočka", "auto", "Praha."])) == [1, 2, 2, 2]

# between-unit spread: equal unit means give 0
s = np.tile([1.0, 3.0], 60)
r = E.extended(w_from(s, sent=np.repeat(np.arange(12), 10)))
assert r["sent_var"] < 0.2

# states: known chain
c = E.fit_states(np.r_[rng.normal(1, 0.1, 500), rng.normal(5, 0.1, 500), rng.normal(10, 0.1, 500), rng.normal(20, 0.1, 500)])
assert np.allclose(c, [1, 5, 10, 20], atol=0.1)
m = E.state_measures(np.array([0, 1, 20, 1, 20, 1, 20, 1.0]), c)
assert m["mk_spike"] == 1 and m["mk_recovery"] == 1 and m["mk_stay_top"] == 0 and m["mk_entropy_rate"] == 0

# to_words agrees with measures.units, excludes newlines, sums entropy and null, and stops at a truncation
text = "ab cd,\nef gh"
rec = {"offsets": [[0, 2], [2, 5], [5, 6], [6, 7], [7, 9], [9, 12]], "surp": [1.0, 2.0, 3.0, 99.0, 4.0, 5.0],
       "ent": [1.0, 1.0, 1.0, 99.0, 1.0, 1.0], "null": [[1.0, 2.0]] * 3 + [[99.0, 99.0]] + [[1.0, 2.0]] * 2}
w = E.to_words(text, rec)
assert list(w["s"]) == [1, 5, 4, 5] and list(w["ent"]) == [1, 2, 1, 1] and list(w["line"]) == [0, 0, 1, 1]
assert w["null"].shape == (4, 2) and list(w["null"][1]) == [2, 4]
assert list(E.to_words(text, rec, "nopunct")["s"]) == [1, 2, 4, 5]
keep = {"offsets": rec["offsets"][:3] + rec["offsets"][4:], "surp": rec["surp"][:3] + rec["surp"][4:]}
assert list(M.units(text, keep["offsets"], keep["surp"], "untok")) == list(w["s"])
assert len(E.to_words(text, {**rec, "offsets": rec["offsets"][:2], "surp": rec["surp"][:2], "ent": rec["ent"][:2], "null": rec["null"][:2]})["s"]) == 2
# ranks: the first unit's tokens are dropped, punctuation is dropped under nopunct, and whitespace tokens never count
rr = {**rec, "rank": [1, 1, 3, 99, 1, 7]}
w = E.to_words(text, rr)
assert list(w["rank"]) == [1, 3, 1, 7]
assert list(E.to_words(text, rr, "nopunct")["rank"]) == [1, 1, 7]
z = E.extended({"s": np.arange(1.0, 9.0), "ent": None, "null": None, "words": ["ab"] * 8, "line": np.zeros(8, int), "sent": np.zeros(8, int), "rank": np.array([1, 1, 4, 8.0])})
assert z["top1"] == 0.5 and z["top5"] == 0.75 and abs(z["log2rank"] - 1.25) < 1e-9
# Pimentel & Meister correction: corrected word surprisals telescope to token sum - lb[after last] + lb[first]
E.PM = True
t2 = "ab cd ef"
r2 = {"offsets": [[0, 1], [1, 2], [2, 5], [5, 8]], "surp": [1.0, 2.0, 3.0, 4.0], "bow_ws": [-0.1, -2.0, -0.5, -0.3, -0.2], "bow_wp": [-0.1, -1.0, -0.4, -0.3, -0.2]}
w2 = E.to_words(t2, r2)
# "ab" starts the text: P_before = 1. " cd" and " ef" carry their own leading space: P_before = their word-start mass
assert np.allclose(w2["s"], [1 + 2 - (-0.5), 3 + (-0.5) - (-0.3), 4 + (-0.3) - (-0.2)])
assert np.isclose(w2["s"].sum(), 10 - (-0.2))
# newline before a word: the whitespace token is the word boundary
t3 = "ab\ncd"
r3 = {"offsets": [[0, 2], [2, 3], [3, 5]], "surp": [1.0, 5.0, 2.0], "bow_ws": [-0.1, -0.7, -3.0, -0.2], "bow_wp": [-0.1, -0.7, -3.0, -0.2]}
w3 = E.to_words(t3, r3)
assert np.allclose(w3["s"], [1 - (-0.7), 2 - (-0.2)]) and w3["null"] is None  # after a newline token, P_before = 1
# punctuation splits words under nopunct: the word after "(" gets P_before = 1 and the corrected value stays >= 0
t4 = "a (bc"
r4 = {"offsets": [[0, 1], [1, 3], [3, 5]], "surp": [1.0, 2.0, 0.1], "bow_ws": [-0.1, -0.2, -6.0, -0.1], "bow_wp": [-0.1, -0.2, -6.0, -0.1]}
w4 = E.to_words(t4, r4, "nopunct")
assert np.allclose(w4["s"], [1 - (-0.2), 0.1 - (-0.1)]) and (w4["s"] >= 0).all()
# BLOOM-style token ".\n" carries the boundary itself: P_after = 1 for the word it closes, P_before = 1 for the next word
t5 = "ab.\ncd"
r5 = {"offsets": [[0, 2], [2, 4], [4, 6]], "surp": [1.0, 2.0, 0.5], "bow_ws": [-0.1, -0.3, -5.0, -0.2], "bow_wp": [-0.1, -0.3, -5.0, -0.2]}
w5 = E.to_words(t5, r5)
assert np.allclose(w5["s"], [1 + 2, 0.5 - (-0.2)]), w5["s"]
E.PM = False
assert np.allclose(E.to_words(t2, r2)["s"], [3, 3, 4])
print("all checks passed")
