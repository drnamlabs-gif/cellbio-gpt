# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Bioelectric physics verifier for RLVR (stage 5b).

Ground truth comes from the Nernst equation — the same membrane physics BETSE
integrates: E_ion = (RT/zF) ln([out]/[in]), computed here at 37 C.
Three verifiable problem families:
  nernst   - compute E_ion in mV                       (numeric, graded reward)
  deltaE   - [ion]out changes a->b: E increases or decreases?   (categorical)
  polarity - membrane predominantly permeable to ion, [out] a->b:
             depolarize or hyperpolarize?                        (categorical)

reward(): parses a model answer and scores it against physics — the model
never sees the equation at reward time, only the scalar. Gold answers must
score exactly 1.0 (self-test in make_problems.py).
"""

import math
import re

RT_F = 26.73           # mV at 310.15 K
IONS = {               # z, physiological concentration menus (mM)
    "K+":  {"z": 1,  "ins": [100, 110, 120, 130, 140, 150, 155],
            "outs": [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40]},
    "Na+": {"z": 1,  "ins": [5, 8, 10, 12, 15, 18, 20, 25],
            "outs": [100, 110, 120, 130, 135, 140, 145, 150, 155, 160]},
    "Cl-": {"z": -1, "ins": [4, 5, 6, 8, 10, 15, 20, 30],
            "outs": [95, 100, 105, 110, 115, 120, 125, 130]},
}

PROMPTS_NERNST = [
    "At 37 C a cell has [{ion}]in = {cin} mM and [{ion}]out = {cout} mM. "
    "Compute the Nernst equilibrium potential of {ion} in mV.",
    "Intracellular {ion} is {cin} mM and extracellular {ion} is {cout} mM "
    "(37 C). What is E_{sym} in mV?",
]
PROMPTS_DELTAE = [
    "Extracellular {ion} changes from {a} mM to {b} mM ([{ion}]in fixed at "
    "{cin} mM). Does the equilibrium potential E_{sym} increase or decrease?",
    "If [{ion}]out changes from {a} to {b} mM at constant [{ion}]in = {cin} mM, "
    "does E_{sym} increase or decrease?",
]
PROMPTS_POLARITY = [
    "A membrane is predominantly permeable to {ion}. Extracellular {ion} "
    "changes from {a} mM to {b} mM. Does the cell depolarize or hyperpolarize?",
    "In a cell whose resting potential is set by {ion}, [{ion}]out shifts "
    "from {a} to {b} mM. Does the membrane depolarize or hyperpolarize?",
]


def nernst(ion, cin, cout):
    z = IONS[ion]["z"]
    return RT_F / z * math.log(cout / cin)


def sym(ion):
    return ion.replace("+", "").replace("-", "")


def make_problem(kind, ion, cin, cout, cout2=None, variant=0):
    """Returns dict {kind, ion, params, prompt, gold} with gold = canonical answer."""
    s = sym(ion)
    if kind == "nernst":
        e = round(nernst(ion, cin, cout))
        prompt = PROMPTS_NERNST[variant % 2].format(ion=ion, cin=cin, cout=cout, sym=s)
        gold = f"E_{s} = {e:+d} mV."
        truth = {"value": e}
    elif kind == "deltaE":
        d = nernst(ion, cin, cout2) - nernst(ion, cin, cout)
        word = "increases" if d > 0 else "decreases"
        prompt = PROMPTS_DELTAE[variant % 2].format(ion=ion, cin=cin, a=cout, b=cout2, sym=s)
        gold = f"E_{s} {word}."
        truth = {"cat": word}
    elif kind == "polarity":
        d = nernst(ion, cin, cout2) - nernst(ion, cin, cout)
        word = "depolarizes" if d > 0 else "hyperpolarizes"
        prompt = PROMPTS_POLARITY[variant % 2].format(ion=ion, a=cout, b=cout2)
        gold = f"The cell {word}."
        truth = {"cat": word}
    else:
        raise ValueError(kind)
    return {"kind": kind, "ion": ion, "cin": cin, "cout": cout, "cout2": cout2,
            "prompt": prompt, "answer": gold, "truth": truth}


NUM_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*mV")
ANY_NUM_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)")


def reward(kind, truth, text):
    """Score a model answer in [0,1]. Format credit keeps early RL from flatlining."""
    t = text.lower()
    if kind == "nernst":
        m = NUM_RE.search(text) or ANY_NUM_RE.search(text)
        if not m:
            return 0.0
        err = abs(float(m.group(1)) - truth["value"])
        if err <= 2:
            return 1.0
        if err <= 10:
            return 0.6
        if err <= 25:
            return 0.3
        return 0.05
    if kind == "deltaE":
        inc, dec = "increas" in t, "decreas" in t
        if inc == dec:                       # both or neither -> unscorable
            return 0.0
        said = "increases" if inc else "decreases"
        return 1.0 if said == truth["cat"] else 0.0
    if kind == "polarity":
        dep, hyp = "depolariz" in t and "hyperpolariz" not in t, "hyperpolariz" in t
        if dep == hyp:
            return 0.0
        said = "depolarizes" if dep else "hyperpolarizes"
        return 1.0 if said == truth["cat"] else 0.0
    raise ValueError(kind)
