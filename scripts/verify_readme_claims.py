#!/usr/bin/env python3
"""No number in the README without a runnable path behind it.

Three assertions, wired into CI:
1. Every figure the README quotes from a notebook is recomputed here. The
   transition matrices are lifted out of the notebook JSON by name and
   pushed back through markov.AbsorbingChain, so the README, the module
   and the notebook cannot drift apart silently. Nothing is retyped.
2. Figures that only a full run can produce (the 100,000-step walk, the
   HMM's year-by-year stress calls) are read from the outputs the
   notebooks committed, never re-run.
3. The structural claims: five notebooks, executed top to bottom from 1,
   no error outputs, no network, and the data file the README describes.

Offline, stdlib plus numpy, a couple of seconds.
"""

import ast
import json
import re
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np                                       # noqa: E402

from markov import AbsorbingChain                        # noqa: E402

NOTEBOOKS = sorted((REPO / "notebooks").glob("*.ipynb"))
NB01, NB02, NB03, NB04, NB05 = (REPO / "notebooks" / n for n in (
    "01_markov_chains.ipynb",
    "02_absorbing_chains_credit_risk.ipynb",
    "03_hmm_regime_detection.ipynb",
    "04_agents_as_absorbing_chains.ipynb",
    "05_customer_lifetime_value.ipynb"))

WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
         7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
         12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
         16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
         20: "twenty"}

NETWORK = ("yfinance", "requests", "urlopen", "pandas_datareader",
           "http://", "https://", "fredapi")

problems = []


def word(count):
    """This README spells small numbers out; anything larger stays digits."""
    return WORDS.get(count, str(count))


def check(name, condition, detail=""):
    print(f"  {'ok ' if condition else 'FAIL'} {name}" +
          (f" ({detail})" if detail and not condition else ""))
    if not condition:
        problems.append(name)


def cells(path):
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def code_cells(path):
    return [c for c in cells(path) if c["cell_type"] == "code"]


def prose(path):
    """Every markdown cell of a notebook, as one string."""
    return "\n".join("".join(c["source"]) for c in cells(path)
                     if c["cell_type"] == "markdown")


def committed_output(path):
    """Everything the notebook printed when it was last executed."""
    chunks = []
    for cell in cells(path):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                chunks.append("".join(out.get("text", [])))
            else:
                chunks.append("".join(out.get("data", {})
                                      .get("text/plain", [])))
    return "\n".join(chunks)


def notebook_namespace(path, wanted):
    """Execute only the named assignments and defs of a notebook.

    The matrices, margins and step profiles live in the notebooks. Copying
    them into this file would let the two versions drift apart, which is
    the failure this script exists to prevent, so they are lifted out of
    the JSON by name and evaluated here instead.
    """
    env = {"np": np, "AbsorbingChain": AbsorbingChain}
    for cell in code_cells(path):
        src = "\n".join(line for line in "".join(cell["source"]).splitlines()
                        if not line.startswith(("%", "!")))
        for node in ast.parse(src).body:
            names = set()
            if isinstance(node, ast.FunctionDef):
                names = {node.name}
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    names |= {n.id for n in ast.walk(target)
                              if isinstance(n, ast.Name)}
            if names & wanted:
                exec(compile(ast.Module(body=[node], type_ignores=[]),
                             str(path), "exec"), env)
    missing = wanted - set(env)
    if missing:
        raise SystemExit(f"{path.name} no longer defines {sorted(missing)}")
    return env


def main() -> int:
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    # The README wraps its prose, so a quoted phrase is matched against the
    # text with its line breaks flattened. Everything else is literal.
    flat = " ".join(readme.split())

    print("the notebooks, as the badge describes them:")
    check(f"README badge says notebooks-{len(NOTEBOOKS)}%20executed",
          f"notebooks-{len(NOTEBOOKS)}%20executed" in flat)
    check(f"README calls it a {word(len(NOTEBOOKS))}-notebook tutorial",
          f"A {word(len(NOTEBOOKS))}-notebook tutorial" in flat)
    for path in NOTEBOOKS:
        counts = [c.get("execution_count") for c in code_cells(path)]
        check(f"{path.name} ran top to bottom, 1..{len(counts)}",
              counts == list(range(1, len(counts) + 1)), str(counts))
        errors = [o for c in cells(path) for o in c.get("outputs", [])
                  if o.get("output_type") == "error"]
        check(f"{path.name} committed no error output", not errors,
              f"{len(errors)} error output(s)")
        source = "\n".join("".join(c["source"]) for c in code_cells(path))
        reached = [t for t in NETWORK if t in source]
        check(f"{path.name} touches no network (README: runs offline)",
              not reached, str(reached))

    print("notebook 01, the stationary distribution:")
    nb01 = notebook_namespace(NB01, {"P2", "n_steps"})
    values, vectors = np.linalg.eig(nb01["P2"].T)
    pi = np.real(vectors[:, np.argmin(np.abs(values - 1.0))])
    pi = pi / pi.sum()
    walk = re.search(r"one ([\d,]+)-step walk: A = ([\d.]+), B = ([\d.]+)",
                     committed_output(NB01))
    check("notebook 01 committed its random-walk result", walk is not None)
    if walk:
        steps = f"{nb01['n_steps']:,}"
        check(f"the committed walk is the {steps}-step one the code sets",
              walk.group(1) == steps, walk.group(1))
        check(f"README quotes a {steps}-step random walk",
              f"{steps}-step random walk" in flat)
        gap = np.abs(np.array([float(walk.group(2)),
                               float(walk.group(3))]) - pi).max()
        parts = round(gap * 1000)
        check(f"README says walk and eigenvector agree to "
              f"{word(parts)} parts in a thousand",
              f"{word(parts)} parts in a thousand" in flat,
              f"recomputed gap is {gap:.4f}")

    print("notebook 02, the loan book:")
    nb02 = notebook_namespace(NB02, {"P", "states"})
    loans = AbsorbingChain(nb02["P"], nb02["states"])
    risky = loans.transient_states.index("RL")
    years = loans.expected_steps()[risky]
    default = loans.absorption_probabilities()[
        risky, loans.absorbing_states.index("BL")]
    check(f"README calls it a {word(len(nb02['states']))}-state loan book",
          f"{word(len(nb02['states']))}-state loan book" in flat)
    check("README: a risky loan resolves in just under three years",
          2.0 < years < 3.0 and "just under three years" in flat,
          f"recomputed {years:.2f} years")
    check(f"README: a risky loan defaults {default:.0%} of the time",
          f"{default:.0%} of the time" in flat,
          f"recomputed {default:.2%}")

    print("notebook 03, the HMM on market data:")
    rows = (REPO / "data" / "market_macro.csv").read_text(
        encoding="utf-8").strip().splitlines()
    header = rows[0].split(",")
    first, last = (date.fromisoformat(r.split(",")[0]) for r in
                   (rows[1], rows[-1]))
    series = [c for c in header if c not in ("Date", "SPY", "sret")]
    check(f"README: the bundled data runs {first.year} to {last.year}",
          f"{first.year} to {last.year}" in flat)
    check(f"README: SPY plus {word(len(series))} open FRED series",
          f"{word(len(series))} open FRED series" in flat, str(series))
    out03 = committed_output(NB03)
    window = re.search(r"train:\s*\d+ days, (\d{4}-\d\d-\d\d) to "
                       r"(\d{4}-\d\d-\d\d)", out03)
    check("notebook 03 committed its train/test split", window is not None)
    if window:
        train_start, train_end = (date.fromisoformat(g) for g in window.groups())
        # "sixteen years of real data" is the whole series the notebook works
        # through, train plus the out-of-sample tail the next sentence is about,
        # not the fit window, which is twelve. Measuring the train split here
        # and matching it against that sentence compares two different spans.
        test = re.search(r"test:[^\n]*?(\d{4}-\d\d-\d\d) to (\d{4}-\d\d-\d\d)",
                         out03) or re.search(
            r"(\d{4}-\d\d-\d\d) to (\d{4}-\d\d-\d\d)",
            out03[window.end():])
        check("notebook 03 committed its out-of-sample window", test is not None)
        if test:
            last = date.fromisoformat(test.group(2))
            # Completed years, floored. Jan 2010 to Aug 2026 is 16.6 elapsed
            # years: sixteen full ones and a tail. Rounding that to seventeen
            # would claim a year of data the series does not have.
            span = int((last - train_start).days / 365.25)
            check(f"README: the HMM is fitted to {word(span)} years of real data",
                  f"{word(span)} years of real data" in flat,
                  f"{train_start} to {last} is {span} complete years")
        # The fit window is pinned by its dates rather than a year count: it
        # spans 2010 to 2021, which is twelve calendar years but 11.99 elapsed,
        # and a check that has to pick between those is a check about rounding.
        check("the fit window is still 2010-01-05 to 2021-12-31",
              (train_start, train_end) == (date(2010, 1, 5), date(2021, 12, 31)),
              f"{train_start} to {train_end}; the sentence above measures from "
              f"this start, so a change here moves it too")
    table = re.search(r"Date\n((?:\s*\d{4}\s+[\d.]+\n)+)Name: stress share",
                      out03)
    check("notebook 03 committed its stress share per year", table is not None)
    if table:
        shares = [(int(y), float(v)) for y, v in
                  re.findall(r"(\d{4})\s+([\d.]+)", table.group(1))]
        run = best = 0
        for _, value in shares:
            run = run + 1 if value == 1.0 else 0
            best = max(best, run)
        check(f"README: the frozen model overcalls stress for "
              f"{word(best)} straight years",
              f"overcalling stress for {word(best)} straight years" in flat,
              str(shares))

    print("notebook 04, the agent pipeline:")
    nb04 = notebook_namespace(NB04, {"pipeline_chain", "effective_p",
                                     "calls_per_step", "p", "n", "r",
                                     "profile"})
    pipeline, rate, calls = (nb04["pipeline_chain"], nb04["effective_p"],
                             nb04["calls_per_step"])
    p, n = nb04["p"], nb04["n"]
    lab = {}
    for mode in ("none", "verify", "retry"):
        chain = pipeline([rate(p, mode)] * n)
        lab[(mode, "end-to-end success")] = chain.absorption_probabilities()[
            0, chain.absorbing_states.index("done")]
        lab[(mode, "calls, lab semantics")] = n * calls(p, mode)
    asserted = re.findall(
        r"table\.loc\['([^']+)', '([^']+)'\] - ([\d.]+)\) < ([\d.e+-]+)",
        "\n".join("".join(c["source"]) for c in code_cells(NB04)))
    for mode, column, target, tol in asserted:
        got = lab[(mode, column)]
        check(f"lab number reproduced: {mode} {column} = {target}",
              abs(got - float(target)) < float(tol), f"recomputed {got:.4f}")
    check(f"README: all {word(len(asserted))} published numbers of "
          f"agent-failure-lab",
          f"all {word(len(asserted))} published numbers" in flat,
          f"{len(asserted)} assertions in the notebook")

    steps = np.array(list(nb04["profile"].values()))
    base = pipeline(list(steps))
    done = base.absorbing_states.index("done")
    baseline = base.absorption_probabilities()[0, done]
    gains = []
    for i in range(len(steps)):
        halved = steps.copy()
        halved[i] = 1 - (1 - halved[i]) / 2
        chain = pipeline(list(halved))
        gains.append(chain.absorption_probabilities()[0, done] - baseline)
    leverage = max(gains) / min(gains)
    check(f"README: the weakest step repays effort ten times over "
          f"(recomputed {leverage:.1f}x)",
          leverage >= 10 and "engineering effort ten times over" in flat)
    check(f"notebook 04 calls that roughly {word(round(leverage))} times",
          f"roughly {word(round(leverage))} times" in prose(NB04),
          f"recomputed {leverage:.2f}x")

    print("notebook 05, customer lifetime value:")
    nb05 = notebook_namespace(NB05, {"P", "states", "margin", "terminal",
                                     "P_prog", "levers", "clv_new"})
    states, margin, terminal = (nb05["states"], nb05["margin"],
                                nb05["terminal"])
    subs = AbsorbingChain(nb05["P"], states)
    prog = AbsorbingChain(nb05["P_prog"], states)
    clv = subs.N @ margin + subs.absorption_probabilities() @ terminal
    clv_prog = prog.N @ margin + prog.absorption_probabilities() @ terminal
    net = (clv_prog[0] - clv[0]) - 6.0 * prog.N[0, 2]   # $6 per at-risk month
    check(f"README calls it a {word(len(states))}-state chain",
          f"{word(len(states))}-state chain" in flat)
    check(f"README: the retention program nets ${net:.0f} per new customer",
          f"net \\${net:.0f} per new customer" in flat,
          f"recomputed ${net:.2f}")

    base_clv = nb05["clv_new"](nb05["P"])
    deltas = {}
    for name, edits in nb05["levers"].items():
        edited = nb05["P"].copy()
        for i, j, delta in edits:
            edited[i, j] += delta
        deltas[name] = nb05["clv_new"](edited) - base_clv
    ranked = sorted(deltas.items(), key=lambda kv: -kv[1])
    upsell = next(v for k, v in deltas.items() if k.startswith("upsell"))
    check(f"README: {word(len(deltas))} competing initiatives, one ranking",
          f"{word(len(deltas))} competing initiatives" in flat)
    check(f"README: the engagement lever beats everything "
          f"({ranked[0][1] / ranked[1][1]:.1f}x the next best)",
          ranked[0][0].startswith("engagement")
          and ranked[0][1] / ranked[1][1] >= 8.0,
          f"top lever is {ranked[0][0]!r}")
    check("README: the upsell push is worth almost exactly nothing",
          abs(upsell) < 1.0 and "worth almost exactly nothing" in flat,
          f"recomputed ${upsell:+.2f} per new customer")

    print("what the README tells a reader to run:")
    requires = re.search(r"Python >= (\d+\.\d+)",
                         (REPO / "requirements.txt").read_text(
                             encoding="utf-8"))
    check("requirements.txt states a minimum Python", requires is not None)
    if requires:
        check(f"README badge says python-{requires.group(1)}%2B",
              f"python-{requires.group(1)}%2B" in flat)
    check("python tests/test_absorbing.py exists as written",
          (REPO / "tests" / "test_absorbing.py").exists()
          and "python tests/test_absorbing.py" in flat)
    check("pip install -r requirements.txt exists as written",
          (REPO / "requirements.txt").exists()
          and "pip install -r requirements.txt" in flat)
    check("python data/build_dataset.py exists as written",
          (REPO / "data" / "build_dataset.py").exists()
          and "python data/build_dataset.py" in flat)

    if problems:
        print(f"\n{len(problems)} README claim(s) drifted: {problems}")
        return 1
    print("\nevery quoted README number matches its artifact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
