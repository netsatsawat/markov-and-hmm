<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="pic/banner-dark.png">
    <img src="pic/banner-light.png" alt="Markov chains and hidden Markov models" width="100%">
  </picture>
</h1>

<p align="center">
  <a href="#-running-it-yourself">Run it</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="#what-the-notebooks-find">Findings</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="#-the-notebooks-in-reading-order">Notebooks</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="#-where-this-applies-in-the-real-world">Real-world uses</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://satsawat.ai/#newsletter">Newsletter</a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.9+"></a>
  <a href="notebooks/"><img src="https://img.shields.io/badge/notebooks-5%20executed-eb6834?style=for-the-badge&logo=jupyter&logoColor=white" alt="Notebooks: 5, executed"></a>
  <img src="https://img.shields.io/badge/data-bundled%2C%20runs%20offline-1baf7a?style=for-the-badge" alt="Runs offline">
  <img src="https://img.shields.io/badge/headline%20numbers-checked%20in%20CI-8a5cf6?style=for-the-badge" alt="Headline numbers checked in CI">
  <a href="https://satsawat.ai"><img src="https://img.shields.io/badge/author-satsawat.ai-e8a112?style=for-the-badge" alt="Author: satsawat.ai"></a>
</p>

A five-notebook tutorial on Markov chains, written for someone who has
never met one. A Markov chain describes something that moves between a
few conditions over time, where the next move depends only on the
condition it is in now. The whole model fits in one small table of
probabilities. That one small table answers far more than you would
expect from something so simple. In notebook 02 it works out what a lender's loans will
cost over their life. Notebook 04 uses it to predict how often an AI
agent that works through a fixed list of steps gets to the end.
Notebook 05 uses it to put a dollar value on a program meant to stop
customers leaving.

The only prerequisites are basic Python with numpy, and probability at
the level of "a fair die shows a six one time in six". You will meet
three bits of matrix maths, each explained when it shows up:
multiplying two tables, reversing a table (its inverse), and finding a
row of numbers that a table sends back unchanged (an eigenvector). You
do not need to know any of them going in.

![SPY from 2010 to 2021, each day coloured by the market mood that notebook 03's model assigned to it](pic/spy_regimes_hmm.png)

The picture above is from notebook 03. It shows SPY, the fund that
tracks the S&P 500 stock index, from 2010 to 2021, with the model's
sorting of the days into three groups laid over the price. I call each
group a regime: a market mood that lasts a while. Two of the regimes
are calm periods. They differ in one interest-rate signal, the yield
curve, which is the gap between what the US government pays to borrow
for ten years and for two years. The third regime is stress, drawn in
orange. Its bands include four periods of market trouble you can name:
the euro debt crisis, the 2015 to 2016 wobbles, the late 2018 selloff
and the COVID crash of March 2020. The model was never told any dates.
One orange band needs a note. It starts with the late 2018 selloff and
runs on through 2019 into early 2021. Prices mostly rose in 2019. In
the same year the yield-curve gap shrank to almost nothing and briefly
dipped below zero. The model's notion of stress listens to that gap as
much as to price swings. The years after 2021, covered further down,
show the same problem much worse: for three whole years the model calls
stress on every day.

## What you need to know first

Six terms carry the whole tutorial. Each notebook defines them again in
more depth when it uses them.

- state: the condition a thing is in each time you check it, at fixed
  intervals such as each day, each month or each year. A loan is good
  or risky. A subscriber is active or at risk. This page calls one
  interval a tick.
- Markov chain: a system where the chance of the next state depends only
  on the current state, not on the path that led there.
- transition matrix: one square table. Say you are in state i now. Row
  i then gives the chance of landing in each state at the next tick,
  including the chance of staying in state i. Every row adds to 1.
- absorbing state: an exit. You can enter it but never leave. Paid up,
  written off (a loan the lender has given up on collecting), churned
  (a customer who has left), done and failed are all absorbing states.
  A chain is an absorbing chain when it has at least one exit and every
  other state can eventually reach one. The states that are not exits
  are called transient, or live, states. This page says live and exit.
- fundamental matrix: a second table built from the transition matrix.
  It answers, in one step, how long until an exit on average, which
  exit, and what the trip costs. The notebooks call it N. Notebook 02
  builds it slowly, and you do not need the formula to read this page.
- hidden Markov model (HMM): a Markov chain whose state you cannot see.
  Each day you only see symptoms. Each hidden state produces its own
  typical pattern of symptoms, and the model works backward from the
  symptoms to guess the state. In notebook 03 the hidden state is a
  market regime. This page says HMM from here on.

## 🚀 Running it yourself

The five notebooks are saved in the repository together with their
results (in git terms, committed with their outputs). A notebook is a
file of code cells and their results. It opens in your browser. Open
any of them on GitHub and every table, number and chart is already
there. Nothing needs installing for that.

To run the code yourself, with Python 3.9 or newer:

```
git clone https://github.com/netsatsawat/markov-and-hmm.git && cd markov-and-hmm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tests/test_absorbing.py
```

The `pip install` line installs numpy, pandas, matplotlib, scikit-learn,
hmmlearn, Jupyter and yfinance. Jupyter is the program that opens
notebooks. The last command runs the seven tests behind the `markov/`
package. They check the algebra against small cases where the answer
can be worked out by hand, and against a simulation. You should see:

```
PASS test_absorption_probabilities_sum_to_one
PASS test_censored_runs_are_reported_not_credited
PASS test_detects_absorbing_states
PASS test_expected_cost_equals_weighted_visits
PASS test_matches_long_run_matrix_powers
PASS test_matches_simulation
PASS test_uniform_chain_matches_closed_form

7 tests passed
```

Then start Jupyter and open the notebooks in order:

```
jupyter notebook notebooks/
```

Launch that command from the repository root, exactly as written, and
open each notebook from the browser tab it opens. Each notebook's first
code cell expects the `markov/` folder to sit one level above the
notebook's working directory, and that is only true if you start this
way.

The notebooks run offline. Market data ships in
`data/market_macro.csv`, one row per trading day from 2010 to 2026. Each
run reads that file rather than the network, so the data stays the same. Each row holds SPY plus three open FRED series. FRED
is the St. Louis Fed's free public database of economic series. The
model does not use the SPY price directly. It uses the daily log
return, which the file stores as `sret`. A log return is a version of
the day's percentage move that you can add across days to get the move
over a week or a year. The first FRED series is the VIX (VIXCLS), what
the notebook calls the market's fear gauge. It is worked out from the
prices of stock options, and it rises when traders expect big moves.
Notebook 03 reads it as drowsy around 12 to 15, nervous above 20, and
above 40 as a sign that something in the market is breaking down. The
other two series, T10Y2Y and T10Y3M, describe the yield curve. A
Treasury rate is what the US government pays to borrow for a set number
of years. T10Y2Y is the 10-year rate minus the 2-year rate, and T10Y3M
is the 10-year rate minus the 3-month rate. Both are in percentage
points. A steep curve means a large gap. A flat curve means a small
one. An inverted curve means the gap is below zero, so the short rate
is higher than the long one. If you want a fresher snapshot,
`python data/build_dataset.py` rebuilds the file from sources that are
still fully open. The comment at the top of that script explains why
two of the data series from the original 2020 notebook had to be
replaced. Beyond cloning, only two things need the network: that
rebuild script and `pip install`.

## What the notebooks find

The numbers below are read from the committed notebook outputs. CI is
the automated check GitHub runs on every push. In CI,
`scripts/verify_readme_claims.py` recomputes the headline figures from
the matrices inside the notebooks: the loan book's write-off rate and
time to resolve, the six finish rates and call counts in notebook 04,
the retention program's net value, and the ranking of the four fixes
(the notebook calls them levers) in notebook 05. If one of those drifts
from the code, CI fails. The other figures on this page are quoted from
the notebooks and are not checked by CI.

### Notebook 01

Notebook 01 takes a chain with two states, A and B, and computes the
long-run share of time it spends in each, two ways. One route is exact
algebra. The eigenvector, the row of numbers the matrix sends back
unchanged, puts the share of time in A at 0.1667. Route two is one
100,000-step random walk: run the chain tick by tick with a random
number generator and count the ticks spent in A. That walk gives
0.1639, a gap of 0.0028, about three parts in a thousand. The notebook
describes that gap as the sampling wobble you would expect from a
hundred thousand random steps. It does not actually compute that
figure. Treat it as my opinion, not a measurement. The two methods
share nothing but the matrix, so a bug would have to infect both.

### Notebook 02

Notebook 02 builds a four-state loan book. A loan book is a lender's
whole set of outstanding loans. Two states are live, a good loan (GL)
and a risky loan (RL), and two are exits, paid up (PU) and written off
as a bad loan (BL). The rates are made up. I did not take them from a real
book, but their shape is the one you would see in practice. The
fundamental matrix says a risky loan resolves in just under three
years. That exact figure is 2.94 years, from adding up the RL row. It
also says a risky loan ends written off 88% of the time: 0.8775, read
from the table of ending chances that the notebook calls B. Servicing cost is what the lender
spends each year to run a loan. That runs \$120 a year for a good loan
and \$900 a year for a risky one. Averaged over a risky loan's whole
life, the servicing cost comes to \$2,265, from the fundamental matrix
times the cost per year. A simulation of twenty
thousand loans, using only the transition matrix and a random number
generator, gives 0.8792 for the write-off chance and 2.9355 years. The
differences sit at the third decimal place.

### Notebook 03

Notebook 03 first checks the HMM on made-up data where the true regimes
are known. The made-up data has two regimes, and the model sees only
the daily return. It labels 96.8% of days correctly and recovers each
regime's stickiness and size of swings to a couple of decimal places.
The notebook computes no simpler guess to compare that 96.8% against.
Then the model meets sixteen years of real data: SPY's daily log return
plus the three FRED series. Fitting means letting the model pick the
numbers inside it that best match the data. It learns from the training
years, 2010-01-05 to 2021-12-31. Those are the first twelve years of the
file. It never saw the test years, 2022-01-03 to 2026-08-04. I call that
frozen model the one fitted on the training years and never refitted.

Training gives the three regimes pictured at the top. The calm split
surprised me. I expected the model to separate strong markets from weak
ones. Instead it split calm by the shape of the yield curve. The two
calm states have almost the same size of daily swings. That split is
the notebook's most interesting finding. A model that is never told the
right answers can only sort days by the inputs you give it, and one of
those inputs was the yield curve.

The textbook way to pick the number of regimes is BIC, a score that
rewards fitting the data closely and penalises each extra regime, so
that more regimes do not win automatically. Lower is better, and the
lowest score is supposed to mark the right count. Here BIC never
bottomed out in the range tried: 26072 for one regime, 18410 for six,
still dropping. So it gave no answer. I chose three because three is
readable, then tested whether those three states meant anything on the
test years.

The test years go badly. In 2022, a year when prices fell hard, the
frozen model calls stress on 79% of days, and that is a fair call. Then
in 2023, 2024 and 2025 prices rallied. The frozen model still called
stress on 100% of days, overcalling stress for three straight years
even as prices climbed. The notebook diagnoses why. In the test years, 47% of days had an
inverted yield curve, meaning the 10-year rate minus the 2-year rate
was below zero. In the training years that gap never went below -0.04
percentage points, a hair under zero. A curve inverted that deeply was
outside anything the model had seen. The stress state allows by far
the biggest swings in the inputs, so any day that looks unlike anything
in training gets filed there.

A regime call is useful if it warns that tomorrow will move more than
usual. On the test years the frozen model does not. The average size
of the next day's move after a stress call was only 0.90 times the size
after a calm call. A warning that works gives a ratio above 1. In the
training years this model's ratio was about 2. A plain rule, "is the
VIX above 20 today?", kept a ratio above 2 on the same days. Twenty is
the level the notebook reads as nervous. This failure is the most
useful lesson in the notebook.

### Notebook 04

Notebook 04 models a pipeline agent: an AI task split into a fixed
sequence of steps, with one request to a language model (an LLM, the
kind of model behind chatbots) at each step. Each step can fail. A run
ends in done or failed. The steps are the live states, and done and
failed are the exits. With ten steps that each succeed 85% of the
time, only 19.7% of runs finish. Each run costs 10.0 model calls. A
failed run still costs the full 10 because the lab's experiment ran
every step no matter what happened before it, and the notebook keeps
that convention. The notebook tries two fixes. Both are edits to a
step's success probability. Retry runs a failed step again once. With
retry the finish rate is 79.6% at 11.5 calls per run. Verify adds a
checker after each step. That checker is one more model call. It catches
90% of bad outputs (a recall of 0.9) and reruns the step once. With
verify the finish rate is 69.8% at 21.4 calls per run, because the
checker is billed at every step. Those six numbers are three finish
rates and three call counts. Notebook 04 quotes them from my
[agent-failure-lab](https://github.com/netsatsawat/agent-failure-lab)
repository. They are typed in, and the
notebook asserts all six published numbers against its own results. CI
recomputes them the same way: the three finish rates come from the
fundamental matrix, and the three call counts come from per-step
arithmetic that counts every step as executed. The notebook also
prints what a pipeline that stops at the first failure it cannot fix
would cost, and that column is lower in every row.

The chain then ranks where engineering effort should go. Take an
eight-step profile, a list of eight steps with a success rate for each,
whose weakest step succeeds 90% of the time and whose strongest
succeeds 99% of the time. Halving the error of the weakest step repays
engineering effort ten times over, compared with doing the same work on
the strongest step. The weak step has more error to remove. Halving a 10%
error recovers five percentage points, and halving a 1% error recovers
half a point. Last, the notebook adds a fallback branch. A backup tool
takes over when step 3 fails, then rejoins the main path. Multiplying
the step success rates no longer gives the finish rate, because a run
can reach done by two routes. The fundamental matrix handles the
branched chain unchanged.

### Notebook 05

Notebook 05 models a subscriber base as a five-state chain: New,
Active, At risk, Contract and Churned. Contract means the customer
upgraded to a long-term contract, and the notebook stops tracking them
there. Contract and Churned are the exits. The rates are made up. I did
not take them from a real book, but their shape is the one you would
see in practice. Margin is the profit a customer brings in each month.
The save desk is the team that tries to win back at-risk customers, and
the save rate is the share of at-risk customers it brings back to
Active each month.

Before the money, the notebook draws the cohort retention curve: how
many of a group of customers who signed up together are still paying,
month by month. A dashboard compresses that curve into one monthly
churn rate. A single monthly churn rate hides the shape the chain
shows: the chance of leaving is high in the first month and settles
down later.

Customer lifetime value, CLV, is the total margin a customer contributes
over the whole relationship. Here it has two parts. The first part is
the expected months in each state times the margin per month. The
second part is the chance of reaching Contract times a lump \$350. A
new customer is worth \$302.94 by that formula. A Monte Carlo run,
which walks 40,000 simulated customers through the matrix with random
draws, gives \$306.20, with about \$1.50 of random noise either way
(the standard error is 1.48). The \$3 gap between the two answers is
about two of those noise units, which is the normal size of
disagreement, so the two agree.

Before launch, the notebook prices a retention program. The program
lifts the save rate from 35% to 45%, cuts at-risk churn from 25% to
15%, and costs \$6 per at-risk month. Gross uplift, the extra CLV
before the program's own cost is taken off, is \$47.06 per new
customer. The program costs \$8.00 per new customer. That is more than
the \$6 times the usual 1.2 at-risk months would suggest. The program
keeps at-risk customers there longer, and every extra month gets
billed. The program is worth net \$39 per new customer.

One honest ranking compares four competing initiatives. The notebook
calls them levers, and each one moves a single rate by one percentage
point: Active churn, month-one churn (the share of new customers who
leave in their first month, which onboarding teams exist to lower),
the save rate, and the upsell push. A point off Active churn is worth
about \$37 per new customer. A point of onboarding or save-desk
improvement is worth \$3.50 to \$4.20. The reason is occupancy: a
customer spends 9.3 of their 11.4 months in Active, so losing one
percentage point of them there costs you far more total customer-months
than losing a point in a state where customers barely linger. The
upsell push is worth almost exactly nothing. It moves the share of
Active customers who sign a contract each month from 2% to 3%, and that
moves CLV by minus 24 cents. Converting an Active customer brings in \$350 at once
but removes a customer whose expected remaining value was \$353.

## 📚 The notebooks, in reading order

Each one builds on the previous. Notebook 01 is pure mechanics. The
other four each apply the same matrix to something concrete: a loan
book, a market, an AI pipeline and a subscriber base.

- [01 · Markov chains](notebooks/01_markov_chains.ipynb). States,
  memory, and the transition matrix. You compute a two-step probability
  by hand, then discover that matrix multiplication is that same hand
  calculation done for every route at once. You watch a chain forget its
  own starting point.
- [02 · Absorbing chains and credit risk](notebooks/02_absorbing_chains_credit_risk.ipynb).
  Some states you never leave. A loan book is pushed forward year by
  year until the brute-force approach shows its limits, then the
  fundamental matrix N = (I - Q)^-1 answers everything at once and
  exactly. A simulation confirms the write-off rate and the time to
  resolve.
- [03 · Hidden Markov models](notebooks/03_hmm_regime_detection.ipynb).
  Markets have moods, but nobody rings a bell when one ends. The HMM is
  first vindicated on made-up data where the truth is known, then
  fitted to real market data. The notebook names the regimes from
  their fitted statistics rather than by hope, and then grades the model
  on years it never saw.
- [04 · Agents as absorbing chains](notebooks/04_agents_as_absorbing_chains.ipynb).
  A pipeline agent is an absorbing chain, and retries and verification
  become matrix edits. The chain answers where failing runs die and
  which step deserves the next block of engineering effort. Then it
  goes where a one-line formula cannot and shows what a fallback branch
  does to the arithmetic.
- [05 · Customer lifetime value](notebooks/05_customer_lifetime_value.ipynb).
  The business capstone. It draws cohort retention curves and the shape
  a single monthly rate hides, prices customer-months with the
  fundamental matrix to get CLV, prices a retention program end to end,
  and ranks four levers in dollars.

The repository layout:

- `notebooks/`: the five notebooks, committed with their outputs.
- `markov/`: the shared code. `AbsorbingChain` splits the matrix into its
  live and exit blocks, computes the fundamental matrix, the chance of
  each ending and the expected cost, and includes a simulation checker.
  `MarkovChain` draws the state diagrams.
- `data/`: `market_macro.csv` and the script that rebuilds it.
- `tests/test_absorbing.py`: seven tests of the algebra against cases
  with a hand-checkable answer and against simulation.
- `scripts/verify_readme_claims.py`: the CI gate that recomputes the
  headline numbers in this README from the notebooks.
- `pic/`: the banner and the two figures shown on this page.

## 💼 Where this applies in the real world

Every notebook runs on illustrative or public data, but the last four
are each a working template for a decision someone is paid to make.
Count how many of your own loans, customers or runs moved from each
state to each other state over one tick. Any customer database, loan
book or run log already holds those counts. Turn them into rates and
the analysis carries over unchanged.

- Credit and collections. Banks maintain migration matrices like
  notebook 02's. A migration matrix is a table of how loans move
  between risk grades each year. The fundamental matrix turns those
  matrices into expected default rates (the share of borrowers who stop
  paying) and time on book (how long a loan stays open). It also gives
  the lifetime servicing cost per loan. Those numbers are the raw
  ingredients of pricing, meaning deciding what to charge for a loan.
  They also feed provisioning, the money a bank sets aside for loans
  that will go bad. Stress testing, asking what happens if the economy
  turns, is one edit: raise the chance that a loan ends written off,
  then recompute.
- Subscription economics. Notebook 05 is the meeting where a retention
  program gets funded or killed. It gives CLV per segment (a group of
  similar customers), the dollar value of one point of churn moved
  anywhere in the lifecycle, and a program's net value before launch.
  CLV is also the ceiling on what acquisition, meaning winning a new
  customer, is allowed to cost. The same number disciplines marketing
  spend. The states fit any recurring business: telco subscribers, paid
  seats on business software, streaming accounts, insurance policies.
- AI agent reliability. Notebook 04's chain is how I reason about LLM
  pipelines in production. An SLA is a success rate promised in a
  contract. Fine-tuning is extra training of the model on one step's
  task. The chain says what an SLA is worth, whether retry or
  verification pays for itself, and which step should get the
  fine-tuning budget. The live-model half of that argument is in
  [agent-failure-lab](https://github.com/netsatsawat/agent-failure-lab).
- Market and macro monitoring. Notebook 03's regime machinery supports
  risk dashboards and position limits (caps on how much a trader may
  hold) that change with the regime. Its failure on the test years is
  the more useful export: a worked example of why any deployed model
  that learns without being told the right answers needs a check for
  drift and a schedule for refitting. Drift means the inputs moving
  away from what the model trained on.
- Anything with an exit state. Employee attrition (staff leaving),
  predictive maintenance (machine states ending in failure), clinical
  trial dropout, sales funnels ending in won or lost. The same few lines
  of matrix maths handle them all.

![The loan book as an absorbing chain: GL and RL are the live states, PU and BL are the exits](pic/loan_chain.png)

The diagram above is notebook 02's loan book. GL is a good loan and RL a
risky one. PU (paid up) and BL (bad loan, written off) each carry a
self-loop of probability 1, which is how an absorbing state looks on a
diagram: arrows come in and none leave.

## Limits

- The loan rates in notebook 02 and the subscriber rates in notebook 05
  are illustrative. They come from no real book or customer database.
  Only their shape is realistic.
- Notebook 04 treats every failure as retryable, which the notebook
  says makes retry look as good as it can possibly look. It also treats
  each step's failure as independent of the others. The notebook calls
  that the honest baseline to start from, and notes that real failures
  can be correlated, as when one bad document poisons several steps.
- Notebook 03's real-data model fails on the test years, and the
  notebook spends a whole section on why. There is no second-method
  check for that fit. It is checked on made-up data instead, then
  examined on the test years.
- CI runs the tests and the claim verifier, not the notebooks. The
  numbers quoted on this page are read from the committed outputs. I
  have not checked that a fresh run of the HMM fit, done by the
  hmmlearn library, lands on the same figures across library versions.

## 🧭 Why this repository exists

I first published this in 2020 as a single notebook. When I came back
to it in 2026, half of it no longer ran. The price API it depended on
had shut down, and two of its FRED data series had been discontinued or
moved behind a license. Fixing the plumbing was the excuse.

The real reason for the rewrite is that I wanted the tutorial to teach
the way I now believe technical material should be taught. Build every
idea from an example small enough to check by hand. Nearly every
important number gets computed twice, by methods that cannot share a
bug, usually algebra against simulation. And each notebook says out
loud what its model assumes and where it breaks. The notebooks apply
that standard even when the results are unflattering, and twice the
standard caught something genuinely wrong: BIC, the score for choosing
the number of regimes, kept falling as far as it was run, and a fitted
model quietly stopped making sense on years outside its training
window.

The rewrite also connects the material to my current work. I spend my
days helping enterprises ship AI systems, and the most useful mental
model I have for agent reliability is the absorbing Markov chain.
Notebook 04 makes that concrete, and notebook 05 walks the same
mathematics into a revenue meeting.

## 🔁 What changed since 2020

The original was one notebook with a loan example and an HMM fitted to
one company's stock price (GE). This revision splits it into the
five-part sequence above, replaces the dead data sources with a bundled
snapshot, adds the fundamental-matrix treatment the loan example had
been circling without landing, says plainly where the regime count was
a judgement call and where the model failed on unseen years, and adds
the bridges to AI agents and to customer economics. The concepts are
the same. The answers are now exact where they used to be approximate,
and checked where they used to be asserted.

---

Written by [Satsawat Natakarnkitkul](https://satsawat.ai), a data and AI
practitioner in Southeast Asia (ASEAN). If this repository is useful to
you, my newsletter [AI in Practice](https://satsawat.ai/#newsletter)
covers the same ground: evaluation, reliability, and local AI, with
working code.

License: MIT
