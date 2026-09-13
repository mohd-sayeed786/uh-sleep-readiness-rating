# Take-home exercise — Senior Data Scientist

## Context

The Ring Ring measures a user's body through the night: heart rate, heart rate
variability, movement, skin temperature and blood oxygen. Every morning the app shows a
**Readiness** figure that is meant to answer one question — *how well did you recover, and what
should you do about it today?*

The score we ship today is a hand-tuned heuristic. We want to know whether a model learned from
data would do better, and what it would take to put such a model in front of users.

## The ask

**Predict how a user will report feeling when they wake up, using the night's ring data and the
context around it. Then decide what we should actually show them.**

You have ninety nights of data from a hundred and twenty users.

## What is in the box

| File | What it holds |
|---|---|
| `data/nightly_signals.csv.gz` | Five-minute sensor readings inside each sleep session (~940k rows) |
| `data/sleep_sessions.csv` | One row per recorded sleep session, with the device's own summary |
| `data/user_profiles.csv` | Age, sex, body measurements, timezone, plan |
| `data/daily_context.csv` | Steps, workouts, caffeine, alcohol, travel, and the score we show today |
| `data/morning_checkins.csv` | The user's own morning rating, 1 to 5. **This is the target.** |

`DATA_DICTIONARY.md` lists the columns and their units. It says nothing about data quality.

This is a production export, not a teaching dataset. Treat everything in it with suspicion.

## What to build

A repository we can clone and run. Inside it:

1. **A pipeline** that reads the raw files and produces a modelling table. It must be
   re-runnable end to end, not a notebook that only works if you execute the cells in the right
   order.
2. **Features.** Whatever you think earns its place. Say why.
3. **A model.** Start with something simple so you have a floor to beat. Then go further —
   including a sequence model over the five-minute signals if you judge it worth the effort. We
   care more about *why you chose it* than about which library you reached for.
4. **Tests.** `pytest`, and enough of them that we can see what you consider worth protecting.
5. **Analysis.** Where does the model work, where does it fall over, and for whom?

## The README

One README, with these sections in this order. We read them side by side across candidates, so
please keep the headings.

1. **Run it** — the commands, and roughly how long they take
2. **Data** — what you found wrong, what you did about it, what you threw away and why
3. **Features** — what you built and the reasoning
4. **Model and validation** — what you tried, how you split the data, and why that split
5. **Results** — numbers, plus honest error analysis
6. **Productization** — the section we care most about:
   - What runs in production, and when? On the phone or on a server?
   - What does the user see? A number, a word, a colour, a sentence? Defend the choice.
   - How would you know, from production data alone, whether the model is doing its job?
   - What do you monitor, and what would make you turn it off?
   - How does this behave for someone who bought the ring yesterday?
7. **Next two weeks** — what you would do with more time, ranked
8. **AI use** — which parts you used AI tools for

## Ground rules

- **Use any AI tool you like.** We do. There is one condition: you will walk us through the
  repository live and we will ask you to change things. Do not submit code you cannot defend.
- Budget **eight to twelve hours** across the week. We are not measuring stamina. If you run out
  of time, write down what you would have done — a clear "I stopped here, and here is the plan"
  scores better than a rushed attempt at everything.
- Python. Any libraries. Pin them.
- No external data is needed. Everything is in the box.
- Set your random seeds.
- The brief is deliberately short. If something is ambiguous, either ask us or write down the
  assumption you made and move on. Both are fine; guessing silently is not.

## How we assess

| | Weight |
|---|---|
| Data engineering and cleaning | 20 |
| Feature engineering | 15 |
| Modelling and validation | 20 |
| Testing and code quality | 15 |
| Analysis and communication | 15 |
| Productization and evaluation design | 15 |

A model that scores well but cannot be defended is worth less to us than an honest model with a
clear account of its limits.

## Submitting

Send a link to a private repository, or a zip. Include everything needed to run it apart from the
data. We will book ninety minutes to go through it together.

Questions to **[redacted]**.
