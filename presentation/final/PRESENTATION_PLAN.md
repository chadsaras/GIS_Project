# Final presentation plan

**Project:** Reading the economy from space: municipal GDP from satellite embeddings, text and night lights
**Format assumed:** about 15 minutes plus questions; 18 main slides and 4 backup slides. Adjust if the time limit differs.
**Design:** the mid-term deck's system, unchanged: night slides open and close, content slides on map paper,
one colour per ingredient (embeddings teal, text violet, night lights amber, official GDP brick, POI/roads blue),
Lato, the route line in the footer, slide titles written as the message.

## What changed since the mid-term

| At the mid-term (6 Oct) | Since then (final) |
| --- | --- |
| Models V1–V12 run, RQ1–RQ4 answered | Same answers, **confirmed by a full rerun from scratch** (text gain shrinks: not robust) |
| Caption audit "in progress" | Audit done: 35% of raw captions hold a false claim, 23% after cleaning |
| Error analysis "code ready" | RQ5 answered: mines, steel mills and commuter towns are invisible; errors grow with income half as fast as with night lights |
| No uncertainty | Honest 90% ranges (89.6% held) and a satellite-only risk label |
| One state only | **Transfer test in São Paulo:** satellites keep their ranking (R² 0.33), night lights fail (R² −0.06) |
| No application | **Web application**: estimate any town with range and risk, São Paulo comparison, where satellites are blind |
| – | Negative results checked: Stage C vectors carry size, not wealth (twins, clusters) |

## Storyline

> Local GDP is missing where it matters most; satellites see everywhere. We combined two recent ideas and tested them
> strictly on 853 Brazilian municipalities. Free satellite data explain half of income differences, twice the standard
> night-light method, and keep working in a state they never saw, where night lights fail. The new ingredients add
> little, and mines stay invisible, so every estimate ships with an honest range and a warning. That is the
> application.

Arc: **Why → What changed → How → Is it honest? → What we found → Can we trust it? → Does it travel? → The application → What did not work → Take-away.**

## Slide by slide

| # | Stage | Title (the message) | Main visual | Status vs mid-term |
| --- | --- | --- | --- | --- |
| 1 | – | Reading the economy from space | night-lights hero, group, GitHub badge | same, "final presentation" |
| 2 | Problem | Local GDP is patchy; satellites see every place | official GDP map, 49× gap | same |
| 3 | Problem | Since the mid-term: from first results to a tested application | before/after columns with status chips | **new** |
| 4 | Approach | Two recent ideas, never tested together | UrbanCLIP + embeddings → this project | same |
| 5 | Approach | Three learning stages, from no labels to official GDP | pipeline diagram | same |
| 6 | Approach | Six free layers on one 1 km grid | stacked layers + key numbers | same |
| 7 | Approach | AI captions help, but a third contain a false claim | caption example + audit bars | **merged with the audit** |
| 8 | Test | Every municipality is tested in a region never seen | fold map + rotation grid | same |
| 9 | Results | Satellites explain half of GDP, twice night lights | results bar chart + RQ chips | **RQ2 softened, rerun chip** |
| 10 | Results | Predictions follow the map, but are pulled to the middle | official / predicted / error maps | same |
| 11 | Failures | The biggest misses are mines and steel mills | scatter + ten largest gaps | final-deck slide |
| 12 | Failures | Errors grow with income, half as fast as with night lights | LOWESS + three key numbers | final-deck slide |
| 13 | Application | Every estimate ships with an honest range and a warning | coverage bars + risk-group table + Catas Altas | **new** |
| 14 | Application | Trained in Minas Gerais, it still works in São Paulo; night lights do not | SP official vs predicted maps + R² bars | **new** |
| 15 | Application | The application: estimates where statistics are missing | website screenshot + three tools | **new** |
| 16 | Wrap-up | What did not work, and what it taught us | five short lessons | **new** (absorbs the calibration lesson) |
| 17 | Wrap-up | Answers to the five questions, and what we could not do | RQ table + limits | updated |
| 18 | – | Take-away (night) | four key numbers | updated |
| B1 | Backup | The text space speaks the state's economic geography | word contrast | moved to backup |
| B2 | Backup | A full rerun from scratch gives the same answers | frozen vs rerun table | **new** |
| B3 | Backup | Stage C vectors carry size, not wealth | twins and clusters tables | **new** |
| B4 | Backup | Calibration mattered more than steering | before/after chart | moved to backup |

Footer route line: Problem · Approach · Test · Results · Failures · Application · Wrap-up.

## New visual assets

| Asset | Slide | Source |
| --- | --- | --- |
| `sp_official.png`, `sp_predicted.png` | 14 | `make_figures.py` here: same colour map and scale as the Minas Gerais maps |
| `app_guess.png`, `app_transfer.png` | 15 | screenshots of `app/index.html` |
| coverage chart, R² bars, before/after | 3, 13, 14 | drawn in TikZ in `slides.tex` |
| numbers | all | `assets/numbers_final.tex` (from reports/tables) |

## Timing (about 15 minutes)

Slides 1–3: 1:30 · 4–8: 4:00 · 9–12: 3:30 · 13–15 incl. a 1-minute live demo: 3:30 · 16–18: 2:30.
