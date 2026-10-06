# Presenter guide: mid-term presentation

**Project:** Reading the economy from space: estimating municipal GDP from satellite embeddings, text and night lights
**Course topic:** Geospatial Reasoning using LLMs and Generative AI
**Group 2:** Hemant (22CS30029), Saras Dipak Wagh (22CS30048), Pallav Agarwal (22CS30040)
**Slides:** `presentation/midterm/slides.pdf` (12 slides)

This guide assumes you know nothing about the project. Read Part 1 first (the whole story in five minutes), then Part 2 (slide by slide), and use Part 3 (likely questions) and Part 4 (numbers to remember) to prepare.

---

## Part 1: the whole story in five minutes

### What problem are we solving?

Governments measure how rich places are using **GDP**.

> **GDP (Gross Domestic Product):** the total value of everything produced in a place in one year: goods, services, farming, mining, everything. **GDP per capita** divides it by the population, so it roughly says "how rich is the average person here".

Countries know their national GDP well. For **small areas** (districts, towns), GDP is often published years late, estimated roughly, or not measured at all. Without it, it's hard to target help to poor areas or track development.

**Satellites photograph every place on Earth regularly, with the same instruments.** If we can teach a computer to read wealth from satellite data, we get an independent, cheap, up-to-date estimate for every town.

### Where did we test it?

In **Minas Gerais**, a state in Brazil with **853 municipalities**.

> **Municipality:** Brazil's local administrative unit: a town or city together with its surrounding countryside. Think of it as a small district.

Brazil's statistics office, **IBGE**, publishes official GDP for every municipality every year. That gives us an **answer key**: we can check our satellite guesses against the truth.

### What did we build?

A three-step machine-learning pipeline.

> **Machine learning:** instead of writing rules by hand, we show the computer many examples and it learns the patterns itself.
> **Pipeline:** a chain of steps where each step's output feeds the next.

1. **Stage A: teach the satellite data using text.** An AI model writes descriptions (captions) of satellite photos. We use them to teach the satellite numbers what they show ("houses", "farmland", "river").
2. **Stage B: point it towards economic activity using night lights.** Bright places at night tend to be economically active, so we use brightness as a practice task.
3. **Stage C: predict each town's GDP.** Combine all the 1 km squares inside a town and predict its GDP per capita.

### What did we find?

- Our best models explain **about half (0.50–0.52)** of the differences in wealth between towns. The classic night-lights method explains about a quarter (0.27).
- **Text (AI captions) gave the biggest improvement,** but the gain is borderline in the statistical test.
- **Night-light steering and road/shop data added nothing clear.**
- A simple **correction step (calibration)** improved results more than any fancy ingredient.
- We always test on **regions the model never saw**, so the results are honest.

### What's left?

- **RQ5:** studying where and why the model fails, for example mining towns.
- **The caption audit:** checking AI captions by hand to measure their accuracy.
- **The final report.**

---

## Part 2: slide by slide

For each slide: **what's on it**, **what it means** (with the jargon explained), and **what to say**.

### The footer, on every slide

At the bottom right there's a small **route line** with stations: *Problem → Approach → Data → Test → Results → Next*. The amber dot shows where you are in the talk. You don't need to mention it; it simply helps the audience follow along.

---

### Slide 1: Title

**What's on it**
- The course topic: *Geospatial Reasoning using LLMs and Generative AI*.
- The project title and subtitle, the group members, and an amber badge with the GitHub link to our code.
- On the right, a **night-lights map** of Minas Gerais. Each dot of light is a 1 km square that glows at night; the big bright patch is the capital, Belo Horizonte.

**Jargon**
- **Geospatial:** related to locations on Earth: maps, satellite images, coordinates.
- **LLM (Large Language Model):** an AI that reads and writes text, like ChatGPT.
- **Generative AI:** AI that creates new content (text, images). We use it to *write captions* for satellite photos.
- **Satellite embeddings:** explained on slide 3; in short, numbers that summarise what a patch of land looks like.
- **VIIRS:** the satellite instrument that measures night-time light.

**What to say**
> "Our project asks whether we can estimate how rich each town is using only free satellite data, with generative AI writing descriptions of satellite images. We test it on 853 municipalities in Minas Gerais, Brazil. All our code is on GitHub at this link."

**How it fits the course topic:** we use a **vision-language model** (an AI that looks at an image and writes text about it) to generate captions, and we use that text to help the computer reason about geography. That is "geospatial reasoning with LLMs and generative AI".

---

### Slide 2: "Local GDP is patchy; satellites see every place"

**What's on it**
- **Left:** a map of official GDP per capita for every municipality in 2022. Dark red and purple are rich; pale yellow is poor. It includes a **scale bar** (how many km a length on the map represents) and a **north arrow**.
- **Right:** a big **49×**, plus two bullet points and the project goal.

**What it means**
- Inside one state, the richest town, **Catas Altas** (mining, R$ 391,000 per person), produces **49 times** more per person than the poorest, **São João das Missões** (R$ 8,000). That huge range makes the task hard and realistic.
- The map shows a pattern: richer west and centre (agribusiness, mining, industry), poorer north-east.
- **R$** is the Brazilian real, the currency.

**Jargon**
- **Sub-national:** below the country level (states, districts, towns).

**What to say**
> "GDP for small areas is often late, roughly estimated, or missing, especially where statistics systems are weak. Satellites, by contrast, see every place every year with the same sensor. Inside this one state, GDP per capita varies 49-fold. Our goal is to estimate it for all 853 municipalities from free satellite data, and to test that honestly."

---

### Slide 3: "Two recent ideas, never tested together"

**What's on it:** two cards on the left (two research papers) with arrows into a dark box on the right ("This project").

**Card 1: UrbanCLIP** (Yan et al., WWW 2024)
- A language model writes a **caption** for every satellite image. Then images and captions are **aligned**: the model learns which images go with which words.
- **Its weaknesses (in red):**
  - Captions improved GDP prediction by only **+0.5%**.
  - It tested with a **random split**, which lets neighbouring images leak into the test (explained on slide 7).
- **WWW** is a major computer-science conference.

**Card 2: Satellite embeddings** (Yue, Zhao & Hu, 2026)
- Uses **Google's satellite embeddings** and **steers** them with night lights towards economic meaning.
- **Its weaknesses:** tested only on whole **countries** (few examples, very coarse), and used no text and no road or shop data.

**Jargon**
- **Satellite embedding:** Google has processed years of satellite imagery into **64 numbers for every 10 m patch of Earth**. The 64 numbers act as a "fingerprint" of what the land looks like: forest, crops, city, water. You can't read the numbers individually, but similar places get similar numbers. The product is called the **Google Satellite Embedding (GSED)**, also known as AlphaEarth.
- **Steering:** training the embeddings to predict night-light brightness, so they emphasise economic activity (more on slide 4).
- **Align / alignment:** teaching a model that an image and its matching description belong together, so their numbers move closer to each other.
- **POI (Point of Interest):** shops, schools, hospitals, banks and similar, as marked on maps.
- **Ablation:** switching off one ingredient at a time to see how much it matters, like removing one ingredient from a recipe to see whether the taste changes.

**What to say**
> "We build on two recent papers. UrbanCLIP showed AI-written captions of satellite images can help, but the gain for GDP was tiny and its testing was leaky. The second paper used Google's satellite embeddings steered by night lights, but only for whole countries and without text. Nobody has combined the two. We combine them in one pipeline, at the municipality level, test honestly, and switch off each ingredient to measure what it really adds."

**The colour code starts here:** each ingredient keeps one colour for the rest of the talk.
- teal: satellite embeddings (images)
- violet: text (captions)
- amber: night lights
- blue: POIs and roads
- brick red: official GDP

---

### Slide 4: "Three learning stages, from no labels to official GDP"

**What's on it**
- Four small images on top, the **inputs**: embeddings, a captioned photo, night lights, official GDP.
- Three boxes, **Stages A, B and C**, then a dark box: "log GDP per capita".
- Coloured labels under the stages: "no labels", "cheap proxy label", "853 real labels".

**Jargon**
- **Label:** the correct answer used to teach a model. For example, "this town's GDP is R$ 20,000".
- **Training:** the phase where the model learns from labelled examples.
- **Proxy label:** a stand-in answer that is easier to get than the real one. Night-light brightness is a *proxy* for economic activity.
- **log GDP:** we predict the *logarithm* of GDP per capita rather than the raw number. A log turns "times" into "plus": the jump from R$ 10k to 20k counts the same as from 100k to 200k. This stops a few super-rich towns from dominating, and is standard in economics.
- **Cell:** one 1 km × 1 km square of our grid.

**The three stages:**

| Stage | Plain explanation | Teacher (label) |
|---|---|---|
| **A: image–text alignment** | Teaches the satellite numbers to match their captions, so the numbers become more meaningful ("this pattern = dense houses") | **no labels needed**: captions come for free from the AI |
| **B: night-light steering** | A small network learns to predict each cell's night brightness from its numbers. We keep its **inner layer** (32 numbers per cell) as an improved description that emphasises activity | **cheap proxy label**: brightness is known for all 575,000 cells |
| **C: add up the cells** | Combines all cells inside a municipality and predicts its GDP per capita | **853 real labels**: official GDP, the scarce, expensive truth |

> **Neural network:** a type of machine-learning model loosely inspired by the brain: layers of simple calculations. The **inner layer** is an intermediate step where the network has built its own summary of the input.

**Key idea:** the labels get scarcer at each stage. Stage A needs no labels, Stage B uses a free, plentiful stand-in, and only Stage C uses the 853 precious official numbers. That way we learn as much as possible before touching the scarce data.

**The note at the bottom:** *"Captions are needed only in training."* Once trained, the model predicts any area from its satellite numbers alone, with no new captions needed. So the method is cheap to apply anywhere.

**What to say**
> "Our pipeline has three stages. Stage A uses AI captions to teach the satellite numbers what they show, with no labels needed. Stage B uses night-light brightness, which we have for every cell, to steer the numbers towards economic activity. Stage C adds up all the cells in each municipality and predicts GDP using the 853 official values. Captions are needed only in training; afterwards the model works on satellite data alone."

---

### Slide 5: "Six free layers on one 1 km grid"

**What's on it:** six stacked maps (the classic GIS "layer stack") and four big numbers.

> **GIS (Geographic Information System):** software and methods for storing and analysing map data. GIS data is organised as **layers**, stacked like transparent sheets over the same area.
> **Grid:** we cover the state with squares of 1 km × 1 km and put every dataset onto the same squares, so they line up exactly.

**The six layers, bottom to top:**

| Layer | Source | What it is |
|---|---|---|
| Satellite embeddings | Google GSED | 64 numbers per 10 m patch, averaged to each 1 km cell |
| Captioned tiles | Esri imagery + Qwen2.5-VL | 10,000 satellite photos with AI descriptions |
| Buildings | Microsoft + OpenStreetMap | outlines of about 10 million buildings |
| Roads and POIs | OpenStreetMap | the road network, shops, schools and so on |
| Night lights | VIIRS 2022 | night-time brightness |
| GDP per capita (the target) | IBGE 2022 | official numbers, the answer key |

> **OpenStreetMap (OSM):** a free world map made by volunteers, like Wikipedia for maps.
> **Esri World Imagery:** a high-resolution satellite photo map (about 1 m per pixel, so you can see houses).
> **Qwen2.5-VL-7B:** the AI that writes the captions. It is a **vision-language model**, meaning it looks at an image and writes text about it. "7B" means about 7 billion internal numbers (parameters), a mid-sized model.
> **Target:** the thing we try to predict.

**The big numbers:**
- **853** municipalities.
- **575k** 1 km cells. These are the cells kept after removing empty or water-only land.
- **10,000** captioned tiles. A **tile** is the satellite photo of one cell.
- **10M** building footprints.

**The note:** *"EPSG:31983"* is the code of the map projection we use.
> **Map projection:** a method for flattening the round Earth onto a flat map. EPSG:31983 is the standard one for this part of Brazil; it keeps distances accurate, so every cell really is 1 km × 1 km.

"The target is official statistics, not night lights" means we check our answers against real GDP, not against another satellite measurement.

**What to say**
> "We built a dataset of six free layers, all aligned on the same 1 km grid: Google's satellite embeddings, 10,000 captioned satellite photos, building outlines, roads and points of interest, night lights, and official GDP as the target. In total, 575,000 cells across 853 municipalities."

---

### Slide 6: "An AI model describes each tile; map data vetoes false claims"

**What's on it**
- **Left:** a real satellite photo, a farm with a large circular field, circled in red with the label *"centre-pivot irrigation, not a mine"*.
- **Middle:** the AI's caption, sentence by sentence. Green ✓ means kept; red ✗ with strike-through means removed, with the reason.
- **Right:** a bar chart of how many sentences each rule removed, plus "84% kept of 60,023 sentences".

**What it means**
- AI captioning models sometimes **hallucinate**.
  > **Hallucination:** when an AI confidently states something false. Here, the model called a circular irrigated field "a large circular mine pit".
- So we check every sentence automatically against real map data:
  - It claims a mine or industry, but the land-cover map shows farmland: **removed**.
  - It claims a road, but OpenStreetMap shows no road there: **removed**.
  - The same checks run for buildings, water, farmland and urban areas.
  - **Filler sentences** with no information ("The overall scene appears to be…") are removed too.
- **Centre-pivot irrigation:** a farming method where a long rotating sprinkler arm waters a circle of crops, which looks like a big circle from above.

**The numbers**
- 60,023 sentences in 10,000 captions; **84% kept**.
- The biggest removal reasons were filler (4,183), then industry or mine, building, water and road claims (each about 1,000), then farmland (732).
- *"Image–text matching removes 246 more captions":* a separate AI check (RemoteCLIP) measures whether the caption fits the image at all, and removes the 246 worst-matching captions.
  > **CLIP / RemoteCLIP:** an AI that scores how well a piece of text matches an image. RemoteCLIP is a version trained on satellite images.
- *"A blind hand audit is in progress":* we're labelling a sample of 1,801 sentences by hand to measure how accurate the captions and our checks really are.
  > **Blind:** the person labelling doesn't know whether a sentence was kept or removed, so their judgement isn't biased.

**What to say**
> "We captioned 10,000 tiles with Qwen2.5-VL, an open vision-language model. It sometimes invents things: here it called a circular irrigated field a mine. So we automatically check each sentence against land-cover maps, roads and building data, and drop contradicted claims and empty filler. 84% of sentences survive. A blind hand audit to measure caption accuracy is in progress."

---

### Slide 7: "Every municipality is tested in a region never seen"

**What's on it**
- **Left:** a map coloured in 5 colours (5 groups). White outlines show **70 official regions**.
- **Right:** a grid of 5 rounds × 5 folds, coloured train (grey), validate (amber) and test (red).

**Jargon**
- **Fold:** one of 5 groups we split the data into.
- **Cross-validation:** repeat training and testing several times, each time hiding a different group for testing, so that every town gets tested once.
- **Spatial cross-validation:** the groups are made of whole geographic regions, not random towns.
- **Training set:** what the model learns from.
- **Validation set:** used to tune settings and pick options, never for final scoring.
- **Test set:** used only once, at the end, to measure the score.

**How it works**
- The 853 towns belong to 70 official IBGE "immediate regions" (groups of neighbouring towns). These are grouped into **5 folds**.
- In each **round**: 3 folds train, 1 fold validates, **1 fold is tested**. All three stages are retrained from scratch every round.
- After 5 rounds, every municipality has been tested exactly once, by a model that never saw its region.

**Why this matters (the key honesty point)**
- Neighbouring places look alike: same climate, same crops, same economy.
- If you split randomly, a test town's neighbour is probably in training, and the model can "cheat" by recognising the neighbourhood. That inflates scores. UrbanCLIP split randomly.
- Holding out **whole regions** prevents this.
  > **Data leakage:** when information from the test data sneaks into training, making results look better than they really are.

**The note:** "block bootstrap over regions (95% intervals)".
> **Bootstrap:** a statistics technique that re-runs a comparison many times on random re-samples of the data, to see how much the result could change by luck.
> **Block bootstrap:** the re-sampling picks whole regions (blocks), not single towns, because neighbouring towns aren't independent.
> **95% interval (confidence interval):** the range where the true difference probably lies. If the range includes zero, we can't be sure there is any difference.

**What to say**
> "To test honestly, we hold out whole regions. The 70 official regions are grouped into 5 folds; each round trains on 3, tunes on 1 and tests on 1, and all stages are retrained. Every municipality is tested once, by a model that never saw its region. This avoids the leakage in UrbanCLIP's random split, where look-alike neighbours sat in both training and test."

---

### Slide 8: "Text gives the biggest gain; the hybrid explains half of GDP"

This is the main results slide. Read this section carefully.

**What's on it**
- **Left:** horizontal bars, one per model, showing its score (R²). Next to each name, four dots show which ingredients it uses (image, text, lights, POI).
- **Right:** the big number **0.52**, then four research-question results (RQ1–RQ4).

**The score: R² ("R-squared")**
> **R²:** how much of the rich-versus-poor differences between towns the model gets right, from 0 to 1.
> - 0 means useless (no better than guessing the average for every town).
> - 1 means perfect.
> - 0.52 means it captures about half the differences.
>
> **"On unseen regions"** means measured only on test towns, as described on slide 7.

**The models (what V1, V2, … mean)**

Each **V** (version) is one recipe:

| Model | Plain meaning | R² |
|---|---|---|
| **Hybrid** (V10) | Picks the best model on validation each round, optionally mixed with night lights (see the note below) | **0.52** |
| + text + steering + POI (V7) | Stages A + B + C, plus roads and points of interest | 0.50 |
| + text + steering (V6) | Stages A + B + C | 0.50 |
| Full, uncleaned captions (V8) | Like V7, but with the AI captions *before* our cleaning | 0.50 |
| Full, fact sentences only (V9) | Like V7, but the text is only sentences generated from map data, no AI captions | 0.48 |
| + text alignment (V5) | Stages A + C | 0.48 |
| Embeddings + Stage C (V3) | Satellite numbers with our Stage C only | 0.46 |
| + night-light steering (V4) | Stages B + C | 0.44 |
| LightGBM, mean embedding (V12) | A standard method on each town's average satellite numbers | 0.39 |
| Ridge, mean embedding (V11) | A simpler standard method on the same input | 0.35 |
| Night lights only (V1) | The classic method: brightness per person | 0.27 |
| Land-cover shares (V2) | Share of forest, crops, buildings and so on per town | 0.12 |

- **Grey bars** are simple **baselines**.
  > **Baseline:** a standard, simple method used as the minimum to beat.
- **Teal bars** are our pipeline. The **gold bar** is the hybrid.
  > **Ridge regression:** a classic statistical formula that weighs each input, with a brake against over-fitting.
  > **Over-fitting:** memorising the training examples instead of learning general patterns, so the model fails on new data.
  > **LightGBM:** a popular, stronger method built from many small decision trees.
  > **Mean embedding:** the average of the satellite numbers over all cells in a town.

**About the hybrid (important, in case you're asked)**

In each of the 5 rounds, the hybrid:
1. picks whichever of V3–V9 had the lowest error on the **validation** towns;
2. tries mixing in the night-lights model at 0%, 25%, 50%, 75% or 100%, and keeps the best mix on validation.

What it actually chose:

| Round | Model picked | Night lights mixed in |
|---|---|---|
| 1 | V6 (text) | 0% |
| 2 | V5 (text) | 0% |
| 3 | V8 (text) | 0% |
| 4 | V3 (no text) | 25% |
| 5 | V3 (no text) | 25% |

So in 3 of 5 rounds the hybrid **was a text model with no night lights**. The slide's label ("blended with night lights") and its dots ("image + lights") understate how much text it used. If asked, say: *"The hybrid is a validation-chosen best model, which was a text model in three of five rounds, with night lights added only lightly in the other two."*

**The four research questions**

Each line shows **ΔRMSE** and a **95% CI**:
> **RMSE (Root Mean Square Error):** the typical size of a wrong guess, on the log scale. Lower is better.
> **ΔRMSE ("delta RMSE"):** the change in error from adding an ingredient. **Negative means it helped.**
> **95% CI:** the range the true change probably lies in. If the range **crosses zero**, the effect isn't proven.

| Question | Comparison | ΔRMSE and 95% CI | Verdict | Plain meaning |
|---|---|---|---|---|
| **RQ2: does text help?** | V6 vs V4 | −0.025 [−0.050, +0.004] | largest gain, **borderline** | Text gave the biggest error reduction, but the range just touches zero: very likely real, not proven |
| **RQ1: does steering help?** | V4 vs V3 | +0.007 [−0.028, +0.041] | no clear effect | Night-light steering didn't help |
| **RQ3: do POIs and roads help?** | V7 vs V6 | −0.001 [−0.017, +0.016] | nothing added | Captions probably already describe roads and buildings |
| **RQ4: does the hybrid beat the best single model?** | V10 vs V8 | −0.009 [−0.035, +0.015] | best, but **not significant** | Its lead is within the margin of luck |

> **Significant (statistically significant):** the effect is big enough that it's unlikely to be luck.

**What to say**
> "Night lights alone explain about a quarter of the differences between towns. Satellite embeddings with standard methods reach 0.35 to 0.39, our pipeline 0.46, and adding text brings it to about 0.50. The text gain is the largest of any ingredient, though borderline significant. Steering and roads or POIs add nothing measurable. The best overall score, 0.52, comes from the hybrid, which picks the best model on validation, mostly a text model, but its edge over the best single model is within the margin of error."

---

### Slide 9: "Predictions follow the map, but are pulled towards the middle"

**What's on it:** three maps of the state.
1. **Official:** real GDP per capita.
2. **Predicted (hybrid):** our test predictions, on the same colour scale.
3. **Error:** teal means we predicted too low; red means too high; pale means about right.

Plus **75%** and two findings.

**What it means**
- The first two maps look broadly similar: the model captures the big pattern (richer west and centre, poorer north-east).
- **75% of municipalities** are predicted within a **factor of 1.5** of the official value. For example, if the truth is R$ 20k, the guess falls between R$ 13.3k and 30k.
- **Pulled towards the middle:** the model plays safe.
  - **Very rich towns are predicted too low (teal),** especially mining and agribusiness towns. A mine produces enormous GDP but looks like a patch of bare ground from space.
  - **Poor towns in the north-east are predicted too high (red).**
- Understanding these errors is **RQ5**, the next phase.

**The colour bars:** "÷2.7" to "×2.7" means predicted values from 2.7 times too low to 2.7 times too high.

**What to say**
> "Here are the official values, our predictions for unseen regions, and the error. The model gets the broad pattern right, and 75% of municipalities are within a factor of 1.5. But it's pulled towards the average: rich mining and agribusiness towns are under-predicted, and the poor north-east is over-predicted. Explaining these errors is our next research question."

---

### Slide 10: "Calibration mattered more than night-light steering"

**What's on it**
- **Left:** a chart with two rows, each showing the error before (hollow circle) and after (filled circle) a correction.
- **Right:** three numbered lessons.

**Jargon**
- **Calibration:** correcting systematic errors in predictions. Like a bathroom scale that always reads 2 kg heavy: you subtract 2 kg. Ours was a straight-line correction, *corrected = a + b × prediction*, learned on validation towns.
- **Over-spread:** the first models exaggerated, predicting rich towns too rich and poor towns too poor. The correction shrinks the guesses back towards the middle.

**The chart**

| Model | Error before | Error after | Change |
|---|---|---|---|
| Embeddings + C (V3) | 0.554 | 0.439 | **−21%** |
| + steering (V4) | 0.491 | 0.446 | −9% |

**The three lessons**
1. **Predictions were over-spread.** The simple correction cut error by about 21%, more than any new ingredient.
2. **Honest testing changes conclusions.** Before calibration, steering seemed to help (0.491 against 0.554). After calibration, both are about equal. Steering was only accidentally reducing the exaggeration; it wasn't adding real knowledge.
3. **Captions need checking.** The AI invents mines, roads and farmland on plain tiles, and our data checks remove most of this.

**Is calibration cheating?** No. The correction is learned only from **validation** towns and then applied unchanged to the **test** towns, which never influence it. That's standard practice.

**What to say**
> "Our biggest lesson: our first models exaggerated differences between rich and poor towns. A simple straight-line correction, learned only on validation towns, cut error by 21%. It also changed a conclusion: night-light steering looked helpful before the correction, but that benefit vanished afterwards. And AI captions need checking, which is why we built the automatic filters."

---

### Slide 11: "Status"

**What's on it**
- **Left:** progress bars.
  - Data, grid and folds: **done**.
  - Captions and cleaning: **85%, audit left**.
  - Models and experiments: **done**, all 12 versions.
  - Error analysis and maps: **30%, code ready**.
- **Right:** research-question status (RQ1–RQ4 answered, RQ5 next) and the next steps.

**What to say**
> "Data, models and all experiments are complete; four of our five research questions are answered. What's left is the hand audit of captions, the error analysis (RQ5: where and why the model fails, with detailed 1 km maps) and the final report."

---

### Slide 12: Closing

**What's on it:** the main message, four key numbers, a summary line, "Questions?" and the GitHub link.

**The four numbers**

| Number | Meaning |
|---|---|
| **0.52** | R² of the hybrid on unseen regions: explains about half the wealth differences |
| **0.27** | R² of night lights alone, the classic method |
| **−21%** | error reduction from calibration |
| **84%** | caption sentences that passed the automatic checks |

**What to say**
> "To summarise: satellite embeddings, AI captions and night lights, tested honestly on regions never seen, explain about half of the variation in local GDP, roughly double the classic night-lights approach. Captions gave the largest gain; steering and POI data added nothing clear. Next, we'll study where the model fails. Thank you. Questions?"

---

## Part 3: likely questions, with simple answers

**Q: Why Minas Gerais?**
It has the most municipalities of any Brazilian state (853), so plenty of training examples. It has reliable official GDP for 2022 and a 2022 census. GDP per capita varies 49-fold, so the test is hard and realistic. And it is a manageable size for our computing budget: all of Brazil would have been too heavy.

**Q: Why not India?**
India lacks consistent, recent district-level GDP. Only some states publish it, with different years and methods, and the last census was in 2011, so there's no reliable answer key. We validate the method where the truth is known; it can later be applied to data-poor places like India, because all the inputs are global and free.

**Q: What is a satellite embedding, exactly?**
Google processed years of satellite imagery into 64 numbers per 10 m patch of Earth, a fingerprint of what the land looks like. Similar places get similar numbers. We average them over each 1 km cell.

**Q: Why use captions if the model doesn't need them for prediction?**
Captions are a teaching aid. They help the model learn what the satellite numbers mean during training. Once trained, it predicts from satellite numbers alone, so it's cheap to use anywhere.

**Q: Why only 10,000 captions out of 575,000 cells?**
Captioning needs a GPU and takes time, and we used Kaggle's free GPUs. 10,000 was the affordable amount, and enough to teach the patterns. The 10,000 were chosen evenly across 25 land types (5 levels of built-up × 5 levels of night brightness) so that towns and villages aren't drowned out by empty pasture.

**Q: How do you know the captions are correct?**
Two automatic layers. First, each sentence is checked against land-cover, road and building data. Second, RemoteCLIP checks that each caption matches its image. A blind hand audit of 1,801 sentences is underway to measure accuracy directly.

**Q: Did cleaning the captions improve accuracy?**
No. The model trained on uncleaned captions (V8) scored the same, 0.50. Cleaning improves the trustworthiness of the text, not the GDP score; the model tolerates some noise.

**Q: Isn't calibration on validation data cheating?**
No. It's learned only from validation towns and applied unchanged to test towns, which never influence anything. That's standard practice, like tuning any setting.

**Q: Why didn't night-light steering help?**
Likely reasons:
- Mining and agribusiness produce lots of GDP but little light, so steering towards light can pull attention away from those places.
- Night lights are already used directly in the hybrid.
- Its apparent benefit disappeared once we fixed the over-spread predictions.

**Q: What does "the hybrid" mean?**
In each round, it picks the best model on validation and optionally mixes in the night-lights model. It picked a text model in 3 of 5 rounds (with no night lights), and an image-only model with 25% night lights in 2 rounds.

**Q: Is 0.52 good?**
It's double the classic night-lights method (0.27), achieved under strict testing on unseen regions. Results in papers that use random splits look higher, partly because of leakage.

**Q: Why is the text effect "borderline"?**
Its 95% interval, [−0.050, +0.004], just crosses zero. The improvement is consistent but not large enough to rule out luck with complete certainty.

**Q: What is next?**
RQ5 (error analysis: why mining and agribusiness towns are under-predicted, with 1 km maps), the caption audit results, and the final report.

**Q: Where is the code?**
github.com/chadsaras/GIS_Project. Everything is reproducible from scripts.

---

## Part 4: numbers to remember

| What | Number |
|---|---|
| Municipalities | 853 |
| 1 km cells (kept) | about 575,000 |
| Captioned tiles | 10,000 |
| Caption sentences | 60,023, of which 84% kept |
| Building footprints | about 10 million |
| Official regions / folds | 70 regions, 5 folds |
| Richest vs poorest town | 49× (Catas Altas R$ 391k vs São João das Missões R$ 8k) |
| Night lights only | R² 0.27 |
| Our pipeline, images only | R² 0.46 |
| With text | R² 0.48–0.50 |
| Best (hybrid) | R² 0.52 |
| Calibration gain | −21% error |
| Within a factor of 1.5 | 75% of municipalities |

---

## Part 5: quick glossary

| Term | Meaning |
|---|---|
| **Ablation** | removing one ingredient to see its effect |
| **Alignment** | teaching a model that an image and its text belong together |
| **Baseline** | a simple standard method used for comparison |
| **Block bootstrap** | resampling whole regions many times to estimate uncertainty |
| **Calibration** | a simple correction of systematic prediction errors |
| **Caption** | an AI-written description of a satellite photo |
| **Cell** | one 1 km × 1 km grid square |
| **Confidence interval (CI)** | the likely range of the true effect; crossing zero means not proven |
| **Cross-validation** | repeated train/test splits so every item is tested once |
| **Data leakage** | test information sneaking into training, which inflates results |
| **Embedding** | a list of numbers summarising something (here, land appearance) |
| **Fold** | one of the 5 groups of regions |
| **GDP per capita** | economic output per person |
| **GIS** | Geographic Information System: map data and analysis |
| **GSED** | Google Satellite Embedding dataset (64 numbers per 10 m patch) |
| **Hallucination** | an AI confidently stating something false |
| **IBGE** | Brazil's national statistics office |
| **Label** | the correct answer used for training |
| **LLM** | Large Language Model: an AI that reads and writes text |
| **log GDP** | the logarithm of GDP, so ratios are treated equally |
| **Municipality** | a Brazilian town plus its countryside |
| **Night lights (VIIRS)** | satellite measurements of night-time brightness |
| **OSM** | OpenStreetMap, the free volunteer world map |
| **Over-fitting** | memorising training data instead of learning general patterns |
| **POI** | Point of Interest: shops, schools, hospitals and so on |
| **Proxy label** | a cheap stand-in for the real answer |
| **R²** | share of differences explained, from 0 (useless) to 1 (perfect) |
| **RMSE** | the typical size of a prediction error; lower is better |
| **Spatial cross-validation** | holding out whole regions, not random items |
| **Steering (Stage B)** | using night lights to point features towards economic activity |
| **Test set** | data used only for final scoring |
| **Tile** | the satellite photo of one cell |
| **Validation set** | data used to tune choices, never for final scoring |
| **Vision-language model** | an AI that looks at images and writes text (here, Qwen2.5-VL) |
