# Running the captioning on Kaggle (PLAN 4.4–4.5)

Kaggle gives two free T4 GPUs; we use them only to caption the 10,000 sampled tiles with Qwen2.5-VL-7B.
Everything else runs on the institute server.

**Inputs:** the dataset `gisecon-tiles` (one zip of 10,000 PNGs, about 5 GB) and the notebook
`notebooks/kaggle_captions.ipynb` from this repo.
**Output:** `captions_raw_base_v1.jsonl`, one line per tile.

## A. Upload the tiles (once)

The server builds the zip: `python scripts/16_kaggle_dataset.py` writes
`~/GIS/data/kaggle/gisecon-tiles/gisecon-tiles.zip`.

- **If Claude has your Kaggle API token** (`~/.kaggle/kaggle.json` on the server):
  `python scripts/16_kaggle_dataset.py --upload` creates the private dataset. Nothing to do by hand.
- **By hand:**
  1. Copy the zip to your laptop: `scp 22CS30048@10.5.18.77:GIS/data/kaggle/gisecon-tiles/gisecon-tiles.zip .`
  2. kaggle.com → **Create** → **New Dataset** → drag in `gisecon-tiles.zip`.
  3. Title `gisecon-tiles`, visibility **Private** → **Create**. Kaggle unzips it (takes a few minutes).

## B. Set up the notebook (once)

1. kaggle.com → **Create** → **New Notebook** → **File** → **Import Notebook** → upload `notebooks/kaggle_captions.ipynb`.
2. Right panel → **Session options**: *Accelerator* = **GPU T4 x2**, *Internet* = **On**
   (both need a phone-verified account).
3. Right panel → **Input** → **Add Input** → **Your Datasets** → `gisecon-tiles`.

## C. Pilot run (about 15–20 minutes)

1. Leave `LIMIT = 200` in cell 2. Click **Run All**.
2. Cell 1 must print the GPUs (two Tesla T4). Cell 4 prints the first 3 captions: they should be normal
   sentences about buildings, roads and fields. Empty or `!!!!` output means an fp16 problem: switch `MODEL`
   to the 3B version and rerun.
3. Read about 30 captions in cell 5's output. Note the **s/image** figure printed in cell 4.
4. Optional: set `PROMPT_ID = "list_v1"`, run cells 2–5 again, and compare the two prompts.
5. Download `captions_raw_base_v1.jsonl` (right panel → **Output** → ⋮ → Download) and send it, with your
   prompt choice, for a quick check before the full run.

## D. Full run (about 6–10 GPU-hours)

1. Set `LIMIT = None` (and the chosen `PROMPT_ID`) in cell 2.
2. Click **Save Version** → **Save & Run All (Commit)** → **Save**. This runs in the background with the
   browser closed, up to 12 hours.
3. Track it under **View Active Events**. Kaggle's weekly GPU quota is about 30 hours, enough for this.
4. When it finishes, open the version → **Output** → download `captions_raw_<PROMPT_ID>.jsonl`.
5. If it stopped early (12-hour limit or quota): open the notebook → **Add Input** → **Your Work** → pick
   this notebook's last version (its output holds the partial jsonl) → **Save Version** again. Cell 2 copies
   the partial file in and only the remaining tiles are captioned.

## E. Hand back

Put the file on the server at `~/GIS/data/interim/` (or send it). Then
`python scripts/12_build_texts.py --captions interim/captions_raw_base_v1.jsonl` builds the texts.
