# Phase 4.4-4.5: caption the tiles with Qwen2.5-VL on Kaggle (2x T4). HEAVY: ~6-10 GPU-hours for 10,000 tiles.
#
# How to run:
#   1. Zip ~/GIS/data/raw/tiles and upload it as a private Kaggle dataset.
#   2. New notebook -> Add data -> that dataset; Settings: Accelerator "GPU T4 x2", Internet on.
#   3. Paste this file into one cell (or split at the "# %%" marks) and run.
#      Pilot first (PLAN 4.4): LIMIT = 200, run once per prompt id, read ~30 outputs of each.
#   4. Download /kaggle/working/captions_raw_<PROMPT_ID>.jsonl to ~/GIS/data/interim/ on the server.
#   Check the first printed captions: fp16 on T4 can produce empty or repeated-character output; if so,
#   switch to torch_dtype=torch.float32 for the vision tower or use the 3B model.
# Resumable: cells already in the JSONL are skipped, so rerun after a session timeout.

# %%
import glob
import json
import random
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"  # fallback if too slow: "Qwen/Qwen2.5-VL-3B-Instruct"
PROMPTS = {
    "base_v1": ("Describe this 1 km x 1 km satellite image in 4-6 sentences. Cover buildings (density, size, roof "
                "types), roads, green areas, water, farmland, industry or mining, and signs of human activity. Only "
                "describe what is clearly visible; do not guess place names."),
    "list_v1": ("You are looking at a 1 km x 1 km satellite image. In 4-6 short factual sentences, state: how built-up "
                "it is and what the buildings look like; the roads; vegetation and farmland; any water; any industry, "
                "mining or large facilities. Say 'none visible' for anything absent. No place names, no guesses."),
}
PROMPT_ID = "base_v1"
LIMIT = None          # 200 for the pilot
BATCH = 8             # lower to 4 if CUDA runs out of memory
MAX_NEW, TEMPERATURE, SEED = 200, 0.2, 42
TILES = sorted(glob.glob("/kaggle/input/**/*.png", recursive=True))
if LIMIT:  # pilot: a seeded random sample, not the first files (sorted by cell_id = northernmost cells)
    TILES = sorted(random.Random(SEED).sample(TILES, min(LIMIT, len(TILES))))
OUT = Path(f"/kaggle/working/captions_raw_{PROMPT_ID}.jsonl")

# %%
torch.manual_seed(SEED)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto",
                                                           attn_implementation="sdpa")
proc = AutoProcessor.from_pretrained(MODEL, max_pixels=768 * 768)
proc.tokenizer.padding_side = "left"
chat = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text",
                                  "text": PROMPTS[PROMPT_ID]}]}], tokenize=False, add_generation_prompt=True)

done = {json.loads(line)["cell_id"] for line in OUT.open()} if OUT.exists() else set()
todo = [p for p in TILES if int(Path(p).stem) not in done]
print(f"{len(TILES)} tiles, {len(done)} already captioned, {len(todo)} to go")

# %%
t0 = time.time()
with OUT.open("a") as f:
    for i in range(0, len(todo), BATCH):
        paths = todo[i:i + BATCH]
        inputs = proc(text=[chat] * len(paths), images=[Image.open(p).convert("RGB") for p in paths],
                      padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=True, temperature=TEMPERATURE)
        texts = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        for p, txt in zip(paths, texts):
            f.write(json.dumps({"cell_id": int(Path(p).stem), "caption": txt.strip(), "model": MODEL,
                                "prompt_id": PROMPT_ID, "seed": SEED}) + "\n")
        f.flush()
        if i == 0:  # eyeball the first batch for broken fp16 output
            for p, txt in list(zip(paths, texts))[:3]:
                print(f"--- {Path(p).stem}: {txt.strip()[:300]}")
        n = i + len(paths)
        if n % 200 < BATCH or n == len(todo):
            rate = (time.time() - t0) / n
            print(f"{n}/{len(todo)}  {rate:.2f} s/image  ~{rate * (len(todo) - n) / 3600:.1f} h left", flush=True)
