"""Caption audit (PLAN 4.10): blind sentence labels -> raw vs cleaned false-claim shares and cleaning accuracy."""
import importlib.util

import pandas as pd
import pytest

from gisecon.config import REPO_ROOT, load_config

spec = importlib.util.spec_from_file_location("build_texts", REPO_ROOT / "scripts" / "12_build_texts.py")
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)


def test_audit_export_and_score(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "processed").mkdir(parents=True)
    (data / "interim").mkdir()
    (tmp_path / "reports" / "tables").mkdir(parents=True)
    (data / "raw" / "tiles").mkdir(parents=True)
    for c in range(12):  # the export copies each tile into the self-contained audit folder
        (data / "raw" / "tiles" / f"{c}.png").write_bytes(b"png")
    monkeypatch.setenv("GISECON_DATA_DIR", str(data))
    monkeypatch.setattr(bt, "REPO_ROOT", tmp_path)
    cfg = load_config()
    cfg["text"]["audit_n"] = 12

    # 12 captions x 3 sentences: s0 true and kept, s1 false and rule-dropped, s2 false (kept) in even captions.
    # Caption 11 was dropped as a whole by the CLIP filter.
    sents, labels = [], []
    for c in range(12):
        sents += [{"cell_id": c, "sentence": f"c{c} s0.", "drop_reason": None},
                  {"cell_id": c, "sentence": f"c{c} s1.", "drop_reason": "claim_water"},
                  {"cell_id": c, "sentence": f"c{c} s2.", "drop_reason": None}]
        labels += ["correct", "false", "false" if c % 2 == 0 else "correct"]
    pd.DataFrame(sents).to_parquet(data / "interim" / "caption_sentences.parquet")
    pd.DataFrame({"cell_id": range(12), "text_raw": "x", "keep_caption": [True] * 11 + [False]}).to_parquet(
        data / "processed" / "texts.parquet")

    bt.audit_export(cfg)
    sheet = pd.read_csv(data / "interim" / "audit" / "audit_labels.csv")
    assert len(sheet) == 36 and "drop_reason" not in sheet  # labeller cannot see the cleaning decisions
    assert len(list((data / "interim" / "audit" / "tiles").glob("*.png"))) == 12
    truth = pd.DataFrame(sents).assign(sent_idx=lambda d: d.groupby("cell_id").cumcount(), label=labels)
    sheet = sheet.drop(columns="label").merge(truth[["cell_id", "sent_idx", "label"]], on=["cell_id", "sent_idx"])
    sheet.to_csv(data / "interim" / "audit" / "audit_labels.csv", index=False)

    bt.audit_score(cfg)
    out = pd.read_csv(tmp_path / "reports" / "tables" / "caption_audit.csv").set_index("measure")["value"]
    v = {k.split(":")[-1].strip() if ":" in k else k.split(" (")[0]: x for k, x in out.items()}
    assert v["raw captions"] == pytest.approx(1.0, abs=1e-3)
    assert v["cleaned captions (those still used)"] == pytest.approx(6 / 11, abs=1e-3)   # caption 11 no longer used
    assert v["rule-dropped sentences that were false or vague"] == pytest.approx(1.0, abs=1e-3)
    assert v["false sentences removed by any cleaning pass"] == pytest.approx(12 / 18, abs=1e-3)
