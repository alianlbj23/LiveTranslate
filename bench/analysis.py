"""Turn the raw result logs of a run into tables, rankings, combinations and
recommendations. Used between stages (top-2 ASR) and by the report.

Record kinds (results/<kind>.jsonl), written by bench.orchestrator:
  asr            one clip through one ASR (raw text, displayed text, ms)
  asr_meta       per ASR: status, load time, VRAM peak
  vad            per (ASR, threshold, recording): segments, judged errors
  tl             one sentence through one translation model (alone)
  tl_meta        per translation model: status, load time
  together       one sentence, ASR + translation loaded together, replayed in real time
  together_meta  per (ASR, model): status, VRAM peak
  judge          one judge score for one item (translation or calibration)"""
from statistics import mean

from bench import judge as judge_mod
from bench import scoring
from bench.textnorm import clip_errors

CATEGORIES = ("anime", "vtuber")
CATEGORY_LABELS = {"anime": "動畫", "vtuber": "VTuber"}


def _latest(records, key_fields):
    """Last record per key (a resumed task may have rewritten an item)."""
    out = {}
    for r in records:
        out[tuple(r.get(k) for k in key_fields)] = r
    return list(out.values())


# ---------------------------------------------------------------- stage 1

def asr_table(store) -> dict:
    """{dataset: {asr: stats}}; stats["complete"] False when the ASR failed / did not
    finish the dataset (it is then not ranked)."""
    records = _latest(store.read("asr"), ("asr", "dataset", "clip"))
    meta = {(m["asr"]): m for m in store.read("asr_meta")}
    expected = store.plan.get("clip_counts", {})
    table = {}
    for r in records:
        table.setdefault(r["dataset"], {}).setdefault(r["asr"], []).append(r)
    out = {}
    for dataset, by_asr in table.items():
        for asr, recs in by_asr.items():
            dist = length = dist_raw = 0
            for r in recs:
                d, n = clip_errors(r["ref"], r.get("display") or "")
                dist += d
                length += n
                dist_raw += clip_errors(r["ref"], r.get("raw") or "")[0]
            ms = [r["ms"] for r in recs if r.get("ms") is not None]
            m = meta.get(asr, {})
            out.setdefault(dataset, {})[asr] = {
                "n": len(recs),
                "cer": dist / length if length else None,
                "cer_raw": dist_raw / length if length else None,
                "mean_ms": mean(ms) if ms else None,
                "p90_ms": scoring.p90(ms),
                "dropped": sum(1 for r in recs if r.get("reason") and not str(r["reason"]).startswith("language-partial")),
                "complete": m.get("status") == "ok" and len(recs) >= expected.get(dataset, len(recs)),
                "vram_peak_mb": m.get("vram_peak_mb"),
                "vram_base_mb": m.get("vram_base_mb"),
                "load_s": m.get("load_s"),
            }
    for m in store.read("asr_meta"):
        if m.get("status") != "ok":
            for dataset in store.plan.get("datasets", []):
                out.setdefault(dataset, {}).setdefault(m["asr"], {"n": 0, "cer": None, "complete": False,
                                                                  "status": m.get("status"), "reason": m.get("reason")})
    return out


def asr_rankings(store) -> dict:
    """{category: [asr ids best first]} for the categories in this run."""
    table = asr_table(store)
    return {cat: scoring.rank_asr(table.get(cat, {})) for cat in CATEGORIES if cat in table}


# ---------------------------------------------------------------- stage 2

def vad_table(store) -> dict:
    """{asr: {threshold: {"errors_per_min", "hallucinated_per_min", "missed_per_min",
    "segments", "speech_ratio", "minutes"}}} summed over recordings."""
    acc = {}
    for r in _latest(store.read("vad"), ("asr", "threshold", "recording")):
        a = acc.setdefault(r["asr"], {}).setdefault(float(r["threshold"]), {
            "minutes": 0.0, "hallucinated": 0, "missed": 0, "segments": 0, "speech_s": 0.0, "judged": 0})
        a["minutes"] += r.get("minutes", 0.0)
        a["hallucinated"] += r.get("hallucinated", 0)
        a["missed"] += r.get("missed", 0)
        a["segments"] += r.get("segments", 0)
        a["speech_s"] += r.get("speech_s", 0.0)
        a["judged"] += r.get("judged_windows", 0)
    out = {}
    for asr, by_th in acc.items():
        for th, a in by_th.items():
            minutes = a["minutes"] or 1e-9
            out.setdefault(asr, {})[th] = {
                "hallucinated_per_min": a["hallucinated"] / minutes,
                "missed_per_min": a["missed"] / minutes,
                "errors_per_min": (a["hallucinated"] + a["missed"]) / minutes,
                "segments": a["segments"], "speech_ratio": a["speech_s"] / (minutes * 60),
                "minutes": a["minutes"], "judged_windows": a["judged"]}
    return out


def best_vad_thresholds(store) -> dict:
    from bench.vad_runner import best_threshold

    return {asr: best_threshold({th: v["errors_per_min"] for th, v in by_th.items() if v["judged_windows"]})
            for asr, by_th in vad_table(store).items()}


# ---------------------------------------------------------------- stage 3

def tl_table(store) -> dict:
    """{(model fp, category): {"n", "fail_rate", "ttft_mean", "total_mean", "errors"}}"""
    out = {}
    for r in _latest(store.read("tl"), ("model", "category", "clip")):
        out.setdefault((r["model"], r["category"]), []).append(r)
    table = {}
    for key, recs in out.items():
        tried = [r for r in recs if r.get("source")]
        ok = [r for r in tried if r["ok"]]
        errors = {}
        for r in tried:
            if not r["ok"]:
                errors[r.get("error") or "?"] = errors.get(r.get("error") or "?", 0) + 1
        table[key] = {
            "n": len(recs), "tried": len(tried),
            "fail_rate": (len(tried) - len(ok)) / len(tried) if tried else None,
            "ttft_mean": mean(r["ttft_ms"] for r in ok) if ok else None,
            "total_mean": mean(r["total_ms"] for r in ok) if ok else None,
            "errors": errors}
    return table


def quality_table(store) -> dict:
    """{(model fp, category): {"quality", "criteria", "n", "judge_fail", "self_judged",
    "judges"}}. A sentence the ASR dropped or the model failed counts as 1 on every
    criterion; judge scores are aligned with per-judge offsets (cross judging)."""
    judge_recs = _latest(store.read("judge"), ("judge", "item"))
    offsets = judge_mod.judge_offsets(judge_recs)
    by_item = {}
    for r in judge_recs:
        by_item.setdefault(r["item"], []).append(r)
    table = {}
    for r in _latest(store.read("tl"), ("model", "category", "clip")):
        key = (r["model"], r["category"])
        t = table.setdefault(key, {"items": [], "criteria": {c: [] for c in judge_mod.CRITERIA},
                                   "judge_fail": 0, "self_judged": False, "judges": set()})
        if not r.get("source") or not r.get("ok"):
            t["items"].append(1.0)
            for c in judge_mod.CRITERIA:
                t["criteria"][c].append(1.0)
            continue
        item = f"{r['model']}/{r['category']}/{r['clip']}"
        scored = [j for j in by_item.get(item, []) if j.get("scores")]
        if not scored:
            t["judge_fail"] += 1 if by_item.get(item) else 0
            continue
        t["self_judged"] |= any(j.get("self") for j in scored)
        t["judges"].update(j["judge"] for j in scored)
        t["items"].append(mean(judge_mod.item_mean(j["scores"]) - offsets.get(j["judge"], 0.0) for j in scored))
        for c in judge_mod.CRITERIA:
            t["criteria"][c].append(mean(j["scores"][c] - offsets.get(j["judge"], 0.0) for j in scored))
    out = {}
    for key, t in table.items():
        out[key] = {"quality": mean(t["items"]) if t["items"] else None,
                    "criteria": {c: (mean(v) if v else None) for c, v in t["criteria"].items()},
                    "n": len(t["items"]), "judge_fail": t["judge_fail"], "self_judged": t["self_judged"],
                    "judges": sorted(t["judges"])}
    return out


def together_table(store) -> dict:
    """{(asr, model fp, category): {"e2e_mean", "e2e_p90", "asr_mean", "queue_mean",
    "ttft_mean", "n", "fail_rate", "vram_peak_mb", "status"}}"""
    meta = {(m["asr"], m["model"]): m for m in store.read("together_meta")}
    groups = {}
    for r in _latest(store.read("together"), ("asr", "model", "category", "clip")):
        groups.setdefault((r["asr"], r["model"], r["category"]), []).append(r)
    out = {}
    for key, recs in groups.items():
        ok = [r for r in recs if r.get("ok")]
        m = meta.get((key[0], key[1]), {})
        out[key] = {
            "n": len(recs), "fail_rate": (len(recs) - len(ok)) / len(recs) if recs else None,
            "e2e_mean": mean(r["e2e_ms"] for r in ok) if ok else None,
            "e2e_p90": scoring.p90([r["e2e_ms"] for r in ok]),
            "asr_mean": mean(r["asr_ms"] for r in ok) if ok else None,
            "queue_mean": mean(r["queue_ms"] for r in ok) if ok else None,
            "ttft_mean": mean(r["ttft_ms"] for r in ok if r.get("ttft_ms") is not None) if ok else None,
            "vram_peak_mb": m.get("vram_peak_mb"), "status": m.get("status")}
    for (asr, model), m in meta.items():
        if m.get("status") != "ok":
            for cat in CATEGORIES:
                out.setdefault((asr, model, cat), {"n": 0, "e2e_mean": None, "vram_peak_mb": m.get("vram_peak_mb"),
                                                   "status": m.get("status"), "reason": m.get("reason")})
    return out


# ---------------------------------------------------------------- combinations

def combos(store) -> dict:
    """{category: [combo]} with composite score and flags, best first."""
    frozen = store.state.get("frozen", {})
    rankings = frozen.get("rankings") or asr_rankings(store)
    thresholds = best_vad_thresholds(store)
    asr_stats = asr_table(store)
    quality = quality_table(store)
    tl = tl_table(store)
    together = together_table(store)
    models = store.plan.get("tl_models", [])
    out = {}
    for cat, ranked in rankings.items():
        if not ranked:
            continue
        best = ranked[0]
        rows = []
        for asr in ranked[:2]:
            for m in models:
                fp = m["fp"]
                q = quality.get((fp, cat), {})
                t = tl.get((fp, cat), {})
                tog = together.get((asr, fp, cat), {})
                combo = {
                    "category": cat, "asr": asr, "tl": fp, "tl_name": m["name"], "tl_model": m["model"],
                    "vad_threshold": thresholds.get(asr, 0.5), "vad_measured": asr in thresholds,
                    "quality": q.get("quality"), "criteria": q.get("criteria", {}),
                    "self_judged": q.get("self_judged", False),
                    "cer": (asr_stats.get(cat, {}).get(asr) or {}).get("cer"),
                    "latency_ms": tog.get("e2e_mean"), "latency_p90_ms": tog.get("e2e_p90"),
                    "asr_ms": tog.get("asr_mean"), "ttft_ms": tog.get("ttft_mean") or t.get("ttft_mean"),
                    "tl_alone_ms": t.get("total_mean"),
                    "vram_peak_mb": tog.get("vram_peak_mb"),
                    "fail_rate": t.get("fail_rate"),
                    "quality_approx": asr != best,
                }
                rows.append(scoring.judge_combo(combo))
        rows.sort(key=lambda c: (c["composite"] is None, -(c["composite"] or 0)))
        out[cat] = rows
    return out


def recommendations(store, combo_table=None) -> dict:
    combo_table = combo_table if combo_table is not None else combos(store)
    prefix = {"anime": "A", "vtuber": "V"}
    return {cat: scoring.recommend(rows, prefix[cat]) for cat, rows in combo_table.items()}
