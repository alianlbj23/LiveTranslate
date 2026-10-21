"""Turn the raw result logs of a run into tables, rankings, combinations and
recommendations. Used between stages (top-2 ASR) and by the report.

Record kinds (results/<kind>.jsonl), written by bench.orchestrator; every record has
"task" and "key" (the item inside the task):
  asr            one clip through one ASR: asr, dataset, clip, ref, duration, raw, label,
                 display (what the app would show), reason (why a filter dropped text),
                 ms; "error" when the ASR failed on it (counts as all wrong), "skip" when
                 the clip itself could not be read (not counted)
  asr_meta       stage 1 only, per ASR: status ok / failed / incomplete, reason, load_s,
                 errors, vram_peak_mb, vram_base_mb
  vad_raw        stage 2, per (ASR, recording, threshold): lines, segments, speech_s,
                 minutes; threshold None = the continuous reference (windows)
  vad            the judged vad_raw: hallucinated, missed, windows, judged_windows,
                 judged_minutes
  tl             one sentence through one translation model alone: model, category,
                 clip, ref, source (the best ASR's text), ok, text (the translation),
                 ttft_ms, total_ms, error, detail
  tl_meta        per translation model: status, reason, load_s, errors, vram_peak_mb
  together       one sentence with ASR + translation loaded together, replayed in real
                 time: asr, model, category, clip, source (this ASR's text), translation,
                 ok, error, detail, queue_ms, asr_ms, ttft_ms, tl_ms, e2e_ms
  together_meta  per (ASR, model): status, reason, errors, vram_peak_mb
  judge          one judge score for one item (translation or calibration): judge,
                 item, candidate, category, scores (None = the judge failed), self
The latest record per item wins (a resume may redo failed items)."""
from statistics import mean

from bench import asr_runner, scoring
from bench import judge as judge_mod
from bench.textnorm import clip_errors

CATEGORIES = ("anime", "vtuber")
CATEGORY_LABELS = {"anime": "動畫", "vtuber": "VTuber"}
MIN_JUDGE_COVERAGE = 0.8
DEFAULT_TL_TIMEOUT_S = 10.0


def _on_scale(score: float) -> float:
    """Subtracting a judge offset can push a score past 1 or 10; clamp so Q stays on the
    1-10 scale that the composite's q = (Q - 1) / 9 assumes."""
    return max(1.0, min(10.0, score))


def _latest(records, key_fields):
    out = {}
    for r in records:
        out[tuple(r.get(k) for k in key_fields)] = r
    return list(out.values())


# ---------------------------------------------------------------- stage 1

def asr_table(store) -> dict:
    """{dataset: {asr: stats}}; stats["complete"] False (with "reason") when the ASR
    failed or did not finish the dataset (it is then not ranked)."""
    raw = store.read("asr")
    records = [r for r in _latest(raw, ("asr", "dataset", "clip")) if not r.get("skip")]
    meta = {}
    for m in store.read("asr_meta"):
        if str(m.get("task", "")).startswith("s1/"):
            meta[m["asr"]] = m
    expected = store.plan["clip_counts"]
    skipped = {(r["dataset"], r["clip"]) for r in raw if r.get("skip")}
    table = {}
    for r in records:
        table.setdefault(r["dataset"], {}).setdefault(r["asr"], []).append(r)
    out = {}
    for dataset, by_asr in table.items():
        want = expected.get(dataset, 0) - len({c for d, c in skipped if d == dataset})
        for asr, recs in by_asr.items():
            dist = length = dist_raw = 0
            for r in recs:
                d, n = clip_errors(r["ref"], r.get("display") or "")
                dist += d
                length += n
                dist_raw += clip_errors(r["ref"], r.get("raw") or "")[0]
            ms = [r["ms"] for r in recs if r.get("ms") is not None]
            m = meta.get(asr, {})
            reason = None
            if m.get("status") != "ok":
                reason = m.get("reason") or "沒有跑完"
            elif len(recs) < want:
                reason = f"只有 {len(recs)}/{want} 句"
            out.setdefault(dataset, {})[asr] = {
                "n": len(recs),
                "errors": sum(1 for r in recs if r.get("error")),
                "cer": dist / length if length else None,
                "cer_raw": dist_raw / length if length else None,
                "mean_ms": mean(ms) if ms else None,
                "p90_ms": scoring.p90(ms),
                "dropped": sum(1 for r in recs if r.get("reason") and not str(r["reason"]).startswith("language-partial")),
                "complete": reason is None,
                "reason": reason,
                "status": m.get("status"),
                "vram_peak_mb": m.get("vram_peak_mb"),
                "vram_base_mb": m.get("vram_base_mb"),
                "load_s": m.get("load_s"),
            }
    for asr, m in meta.items():
        if m.get("status") != "ok":
            for dataset in store.plan["datasets"]:
                out.setdefault(dataset, {}).setdefault(asr, {
                    "n": 0, "cer": None, "complete": False, "status": m.get("status"),
                    "reason": m.get("reason") or "沒有跑完"})
    return out


def asr_rankings(store) -> dict:
    """{category: [asr ids best first]} for the categories in this run."""
    table = asr_table(store)
    return {cat: scoring.rank_asr(table.get(cat, {})) for cat in CATEGORIES if cat in store.plan["datasets"]}


def frozen_rankings(store) -> dict:
    """The rankings later stages used (frozen once after stage 1), or computed now for a
    run that never got that far."""
    rankings = store.frozen("rankings")
    return rankings if rankings is not None else asr_rankings(store)


# ---------------------------------------------------------------- stage 2

def vad_table(store) -> dict:
    """{asr: {threshold: {...}}} summed over recordings. Error rates use only the
    minutes the judge actually scored; None when no window was judged."""
    acc = {}
    for r in _latest(store.read("vad"), ("asr", "threshold", "recording")):
        a = acc.setdefault(r["asr"], {}).setdefault(float(r["threshold"]), {
            "minutes": 0.0, "judged_minutes": 0.0, "hallucinated": 0, "missed": 0, "segments": 0, "speech_s": 0.0,
            "judged": 0, "windows": 0})
        a["minutes"] += r.get("minutes", 0.0)
        a["judged_minutes"] += r.get("judged_minutes", 0.0)
        a["hallucinated"] += r.get("hallucinated", 0)
        a["missed"] += r.get("missed", 0)
        a["segments"] += r.get("segments", 0)
        a["speech_s"] += r.get("speech_s", 0.0)
        a["judged"] += r.get("judged_windows", 0)
        a["windows"] += r.get("windows", 0)
    out = {}
    for asr, by_th in acc.items():
        for th, a in by_th.items():
            jm = a["judged_minutes"]
            ok = a["judged"] > 0 and jm > 0
            out.setdefault(asr, {})[th] = {
                "hallucinated_per_min": a["hallucinated"] / jm if ok else None,
                "missed_per_min": a["missed"] / jm if ok else None,
                "errors_per_min": (a["hallucinated"] + a["missed"]) / jm if ok else None,
                "segments": a["segments"],
                "speech_ratio": a["speech_s"] / (a["minutes"] * 60) if a["minutes"] else None,
                "minutes": a["minutes"], "judged_windows": a["judged"], "windows": a["windows"]}
    return out


def best_vad_thresholds(store) -> dict:
    """{asr: threshold} only for ASRs whose windows were actually judged."""
    from bench.vad_runner import best_threshold

    out = {}
    for asr, by_th in vad_table(store).items():
        judged = {th: v["errors_per_min"] for th, v in by_th.items() if v["errors_per_min"] is not None}
        if judged:
            out[asr] = best_threshold(judged)
    return out


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
                kind = r.get("error") or "?"
                errors[kind] = errors.get(kind, 0) + 1
        table[key] = {
            "n": len(recs), "tried": len(tried),
            "fail_rate": (len(tried) - len(ok)) / len(tried) if tried else None,
            "ttft_mean": mean(r["ttft_ms"] for r in ok) if ok else None,
            "total_mean": mean(r["total_ms"] for r in ok) if ok else None,
            "errors": errors}
    return table


def judge_scale(store) -> dict:
    """{judge fp: {"offset", "clamped", "scored"}}: how far each judge's scale sits from
    the others (subtracted before averaging) and how many of its aligned item scores fell
    outside 1-10 and were clamped. A large offset means the judges disagree on the scale,
    and clamping squeezes the ends of it."""
    recs = _latest(store.read("judge"), ("judge", "item"))
    offsets = judge_mod.judge_offsets(recs)
    out = {j: {"offset": off, "clamped": 0, "scored": 0} for j, off in offsets.items()}
    for r in recs:
        if r.get("scores"):
            s = out[r["judge"]]
            s["scored"] += 1
            if not 1.0 <= judge_mod.item_mean(r["scores"]) - s["offset"] <= 10.0:
                s["clamped"] += 1
    return out


def quality_table(store) -> dict:
    """{(model fp, category): {"quality", "criteria", "n", "judged", "unjudged",
    "judge_fail", "self_judged", "judges", "note"}}.

    A sentence the ASR dropped or the model failed counts as 1 on every criterion.
    Judge scores are aligned with per-judge offsets (cross judging). When fewer than
    80 % of the sentences that should have been judged have a score (a judge failed or
    has not run), quality is None with a note instead of a misleading number."""
    judge_recs = _latest(store.read("judge"), ("judge", "item"))
    offsets = judge_mod.judge_offsets(judge_recs)
    by_item = {}
    for r in judge_recs:
        by_item.setdefault(r["item"], []).append(r)
    table = {}
    for r in _latest(store.read("tl"), ("model", "category", "clip")):
        key = (r["model"], r["category"])
        t = table.setdefault(key, {"items": [], "criteria": {c: [] for c in judge_mod.CRITERIA},
                                   "judged": 0, "to_judge": 0, "judge_fail": 0, "self_judged": False,
                                   "judges": set(), "failed_items": 0})
        if not r.get("source") or not r.get("ok"):
            t["failed_items"] += 1
            t["items"].append(1.0)
            for c in judge_mod.CRITERIA:
                t["criteria"][c].append(1.0)
            continue
        t["to_judge"] += 1
        records = by_item.get(judge_mod.item_key(r["model"], r["category"], r["clip"]), [])
        scored = [j for j in records if j.get("scores")]
        if not scored:
            if records:
                t["judge_fail"] += 1
            continue
        t["judged"] += 1
        t["self_judged"] |= any(j.get("self") for j in scored)
        t["judges"].update(j["judge"] for j in scored)
        t["items"].append(_on_scale(mean(judge_mod.item_mean(j["scores"]) - offsets.get(j["judge"], 0.0) for j in scored)))
        for c in judge_mod.CRITERIA:
            t["criteria"][c].append(_on_scale(mean(j["scores"][c] - offsets.get(j["judge"], 0.0) for j in scored)))
    out = {}
    for key, t in table.items():
        coverage = t["judged"] / t["to_judge"] if t["to_judge"] else 1.0
        enough = coverage >= MIN_JUDGE_COVERAGE and t["items"]
        note = None if enough else f"評審覆蓋不足 {t['judged']}/{t['to_judge']} 句"
        out[key] = {"quality": mean(t["items"]) if enough else None,
                    "criteria": {c: (mean(v) if v and enough else None) for c, v in t["criteria"].items()},
                    "n": len(t["items"]), "judged": t["judged"], "unjudged": t["to_judge"] - t["judged"],
                    "judge_fail": t["judge_fail"], "failed_items": t["failed_items"],
                    "self_judged": t["self_judged"], "judges": sorted(t["judges"]), "note": note}
    return out


def together_table(store) -> dict:
    """{(asr, model fp, category): {...}}. A translation that failed while loaded
    together counts with the translation timeout plus its ASR time as its latency (it
    would have shown nothing for that long); empty ASR output is not a translation
    failure. Failures keep their reason (error, detail) in the together records."""
    timeout_ms = 1000.0 * float(store.plan["tl_timeout_s"] or DEFAULT_TL_TIMEOUT_S)
    meta = {(m["asr"], m["model"]): m for m in store.read("together_meta")}
    groups = {}
    for r in _latest(store.read("together"), ("asr", "model", "category", "clip")):
        groups.setdefault((r["asr"], r["model"], r["category"]), []).append(r)
    out = {}
    for key, recs in groups.items():
        tried = [r for r in recs if r.get("error") != asr_runner.ASR_EMPTY]
        ok = [r for r in tried if r.get("ok")]
        e2e = [r["e2e_ms"] if r.get("ok") else timeout_ms + (r.get("asr_ms") or 0) for r in tried]
        m = meta.get((key[0], key[1]), {})
        out[key] = {
            "n": len(recs), "tried": len(tried),
            "fail_rate": (len(tried) - len(ok)) / len(tried) if tried else None,
            "e2e_mean": mean(e2e) if e2e else None,
            "e2e_p90": scoring.p90(e2e),
            "asr_mean": mean(r["asr_ms"] for r in recs) if recs else None,
            "queue_mean": mean(r["queue_ms"] for r in recs) if recs else None,
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
    rankings = frozen_rankings(store)
    thresholds = best_vad_thresholds(store)
    asr_stats = asr_table(store)
    quality = quality_table(store)
    tl = tl_table(store)
    together = together_table(store)
    models = store.plan["tl_models"]
    vram_measured = store.frozen("vram_baseline_mb") is not None
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
                data_notes = []
                if q.get("note"):
                    data_notes.append(q["note"])
                if tog.get("reason"):
                    data_notes.append(f"同時載入失敗：{tog['reason']}")
                combo = {
                    "category": cat, "asr": asr, "tl": fp, "tl_name": m["name"], "tl_model": m["model"],
                    "vad_threshold": thresholds.get(asr, 0.5), "vad_measured": asr in thresholds,
                    "quality": q.get("quality"), "criteria": q.get("criteria", {}),
                    "self_judged": q.get("self_judged", False),
                    "cer": (asr_stats.get(cat, {}).get(asr) or {}).get("cer"),
                    "latency_ms": tog.get("e2e_mean"), "latency_p90_ms": tog.get("e2e_p90"),
                    "asr_ms": tog.get("asr_mean"), "ttft_ms": tog.get("ttft_mean") or t.get("ttft_mean"),
                    "tl_alone_ms": t.get("total_mean"),
                    "vram_peak_mb": tog.get("vram_peak_mb"), "vram_measured": vram_measured,
                    "fail_rate": t.get("fail_rate"), "together_fail_rate": tog.get("fail_rate"),
                    "quality_approx": asr != best, "data_notes": data_notes,
                }
                rows.append(scoring.judge_combo(combo))
        rows.sort(key=lambda c: (c["composite"] is None, -(c["composite"] or 0)))
        out[cat] = rows
    return out


def recommendations(store, combo_table=None) -> dict:
    combo_table = combo_table if combo_table is not None else combos(store)
    prefix = {"anime": "A", "vtuber": "V"}
    return {cat: scoring.recommend(rows, prefix[cat]) for cat, rows in combo_table.items()}
