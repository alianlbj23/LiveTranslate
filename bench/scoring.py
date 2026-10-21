"""Rankings, composite score, "not recommended" rules and the three recommendations.

composite = 100 * (0.6 q + 0.2 r + 0.2 l)
  q = (Q - 1) / 9, Q = mean judge score (1-10, failed sentences count as 1)
  r = clamp(1 - CER, 0, 1)
  l = clamp((3000 - L) / 2500, 0, 1): L <= 0.5 s scores 1, L >= 3 s scores 0. Fixed
      anchors so runs can be compared.
  L = mean end-to-end latency with ASR and translation loaded together (queue + ASR +
      full translation); a translation that failed there counts as the translation
      timeout plus its ASR time (analysis.together_table). The VAD silence wait
      (~0.5 s) is the same for every combination: noted in the report, not added.
Not recommended: L > 2000 ms, peak VRAM > 11 GB, or missing measurements."""
import math

LATENCY_LIMIT_MS = 2000.0
VRAM_LIMIT_MB = 11 * 1024.0
FAST_MIN_QUALITY = 6.0
FAIL_WARN_RATE = 0.10
WEIGHTS = {"quality": 0.6, "accuracy": 0.2, "latency": 0.2}


def p90(values):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    k = max(0, math.ceil(0.9 * len(vals)) - 1)
    return vals[k]


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def latency_points(latency_ms: float) -> float:
    return clamp01((3000.0 - latency_ms) / 2500.0)


def composite(quality: float, cer: float, latency_ms: float) -> float:
    q = clamp01((quality - 1.0) / 9.0)
    r = clamp01(1.0 - cer)
    return 100.0 * (WEIGHTS["quality"] * q + WEIGHTS["accuracy"] * r + WEIGHTS["latency"] * latency_points(latency_ms))


def rank_asr(stats: dict) -> list:
    """stats: asr id -> {"cer", "mean_ms", "complete"}; incomplete ones are not ranked.
    Lower CER first, then lower mean latency."""
    ranked = [a for a, s in stats.items() if s.get("complete") and s.get("cer") is not None]
    return sorted(ranked, key=lambda a: (stats[a]["cer"], stats[a].get("mean_ms") or 0.0))


def evaluate_combo(combo: dict) -> dict:
    """Fills composite, recommended, reasons, warnings. composite is not None only when
    quality, cer and latency_ms all are (recommend() relies on that)."""
    reasons, warnings = [], []
    labels = {"quality": "翻譯品質", "cer": "辨識準確度", "latency_ms": "同時載入延遲"}
    missing = [k for k in ("quality", "cer", "latency_ms") if combo.get(k) is None]
    if missing:
        notes = combo.get("data_notes") or []
        reasons.append("資料不足：" + ("；".join(notes) if notes else "、".join(labels[k] for k in missing)))
        combo["composite"] = None
    else:
        combo["composite"] = composite(combo["quality"], combo["cer"], combo["latency_ms"])
        if combo["latency_ms"] > LATENCY_LIMIT_MS:
            reasons.append(f"平均總延遲 {combo['latency_ms'] / 1000:.1f} 秒，超過 2 秒")
    if combo.get("vram_peak_mb") is not None and combo["vram_peak_mb"] > VRAM_LIMIT_MB:
        reasons.append(f"顯示記憶體峰值 {combo['vram_peak_mb'] / 1024:.1f} GB，超過 11 GB")
    if (combo.get("fail_rate") or 0) > FAIL_WARN_RATE:
        warnings.append(f"翻譯失敗率 {combo['fail_rate']:.0%}")
    if (combo.get("together_fail_rate") or 0) > FAIL_WARN_RATE:
        warnings.append(f"和辨識同時載入時翻譯失敗率 {combo['together_fail_rate']:.0%}（失敗的句子以逾時計入延遲）")
    if combo.get("vram_measured") is False:
        warnings.append("顯示記憶體沒有量測，11 GB 的檢查沒有做")
    if combo.get("quality_approx"):
        warnings.append("翻譯品質沿用最佳 ASR 的分數（近似值）")
    if combo.get("self_judged"):
        warnings.append("自評，分數可能偏高")
    combo["recommended"] = not reasons
    combo["reasons"] = reasons
    combo["warnings"] = warnings
    return combo


def recommend(combos: list, prefix: str) -> list:
    """Three picks for one category: best quality, lowest latency (among quality >= 6
    when any, else all), most balanced. Chosen among recommended combinations; when there are none, among all of
    them with a warning. One combination winning twice is shown once with both labels."""
    scored = [c for c in combos if c.get("composite") is not None]
    pool = [c for c in scored if c.get("recommended")]
    fallback = not pool
    pool = pool or scored
    if not pool:
        return []
    picks = []

    def add(kind, combo):
        for p in picks:
            if p["combo"] is combo:
                p["kinds"].append(kind)
                return
        picks.append({"kinds": [kind], "combo": combo, "fallback": fallback})

    add("品質最好", max(pool, key=lambda c: (c["quality"], c["composite"])))
    fast_pool = [c for c in pool if c["quality"] >= FAST_MIN_QUALITY] or pool
    add("延遲最低", min(fast_pool, key=lambda c: (c["latency_ms"], -c["composite"])))
    add("最平衡", max(pool, key=lambda c: c["composite"]))
    for i, p in enumerate(picks, 1):
        p["id"] = f"{prefix}{i}"
    return picks
