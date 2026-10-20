"""report.html (self-contained, inline CSS, no JavaScript), results.csv (UTF-8 with BOM
for Excel) and recommendations.json (read by "apply recommendation")."""
import csv
import html
import json
from datetime import datetime

from bench import analysis
from bench import judge as judge_mod
from bench.asr_runner import CANDIDATES_BY_ID
from bench.datasets import DATASETS
from bench.settings_io import setup_steps

NOTES = [
    "Anime-Whisper 的訓練資料和動畫測試集（japanese-anime-speech）類似，它在動畫類的分數可能偏高。",
    "ReazonSpeech 是電視節目語音，只能近似 VTuber 直播（口語、背景音樂、多人聊天都不同）。",
    "翻譯時不帶上下文（測試句彼此無關），實際使用時有上下文，品質可能略好。",
    "沒有模擬漸進式辨識與即時預覽：實際使用時多出的預覽翻譯會占用模型，延遲可能比這裡高。",
    "總延遲 = 排隊 + 辨識 + 整句翻譯的平均（ASR 與翻譯模型同時載入、照真實時間重播）；"
    "語音偵測判斷一句結束要等的靜音（約 0.5 秒）每個組合都一樣，另外列出、不算進去。",
    "CER 以實際會顯示的字幕計算（標點、雜音、語言過濾後）；SenseVoice 會把數字轉成阿拉伯數字，"
    "和用漢字數字的標準答案比對時 CER 會偏高。",
    "顯示記憶體是整張顯示卡的用量，其他程式（瀏覽器、桌布）也會算進去。",
]

CSS = """
:root{--bg:#f7f7f9;--fg:#1d1d24;--muted:#6b6b78;--card:#fff;--line:#e2e2ea;--ok:#1a7f37;--bad:#b42318;--warn:#9a6700;--accent:#3557d4}
@media (prefers-color-scheme: dark){:root{--bg:#141418;--fg:#e8e8ee;--muted:#a0a0ad;--card:#1e1e25;--line:#33333d;--ok:#4cc26a;--bad:#ff7b72;--warn:#e3b341;--accent:#8aa4ff}}
body{margin:0;background:var(--bg);color:var(--fg);font-family:"Microsoft JhengHei UI","Microsoft JhengHei",system-ui,sans-serif;line-height:1.55}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.6rem;margin:0 0 4px}h2{font-size:1.25rem;margin:32px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}
h3{font-size:1.05rem;margin:18px 0 8px}.muted{color:var(--muted)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card h4{margin:0 0 6px;font-size:1.05rem}.tag{display:inline-block;border:1px solid var(--accent);color:var(--accent);border-radius:999px;padding:0 8px;margin-right:6px;font-size:.8rem}
table{border-collapse:collapse;width:100%;background:var(--card);font-size:.9rem;margin:6px 0 14px}
th,td{border:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}th{background:color-mix(in srgb,var(--card) 85%,var(--line))}
td.num{text-align:right;font-variant-numeric:tabular-nums}.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}
.banner{border:1px solid var(--warn);color:var(--warn);border-radius:8px;padding:10px 14px;margin:12px 0;background:var(--card)}
ol{margin:6px 0 0 20px;padding:0}.scroll{overflow-x:auto}
"""


def e(x) -> str:
    return html.escape("" if x is None else str(x))


def ms(x) -> str:
    return "—" if x is None else f"{x:,.0f} ms"


def pct(x) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def num(x, digits=2) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def mb(x) -> str:
    return "—" if x is None else f"{x / 1024:.1f} GB"


def asr_label(asr_id) -> str:
    c = CANDIDATES_BY_ID.get(asr_id)
    return c.label if c else str(asr_id)


def _table(headers, rows, num_cols=()) -> str:
    out = ['<div class="scroll"><table><tr>' + "".join(f"<th>{e(h)}</th>" for h in headers) + "</tr>"]
    for row in rows:
        cells = []
        for i, v in enumerate(row):
            cls = ' class="num"' if i in num_cols else ""
            cells.append(f"<td{cls}>{v}</td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</table></div>")
    return "".join(out)


def build(store) -> dict:
    """Write report.html, results.csv and recommendations.json; returns the
    recommendations."""
    plan = store.plan
    asr_t = analysis.asr_table(store)
    rankings = store.state.get("frozen", {}).get("rankings") or analysis.asr_rankings(store)
    vad_t = analysis.vad_table(store)
    best_th = analysis.best_vad_thresholds(store)
    tl_t = analysis.tl_table(store)
    q_t = analysis.quality_table(store)
    combo_t = analysis.combos(store)
    recs = analysis.recommendations(store, combo_t)
    calib = judge_mod.calibration_report(store.read("judge"))
    names = {m["fp"]: m["name"] for m in plan.get("tl_models", [])}

    parts = [f"<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>最佳組合評測報告</title><style>{CSS}</style></head><body><main>",
             "<h1>最佳組合評測報告</h1>",
             f"<p class='muted'>執行：{e(store.dir.name)}｜狀態：{e(store.state.get('status'))}｜"
             f"產生時間：{datetime.now():%Y-%m-%d %H:%M}｜版本：{e(plan.get('git_rev', '?'))}｜"
             f"評審：{e('、'.join(names.get(j, j) for j in plan.get('judges', [])) or '無')}"
             f"（{'交叉評審' if plan.get('judge_mode') == 'cross' else '單一評審'}）</p>"]

    status = store.state.get("status")
    if status != "done":
        what = {"interrupted": "中斷", "error": "因錯誤停止", "running": "還在進行或被強制關閉"}.get(status, status)
        err = store.state.get("error")
        parts.append(f"<div class='banner'>評測沒有跑完（{e(what)}），以下結果不完整，推薦只能參考。"
                     + (f"<br>錯誤：{e(err)}" if err else "") + "</div>")
    frozen = store.state.get("frozen", {})
    if frozen.get("vram_baseline_mb") is None and store.read("asr_meta"):
        parts.append("<div class='banner'>讀不到顯示卡的記憶體用量（NVML），顯示記憶體欄位空白，11 GB 的檢查沒有做。</div>")
    settle = store.state.get("settle_issues") or []
    if settle:
        parts.append(f"<div class='banner'>有 {len(settle)} 次換模型時顯示記憶體沒有完全回落（其他程式也在用顯示卡？），"
                     "之後的顯示記憶體峰值可能偏高。</div>")

    unreliable = [j for j, s in calib.items() if s["misses"] >= 2]
    if unreliable:
        parts.append("<div class='banner'>評審可能不可靠：" + e("、".join(names.get(j, j) for j in unreliable))
                     + " 在已知答案的校準題中有 2 題以上不符預期，翻譯品質分數僅供參考。</div>")
    if plan.get("quick"):
        parts.append("<div class='banner'>這是快速測試（每類只測少量句子），結果只用來確認流程。</div>")

    # ---------------------------------------------------------- recommendations
    parts.append("<h2>推薦組合</h2>")
    for cat in analysis.CATEGORIES:
        picks = recs.get(cat) or []
        parts.append(f"<h3>{e(analysis.CATEGORY_LABELS[cat])}</h3>")
        if not picks:
            parts.append("<p class='muted'>沒有足夠的資料產生推薦。</p>")
            continue
        if picks[0]["fallback"]:
            parts.append("<div class='banner'>沒有符合「總延遲 ≤ 2 秒、顯示記憶體 ≤ 11 GB」的組合，以下從全部組合中挑選。</div>")
        parts.append("<div class='cards'>")
        for p in picks:
            c = p["combo"]
            steps = setup_steps(c["asr"], c["vad_threshold"], c["tl_name"], p["id"])
            flags = "".join(f"<div class='warn'>⚠ {e(w)}</div>" for w in c["warnings"])
            flags += "".join(f"<div class='bad'>✗ {e(r)}</div>" for r in c["reasons"])
            parts.append(
                f"<div class='card'><h4>{e(p['id'])}　" + "".join(f"<span class='tag'>{e(k)}</span>" for k in p["kinds"])
                + f"</h4><div>辨識：<b>{e(asr_label(c['asr']))}</b>｜VAD 閾值 {num(c['vad_threshold'])}"
                + ("" if c["vad_measured"] else "（未實測，建議預設 0.5）")
                + f"<br>翻譯：<b>{e(c['tl_name'])}</b></div>"
                f"<div class='muted'>綜合 {num(c['composite'], 1)}｜品質 {num(c['quality'])}/10｜CER {pct(c['cer'])}｜"
                f"總延遲 {ms(c['latency_ms'])}｜顯示記憶體峰值 {mb(c['vram_peak_mb'])}</div>{flags}"
                f"<ol>" + "".join(f"<li>{e(s)}</li>" for s in steps) + "</ol></div>")
        parts.append("</div>")

    # ---------------------------------------------------------- stage 1
    parts.append("<h2>階段 1｜語音辨識</h2>")
    for dataset in plan.get("datasets", []):
        rows = []
        ranked = rankings.get(dataset, [])
        for asr, s in sorted(asr_t.get(dataset, {}).items(),
                             key=lambda kv: (kv[1].get("cer") is None, kv[1].get("cer") or 0)):
            rank = f"{ranked.index(asr) + 1}" if asr in ranked else "—"
            status = "<span class='ok'>完成</span>" if s.get("complete") else \
                f"<span class='bad'>不排名：{e(s.get('reason') or '沒有跑完')}</span>"
            if s.get("errors"):
                status += f"<br><span class='warn'>{s['errors']} 句辨識出錯（算全錯）</span>"
            rows.append([rank, e(asr_label(asr)), pct(s.get("cer")), pct(s.get("cer_raw")), ms(s.get("mean_ms")),
                         ms(s.get("p90_ms")), mb(s.get("vram_peak_mb")), num(s.get("load_s"), 1) + " 秒"
                         if s.get("load_s") is not None else "—", str(s.get("n", 0)), status])
        parts.append(f"<h3>{e(DATASETS.get(dataset, {}).get('label', dataset))}</h3>")
        parts.append(_table(["排名", "辨識模型", "CER（字幕）", "CER（原始）", "平均延遲", "P90 延遲",
                             "顯示記憶體峰值", "載入時間", "句數", "狀態"], rows, num_cols=(2, 3, 4, 5, 6, 8)))
    parts.append("<p class='muted'>基準資料集（Common Voice）只用來確認模型正常，不參與排名。動畫、VTuber 各取前 2 名進入下一階段。</p>")

    # ---------------------------------------------------------- stage 2
    parts.append("<h2>階段 2｜語音偵測閾值</h2>")
    if not plan.get("recordings"):
        parts.append("<p>沒有自己的錄音（bench_clips/live/），已略過。建議使用預設閾值 0.5。</p>")
    elif not vad_t:
        parts.append("<p class='bad'>有自己的錄音，但沒有得到評分結果（辨識或評審失敗，原因見下方「略過或失敗的項目」）。"
                     "建議先用預設閾值 0.5。</p>")
    else:
        rows = []
        for asr, by_th in vad_t.items():
            for th, v in sorted(by_th.items()):
                best = " ★" if best_th.get(asr) == th else ""
                judged = f"{v['judged_windows']}/{v['windows']}"
                if v["errors_per_min"] is None:
                    judged = f"<span class='bad'>評審失敗 {judged}</span>"
                rows.append([e(asr_label(asr)), num(th) + best, num(v["hallucinated_per_min"]), num(v["missed_per_min"]),
                             num(v["errors_per_min"]), str(v["segments"]), pct(v["speech_ratio"]), num(v["minutes"], 1),
                             judged])
        parts.append(_table(["辨識模型", "閾值", "幻聽/分鐘", "漏句/分鐘", "合計/分鐘", "片段數", "送辨識比例", "錄音分鐘",
                             "已評/總時段"], rows, num_cols=(2, 3, 4, 5, 6, 7)))

    # ---------------------------------------------------------- stage 3
    parts.append("<h2>階段 3｜翻譯</h2>")
    tl_failed = {m["model"]: m.get("reason") for m in store.read("tl_meta") if m.get("status") == "failed"}
    for cat in analysis.CATEGORIES:
        rows = []
        for m in plan.get("tl_models", []):
            q = q_t.get((m["fp"], cat))
            t = tl_t.get((m["fp"], cat))
            if not q and not t:
                if m["fp"] in tl_failed and cat in plan.get("datasets", []):
                    rows.append([e(m["name"]), f"<span class='bad'>失敗：{e(tl_failed[m['fp']])}</span>"]
                                + ["—"] * 9)
                continue
            q = q or {}
            t = t or {}
            crit = q.get("criteria") or {}
            note = "（自評）" if q.get("self_judged") else ""
            quality = num(q.get("quality")) if q.get("quality") is not None else \
                f"<span class='bad'>{e(q.get('note') or '—')}</span>"
            rows.append([e(m["name"]) + note, quality, *[num(crit.get(c)) for c in judge_mod.CRITERIA],
                         ms(t.get("ttft_mean")), ms(t.get("total_mean")), pct(t.get("fail_rate")),
                         e("、".join(f"{k} {v}" for k, v in (t.get("errors") or {}).items()) or "—"),
                         f"{q.get('judged', 0)}/{q.get('judged', 0) + q.get('unjudged', 0)}"])
        if rows:
            best = (rankings.get(cat) or [None])[0]
            parts.append(f"<h3>{e(analysis.CATEGORY_LABELS[cat])}（翻譯 {e(asr_label(best))} 的辨識結果）</h3>")
            parts.append(_table(["翻譯模型", "品質", *[judge_mod.CRITERIA_LABELS[c] for c in judge_mod.CRITERIA],
                                 "第一個字", "整句完成", "失敗率", "失敗原因", "已評分句數"], rows,
                                num_cols=(1, 2, 3, 4, 5, 6, 7, 8, 10)))

    # ---------------------------------------------------------- combinations
    parts.append("<h2>全部組合</h2>")
    for cat, rows_c in combo_t.items():
        rows = []
        for c in rows_c:
            status = "<span class='ok'>推薦</span>" if c["recommended"] else \
                "<span class='bad'>不推薦</span><br>" + e("；".join(c["reasons"]))
            if c["warnings"]:
                status += "<br><span class='warn'>" + e("；".join(c["warnings"])) + "</span>"
            rows.append([e(asr_label(c["asr"])), num(c["vad_threshold"]), e(c["tl_name"]), num(c["composite"], 1),
                         num(c["quality"]), pct(c["cer"]), ms(c["latency_ms"]), ms(c["latency_p90_ms"]),
                         ms(c["ttft_ms"]), mb(c["vram_peak_mb"]), status])
        parts.append(f"<h3>{e(analysis.CATEGORY_LABELS[cat])}</h3>")
        parts.append(_table(["辨識", "VAD", "翻譯", "綜合", "品質", "CER", "總延遲", "P90", "第一個字", "顯示記憶體", "結論"],
                            rows, num_cols=(1, 3, 4, 5, 6, 7, 8, 9)))
    parts.append("<p class='muted'>綜合分數 = 翻譯品質 60% + 辨識準確 20% + 延遲 20%（延遲 0.5 秒以內滿分、3 秒以上 0 分）。"
                 "總延遲超過 2 秒或顯示記憶體超過 11 GB 標為不推薦。</p>")

    # ---------------------------------------------------------- judges and notes
    if calib:
        rows = [[e(names.get(j, j)), f"{s['items'] - s['misses']}/{s['items']}",
                 "<span class='bad'>可能不可靠</span>" if s["misses"] >= 2 else "<span class='ok'>正常</span>"]
                for j, s in calib.items()]
        parts.append("<h2>評審校準</h2>")
        parts.append(_table(["評審", "校準題符合", "結論"], rows))
        parts.append("<p class='muted'>交叉評審：每個翻譯模型不由自己評分；各評審的尺度用共同評過的題目（校準題、"
                     "兩位評審都評的模型）對齊。只有一個評審時，評它自己的部分標示「自評」。</p>")
    failed = [t for t in store.state.get("tasks", []) if t.get("status") == "failed"]
    if failed:
        parts.append("<h2>略過或失敗的項目</h2>")
        parts.append(_table(["項目", "原因"], [[e(task_label(t["id"], names)), e(t.get("reason") or "—")] for t in failed]))
    parts.append("<h2>注意事項</h2><ul>" + "".join(f"<li>{e(n)}</li>" for n in NOTES) + "</ul>")
    parts.append("</main></body></html>")
    (store.dir / "report.html").write_text("".join(parts), encoding="utf-8")

    with open(store.dir / "results.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "asr", "vad_threshold", "translation_model", "composite", "quality",
                    *judge_mod.CRITERIA, "cer", "latency_mean_ms", "latency_p90_ms", "ttft_ms", "asr_ms",
                    "tl_alone_ms", "vram_peak_mb", "fail_rate", "recommended", "reasons", "warnings"])
        for cat, rows_c in combo_t.items():
            for c in rows_c:
                crit = c.get("criteria") or {}
                w.writerow([cat, c["asr"], c["vad_threshold"], c["tl_name"], _r(c["composite"]), _r(c["quality"]),
                            *[_r(crit.get(k)) for k in judge_mod.CRITERIA], _r(c["cer"], 4), _r(c["latency_ms"], 0),
                            _r(c["latency_p90_ms"], 0), _r(c["ttft_ms"], 0), _r(c["asr_ms"], 0), _r(c["tl_alone_ms"], 0),
                            _r(c["vram_peak_mb"], 0), _r(c["fail_rate"], 3), c["recommended"],
                            "；".join(c["reasons"]), "；".join(c["warnings"])])

    items = []
    for cat, picks in recs.items():
        for p in picks:
            c = p["combo"]
            items.append({"id": p["id"], "kinds": p["kinds"], "category": cat, "asr": c["asr"],
                          "asr_label": asr_label(c["asr"]), "vad_threshold": c["vad_threshold"],
                          "vad_measured": c["vad_measured"], "tl_fp": c["tl"], "tl_name": c["tl_name"],
                          "tl_model": c["tl_model"], "composite": c["composite"], "quality": c["quality"],
                          "latency_ms": c["latency_ms"], "recommended": c["recommended"]})
    (store.dir / "recommendations.json").write_text(json.dumps(
        {"run": store.dir.name, "status": store.state.get("status"), "items": items}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    return recs


def task_label(task_id: str, names: dict) -> str:
    kind, _, rest = task_id.partition("/")
    if kind == "s1":
        return f"階段 1 辨識：{asr_label(rest)}"
    if kind == "s2asr":
        return f"階段 2 語音偵測：{asr_label(rest)}"
    if kind == "s2judge":
        return "階段 2 評審"
    if kind == "s3tl":
        return f"階段 3 翻譯：{names.get(rest, rest)}"
    if kind == "s3tog":
        asr, _, model = rest.partition("/")
        return f"同時載入：{asr_label(asr)} + {names.get(model, model)}"
    if kind == "s3judge":
        return f"評審：{names.get(rest, rest)}"
    return task_id


def _r(x, digits=2):
    return "" if x is None else round(x, digits)
