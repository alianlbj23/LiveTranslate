"""Benchmark menu actions, called by launcher/bench.ps1 through launcher/bench_run.py.
The user never types these commands: each menu item runs one of them.

  status    one-line-per-item summary for the menu header
  download  1) download the test data
  start     2) choose stages / datasets / models, show the estimate, run
  resume    3) continue the last unfinished run
  report    4) open report.html of a run
  apply     5) apply a recommended combination to LiveTranslate (backup first)"""
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from bench import analysis, costs, datasets, judge as judge_mod, llm_servers, settings_io, state
from bench.asr_runner import ASR_CANDIDATES, CANDIDATES_BY_ID, DOWNLOAD_BYTES, is_cached
from bench.orchestrator import Bench, Deps, describe_estimate, estimate_remaining, git_rev
from bench.progress import fmt_duration
from bench.wizard import Console

log = logging.getLogger("LiveTranslate.Bench")


def _setup_logging(run_dir: Path | None = None):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) and getattr(h, "_bench", False) for h in root.handlers):
        h = logging.StreamHandler(sys.stderr)
        h.setLevel(logging.WARNING)
        h.setFormatter(logging.Formatter("\n[%(levelname)s] %(message)s"))
        h._bench = True
        root.addHandler(h)
    if run_dir is not None:
        fh = logging.FileHandler(run_dir / "bench.log", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        root.addHandler(fh)


def _wait_app_closed(con: Console) -> bool:
    while settings_io.find_running_app():
        con.say("", "LiveTranslate 正在執行。評測和套用設定都需要先關閉它（系統匣圖示 → 結束）。")
        if con.ask("關閉後按 Enter 重新檢查，輸入 0 取消：") == "0":
            return False
    return True


# ---------------------------------------------------------------- status

def cmd_status(con: Console) -> int:
    parts = []
    for key in ("anime", "vtuber", "baseline"):
        have, want = datasets.dataset_status(key)
        label = {"anime": "動畫", "vtuber": "VTuber", "baseline": "基準"}[key]
        parts.append(f"{label} {have}/{want}" + (" ✓" if have >= want else ""))
    live = datasets.live_recordings()
    con.say(" 測試資料：" + "　".join(parts) + f"｜自己的錄音：{len(live)} 檔")
    run = state.RunStore.latest_unfinished()
    if run:
        tasks = run.state.get("tasks", [])
        finished = sum(1 for t in tasks if t["status"] in ("done", "failed"))
        con.say(f" 未完成的評測：{run.dir.name}（已完成 {finished} 個項目，"
                f"剩餘約 {describe_estimate(run)}）")
    return 0


# ---------------------------------------------------------------- download

def cmd_download(con: Console, client=None) -> int:
    con.say("", datasets.LICENSE_NOTE, "")
    total = sum(spec["n"] + datasets.WARMUP_CLIPS for spec in datasets.DATASETS.values())
    con.say(f"會從 Hugging Face 下載 {total} 句（約 80 MB），每次抽到的句子都一樣，大約需要 5–15 分鐘。")
    if not con.confirm("開始下載？", True):
        return 0
    try:
        settings = settings_io.load_settings()
    except settings_io.SettingsError as e:
        log.warning(f"{e}; using the system proxy setting")
        settings = {}
    client = client or datasets.RowsClient(datasets.make_http_client(settings.get("download_proxy", "system")))
    ok = True
    for key, spec in datasets.DATASETS.items():
        con.say(f"\n{spec['label']}")

        def progress(k, n, target, label=spec["label"]):
            sys.stdout.write(f"\r  {n}/{target} 句")
            sys.stdout.flush()

        try:
            m = datasets.download_dataset(key, client, progress=progress)
            con.say(f"\r  完成：{len(m['clips'])}/{spec['n']} 句（略過 {len(m['skipped'])} 筆不合格的資料）")
        except datasets.DownloadError as e:
            ok = False
            con.say("", f"  網路錯誤：{e}", "  已下載的部分會保留，稍後再選 1 會從中斷的地方繼續。")
            break
    return 0 if ok else 1


# ---------------------------------------------------------------- start

def _model_options(settings):
    """[(entry, usable, note)] for the configured translation models."""
    installed = None
    lm_alive = None
    models = settings_io.translation_models(settings)
    for entry in models:
        kind = llm_servers.model_kind(entry)
        if kind == "ollama" and installed is None:
            import local_models

            server = llm_servers.Ollama(local_models.ollama_root(entry["api_base"]))
            installed = server.installed() if server.alive() else _installed_from_manifests(models)
        if kind == "lmstudio" and lm_alive is None:
            lm_alive = llm_servers._port_open(1234)
    out = []
    from bench.tl_runner import availability

    for entry in models:
        usable, note = availability(entry, installed, lm_alive)
        out.append((entry, usable, note))
    return out


def _installed_from_manifests(models) -> set:
    """Ollama not running: read its manifests the way the launcher does."""
    root = Path(os.environ.get("OLLAMA_MODELS") or Path.home() / ".ollama" / "models") / "manifests"
    return {e.get("model", "") for e in models if manifest_path(root, e.get("model", "")).is_file()}


def manifest_path(manifests: Path, model: str) -> Path:
    """Same rule as launcher/gui.ps1 Test-OllamaModelDownloaded:
    qwen3.5:9b -> registry.ollama.ai/library/qwen3.5/9b, hf.co/Org/Repo:TAG -> hf.co/Org/Repo/TAG."""
    name, tag = model, "latest"
    i = model.rfind(":")
    if i > 0 and "/" not in model[i + 1:]:
        name, tag = model[:i], model[i + 1:]
    parts = name.split("/")
    if len(parts) >= 2 and "." in parts[0]:
        rel = parts
    elif len(parts) == 1:
        rel = ["registry.ollama.ai", "library", parts[0]]
    else:
        rel = ["registry.ollama.ai", *parts]
    return manifests.joinpath(*rel, tag)


def build_plan(con: Console, settings: dict, quick: bool = False):
    """Ask the questions; returns a plan dict or None when cancelled."""
    recordings = datasets.live_recordings()
    stage_opts = [
        (1, "階段 1｜語音辨識：4 個 ASR 在測試資料上比較準確度與速度", True, ""),
        (2, "階段 2｜語音偵測閾值（用自己的錄音）", bool(recordings),
         "" if recordings else "沒有 bench_clips/live/ 錄音：自動略過，報告建議 0.5"),
        (3, "階段 3｜翻譯：品質、速度、與 ASR 同時載入的延遲", True, ""),
    ]
    stages = con.choose_many("要跑哪些階段？", stage_opts, {1, 2, 3})
    if stages is None:
        return None
    reuse = None
    if 1 not in stages:
        previous = [r for r in state.RunStore.list_runs()
                    if any(t["id"].startswith("s1/") and t["status"] == "done"
                           for t in state.RunStore(r).state.get("tasks", []))]
        if not previous:
            con.say("沒有之前的第 1 階段結果可以沿用，必須勾選階段 1。")
            return None
        reuse = previous[0].name
        con.say(f"沿用 {reuse} 的第 1 階段結果。")

    ds_opts = []
    for key, spec in datasets.DATASETS.items():
        have, want = datasets.dataset_status(key)
        ds_opts.append((key, spec["label"], have > 0, f"已下載 {have}/{want}" if have else "尚未下載（先選 1 下載）"))
    chosen = con.choose_many("要用哪些資料集？", ds_opts, {"anime", "vtuber", "baseline"})
    if chosen is None:
        return None
    if 3 in stages and not ({"anime", "vtuber"} & set(chosen)):
        con.say("階段 3 需要動畫或 VTuber 資料集至少一個。")
        return None

    hub = settings.get("hub", "ms")
    asr_ids = [c.id for c in ASR_CANDIDATES]
    missing = [c for c in ASR_CANDIDATES if not is_cached(c, hub)]
    download_missing = False
    if missing and (1 in stages and not reuse):
        size = sum(DOWNLOAD_BYTES[c.id] for c in missing)
        con.say("", "這些辨識模型還沒下載：" + "、".join(c.label for c in missing) + f"（約 {size / 1e9:.1f} GB）")
        download_missing = con.confirm("評測時自動下載？（選 n 會略過這些模型）", True)

    options = _model_options(settings)
    tl_opts = [(llm_servers.fingerprint(e), e["name"], usable, note) for e, usable, note in options]
    tl_pick = con.choose_many("要測哪些翻譯模型？", tl_opts, {o[0] for o in tl_opts if o[2]}) if 3 in stages else []
    if tl_pick is None:
        return None
    judge_pick = []
    if 3 in stages or (2 in stages and recordings):
        judge_opts = [(llm_servers.fingerprint(e), e["name"], usable and judge_mod.eligible_judge(e),
                       note if judge_mod.eligible_judge(e) else "只會翻譯，不能當評審")
                      for e, usable, note in options]
        judge_pick = con.choose_many("交叉評審：要用哪些模型當評審？（至少 2 個才能互評，候選不會評自己）",
                                     judge_opts, {o[0] for o in judge_opts if o[2]})
        if judge_pick is None:
            return None
        if len(judge_pick) < 2:
            con.say("只有 1 個評審：評它自己翻譯的部分會標示「自評，分數可能偏高」。")

    restart_ollama = False
    picked = {fp for fp in list(tl_pick) + list(judge_pick)}
    ollama_entries = [e for e, _u, _n in options if llm_servers.fingerprint(e) in picked
                      and llm_servers.model_kind(e) == "ollama"]
    if ollama_entries:
        import local_models

        server = llm_servers.Ollama(local_models.ollama_root(ollama_entries[0]["api_base"]))
        if server.alive() and llm_servers.pid_file_server() is None:
            con.say("", "Ollama 已經在執行，但不是啟動器以「快速設定」開的（顯示記憶體用量和速度會和實際使用不同）。")
            restart_ollama = con.confirm("要以快速設定重新啟動 Ollama 嗎？", True)

    counts = {k: (5 if quick else datasets.DATASETS[k]["n"]) for k in chosen}
    entries = {llm_servers.fingerprint(e): e for e, _u, _n in options}
    plan = {
        "schema": 1, "created": datetime.now().isoformat(timespec="seconds"), "git_rev": git_rev(),
        "seed": datasets.SEED, "quick": quick, "stages": sorted(stages), "datasets": list(chosen),
        "clip_counts": counts, "asr": asr_ids, "download_missing": download_missing,
        "reuse_stage1_from": reuse,
        "recordings": [{"path": str(p), "category": c, "minutes": _minutes(p)} for p, c in recordings] if 2 in stages else [],
        "tl_models": [{"fp": fp, "name": entries[fp]["name"], "model": entries[fp]["model"],
                       "kind": llm_servers.model_kind(entries[fp])} for fp in tl_pick],
        "judges": list(judge_pick), "judge_mode": "cross" if len(judge_pick) >= 2 else "single",
        "together_clips": 3 if quick else 20, "restart_ollama": restart_ollama,
        "settings_snapshot": {k: settings.get(k) for k in ("asr_language", "vad_threshold", "target_language",
                                                           "silence_mode", "silence_duration", "min_speech_duration",
                                                           "max_speech_duration", "vad_min_speech_density")},
    }
    return plan


def _minutes(path) -> float:
    try:
        import soundfile as sf

        return sf.info(str(path)).duration / 60
    except Exception as e:  # unknown format: the estimate falls back to 1 minute
        log.warning(f"Could not read the length of {path}: {e}")
        return 1.0


def _summary(con: Console, plan: dict, settings: dict, store_like) -> None:
    entries = {llm_servers.fingerprint(e): e for e in settings_io.translation_models(settings)}
    con.say("", "======== 開始前確認 ========")
    con.say(" 階段：" + "、".join(str(s) for s in plan["stages"])
            + ("（沿用之前的階段 1）" if plan.get("reuse_stage1_from") else ""))
    con.say(" 資料集：" + "、".join(f"{datasets.DATASETS[k]['label']} {plan['clip_counts'][k]} 句" for k in plan["datasets"]))
    con.say(" 辨識模型：" + "、".join(CANDIDATES_BY_ID[a].label for a in plan["asr"]))
    con.say(" 翻譯模型：" + ("、".join(m["name"] for m in plan["tl_models"]) or "（無）"))
    con.say(" 評審：" + ("、".join(entries[j]["name"] for j in plan["judges"] if j in entries) or "（無）")
            + ("（交叉評審）" if plan["judge_mode"] == "cross" else ""))
    cost_total = 0.0
    unknown = False
    n_tl = sum(plan["clip_counts"].get(c, 0) for c in ("anime", "vtuber"))
    prompt = settings.get("system_prompt") or ""
    for m in plan["tl_models"]:
        tin, tout = costs.translation_tokens(prompt, ["今日はいい天気ですね、散歩に行こうか。"] * n_tl)
        c = costs.call_cost(entries[m["fp"]], tin, tout) if m["fp"] in entries else 0.0
        unknown |= c is None
        cost_total += c or 0.0
    assignment = judge_mod.assign_judges([m["fp"] for m in plan["tl_models"]], plan["judges"])
    for j in plan["judges"]:
        calls = len(judge_mod.CALIBRATION) + n_tl * sum(1 for m in plan["tl_models"]
                                                        if any(x == j for x, _ in assignment[m["fp"]]))
        tin, tout = costs.judge_tokens(calls)
        c = costs.call_cost(entries[j], tin, tout) if j in entries else 0.0
        unknown |= c is None
        cost_total += c or 0.0
    con.say(" 預估時間：" + fmt_duration(estimate_remaining(store_like)))
    con.say(" 預估 API 費用：" + (costs.fmt_cost(cost_total) + ("；有雲端模型沒有設定價格" if unknown else "")))
    con.say(" 執行中請不要開 LiveTranslate；可以隨時關掉視窗，之後選「3) 從上次中斷的地方繼續」。")


class _PlanView:
    """Enough of a RunStore for estimate_remaining before the run exists."""

    def __init__(self, plan):
        self.plan = plan
        self.state = {"tasks": [], "frozen": {}, "timing": {}}

    def done_keys(self, kind, task_id):
        return set()


def _run(con: Console, store, settings, config) -> int:
    if not state.acquire_lock():
        con.say("另一個評測正在執行中。")
        return 1
    _setup_logging(store.dir)
    try:
        Bench(store, settings, config, Deps(app_running=settings_io.find_running_app)).run()
    except KeyboardInterrupt:
        return 130
    finally:
        state.release_lock()
    con.say("", f"報告：{store.dir / 'report.html'}")
    _open(store.dir / "report.html")
    return 0


def cmd_start(con: Console, quick: bool = False) -> int:
    if not _wait_app_closed(con):
        return 0
    if state.running_pid():
        con.say("另一個評測正在執行中。")
        return 1
    try:
        settings = settings_io.load_settings()
        config = settings_io.load_config()
    except settings_io.SettingsError as e:
        con.say(str(e))
        return 1
    plan = build_plan(con, settings, quick=quick)
    if plan is None:
        con.say("已取消。")
        return 0
    _summary(con, plan, settings, _PlanView(plan))
    if not con.confirm("開始評測？", True):
        return 0
    store = state.RunStore.create(plan)
    return _run(con, store, settings, config)


def cmd_resume(con: Console) -> int:
    store = state.RunStore.latest_unfinished()
    if not store:
        con.say("沒有未完成的評測。")
        return 0
    if not _wait_app_closed(con):
        return 0
    failed = [t for t in store.state.get("tasks", []) if t["status"] == "failed"]
    con.say(f"繼續 {store.dir.name}，剩餘約 {describe_estimate(store)}。")
    if failed:
        con.say("上次失敗的項目：")
        for t in failed:
            con.say(f"  - {t['id']}：{t.get('reason')}")
        transient = all("下載" not in str(t.get("reason")) for t in failed)
        if con.confirm("要重試這些項目嗎？", transient):
            for t in failed:
                store.set_task(t["id"], status="pending", reason=None)
    try:
        settings = settings_io.load_settings()
        config = settings_io.load_config()
    except settings_io.SettingsError as e:
        con.say(str(e))
        return 1
    if not con.confirm("開始？", True):
        return 0
    store.state["status"] = "running"
    store.save()
    return _run(con, store, settings, config)


# ---------------------------------------------------------------- report / apply

def _open(path: Path):
    try:
        os.startfile(str(path))  # noqa: S606 - opens the report in the default browser
    except (AttributeError, OSError) as e:
        log.warning(f"Could not open {path}: {e}")


def _pick_run(con: Console, need_done: bool = False):
    runs = state.RunStore.list_runs()
    if need_done:
        runs = [r for r in runs if (r / "recommendations.json").exists()]
    runs = runs[:5]
    if not runs:
        con.say("還沒有評測結果。")
        return None
    status_names = {"done": "完成", "interrupted": "中斷"}
    options = []
    for r in runs:
        st = status_names.get(state.RunStore(r).state.get("status"), "未完成")
        options.append((r, f"{r.name}（{st}）"))
    return con.choose_one("選擇一次評測：", options, 0)


def cmd_report(con: Console) -> int:
    from bench import report

    run_dir = _pick_run(con)
    if run_dir is None:
        return 0
    store = state.RunStore(run_dir)
    report.build(store)
    con.say(f"報告：{run_dir / 'report.html'}")
    _open(run_dir / "report.html")
    return 0


def cmd_apply(con: Console) -> int:
    import json

    if not _wait_app_closed(con):
        return 0
    run_dir = _pick_run(con, need_done=True)
    if run_dir is None:
        return 0
    recs = json.loads((run_dir / "recommendations.json").read_text(encoding="utf-8"))
    try:
        settings = settings_io.load_settings()
    except settings_io.SettingsError as e:
        con.say(str(e))
        return 1
    options = [(r, f"{r['id']} {analysis.CATEGORY_LABELS.get(r['category'], r['category'])}【{'、'.join(r['kinds'])}】"
                   f" {r['asr_label']} + VAD {r['vad_threshold']:.2f} + {r['tl_name']}"
                   + ("" if r["recommended"] else "（不推薦）")) for r in recs]
    backups = settings_io.list_backups()
    if backups:
        options.append(("restore", f"R) 還原上次套用前的設定（{backups[0].name}）"))
    choice = con.choose_one("要套用哪一個組合？", options, 0)
    if choice is None:
        return 0
    if choice == "restore":
        if con.confirm(f"還原 {backups[0].name} 的設定？", False):
            settings_io.restore(backups[0])
            con.say("已還原。")
        return 0
    entry = next((m for m in settings_io.translation_models(settings)
                  if m.get("name") == choice["tl_name"] and m.get("model") == choice["tl_model"]), None)
    if entry is None:
        con.say(f"設定檔裡找不到翻譯模型「{choice['tl_name']}」。")
        return 1
    try:
        changes = settings_io.planned_changes(settings, choice["asr"], choice["vad_threshold"], entry)
    except settings_io.SettingsError as e:
        con.say(str(e))
        return 1
    if not changes:
        con.say("目前的設定已經是這個組合。")
        return 0
    con.say("", "會修改：")
    for key, (old, new) in changes.items():
        con.say(f"  {key}：{old} → {new}")
    if not is_cached(CANDIDATES_BY_ID[choice["asr"]], settings.get("hub", "ms")):
        con.say("  （這個辨識模型還沒下載，LiveTranslate 啟動時會自動下載）")
    if not con.confirm("確定套用？", False):
        return 0
    try:
        saved = settings_io.apply_changes(changes, entry)
    except settings_io.SettingsError as e:
        con.say(str(e))
        return 1
    con.say(f"已套用。原本的設定備份在 {saved}", "下次啟動 LiveTranslate 就會使用這個組合。")
    return 0


COMMANDS = {"status": cmd_status, "download": cmd_download, "start": cmd_start, "resume": cmd_resume,
            "report": cmd_report, "apply": cmd_apply}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError) as e:
            log.debug(f"Could not switch {stream} to UTF-8: {e}")
    _setup_logging()
    if not argv or argv[0] not in COMMANDS:
        print("用法：bench_run.py " + " | ".join(COMMANDS))
        return 2
    con = Console()
    if argv[0] == "start":
        return cmd_start(con, quick="--quick" in argv)
    return COMMANDS[argv[0]](con)
