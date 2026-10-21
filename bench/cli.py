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

from bench import analysis, costs, datasets, judge as judge_mod, llm_servers, paths, report, settings_io, state, vad_runner
from bench.asr_runner import ASR_CANDIDATES, CANDIDATES_BY_ID, is_cached
from bench.orchestrator import Bench, Deps, describe_estimate, estimate_remaining, git_rev
from bench.progress import fmt_duration
from bench.wizard import Console, Option

log = logging.getLogger("LiveTranslate.Bench")


def _setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(getattr(h, "_bench", False) for h in root.handlers):
        h = logging.StreamHandler(sys.stderr)
        h.setLevel(logging.WARNING)
        h.setFormatter(logging.Formatter("\n[%(levelname)s] %(message)s"))
        h._bench = True
        root.addHandler(h)


def _attach_run_log(run_dir: Path, name: str = "bench.log"):
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(run_dir / name, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)
    return fh


def _detach_run_log(fh):
    logging.getLogger().removeHandler(fh)
    fh.close()


def _wait_app_closed(con: Console) -> bool:
    while settings_io.find_running_app():
        con.say("", "LiveTranslate 正在執行。評測和套用設定都需要先關閉它（系統匣圖示 → 結束）。")
        if con.ask("關閉後按 Enter 重新檢查，輸入 0 取消：") == "0":
            return False
    return True


# ---------------------------------------------------------------- status

def cmd_status(con: Console) -> int:
    parts = []
    for key, label in {**analysis.CATEGORY_LABELS, "baseline": "基準"}.items():
        have, want = datasets.dataset_status(key)
        parts.append(f"{label} {have}/{want}" + (" ✓" if have >= want else ""))
    live = datasets.live_recordings()
    con.say(" 測試資料：" + "　".join(parts) + f"｜自己的錄音：{len(live)} 檔")
    run = state.RunStore.latest_unfinished()
    if run:
        tasks = run.state.get("tasks", [])
        finished = sum(1 for t in tasks if t["status"] in state.FINISHED)
        con.say(f" 未完成的評測：{run.dir.name}（已完成 {finished} 個項目，剩餘約 {describe_estimate(run)}）")
    return 0


# ---------------------------------------------------------------- download

def cmd_download(con: Console, source_factory=None) -> int:
    from model_manager import _proxy_env

    con.say("", datasets.LICENSE_NOTE, "")
    total = sum(spec["n"] + datasets.WARMUP_CLIPS for spec in datasets.DATASETS.values())
    con.say(f"會從 Hugging Face 下載 {total} 句（約 {datasets.APPROX_TOTAL_MB} MB；只讀抽到的區塊，不會整包下載），"
            "每次抽到的句子都一樣，大約需要 10–30 分鐘（看網路速度）。")
    if not con.confirm("開始下載？", True):
        return 0
    try:
        settings = settings_io.load_settings()
    except settings_io.SettingsError as e:
        log.warning(f"{e}; using the system proxy setting")
        settings = {}
    log_path = paths.clips_dir() / "download.log"
    fh = _attach_run_log(log_path.parent, log_path.name)
    try:
        with _proxy_env(settings.get("download_proxy", "system")):
            return _download_all(con, source_factory, log_path)
    finally:
        _detach_run_log(fh)


def _progress(k, n, target, skipped=0):
    sys.stdout.write(f"\r  {n}/{target} 句" + (f"（略過 {skipped} 筆）" if skipped else ""))
    sys.stdout.flush()


def _download_all(con: Console, source_factory, log_path) -> int:
    short = []
    for key, spec in datasets.DATASETS.items():
        con.say(f"\n{spec['label']}")
        try:
            source = (source_factory or datasets.make_source)(key)
            m = datasets.download_dataset(key, source, progress=_progress, info=lambda t: con.say(f"  {t}"))
        except datasets.NetworkError as e:
            log.error(f"Download of {key} stopped: {e}")
            con.say("", f"  網路錯誤：{e}", "  已下載的部分會保留，稍後再選 1 會從中斷的地方繼續。")
            return 1
        except datasets.CodecError as e:
            log.error(f"Download of {key} stopped: {e}")
            con.say("", f"  這台電腦無法解碼下載的音檔：{e}",
                    "  資料本身沒有問題，不要刪除已下載的資料。請重新執行安裝或更新（update.bat）修復音訊套件"
                    f"（soundfile / librosa），再選 1 繼續；詳細原因在 {log_path}。")
            return 1
        except datasets.DataError as e:
            log.error(f"Download of {key} stopped: {e}")
            con.say("", f"  資料集和預期的不同，重試也不會好：{e}",
                    f"  如果資料集更新過，刪除 {datasets.manifest_path(key)} 後再選 1 重新抽樣；"
                    f"其他情況請附上 {log_path} 回報問題。")
            return 1
        except datasets.DiskError as e:
            log.error(f"Download of {key} stopped: {e}")
            con.say("", f"  寫入磁碟失敗：{e}", "  請確認磁碟空間和資料夾權限；已下載的部分會保留，之後再選 1 繼續。")
            return 1
        except datasets.DownloadError as e:
            log.error(f"Download of {key} stopped: {e}")
            con.say("", f"  下載停止：{e}", f"  已下載的部分會保留；詳細原因在 {log_path}。")
            return 1
        except Exception as e:  # e.g. a library missing after an update: no traceback, keep the log
            log.exception(f"Download of {key} failed")
            con.say("", f"  下載失敗：{type(e).__name__}: {e}", f"  詳細原因在 {log_path}。")
            return 1
        con.say(f"\r  完成：{len(m['clips'])}/{spec['n']} 句（略過 {len(m['skipped'])} 筆不合格的資料）")
        if len(m["clips"]) < spec["n"]:
            short.append(f"{spec['label']} 只有 {len(m['clips'])}/{spec['n']} 句")
    if short:
        con.say("", "注意：" + "；".join(short) + "。合格的句子不夠，評測會用現有的句子。")
    return 0


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
            try:
                installed = server.installed() if server.alive() else _installed_from_manifests(models)
            except Exception as e:  # a busy server: fall back to the files on disk
                log.warning(f"Could not ask Ollama for its models: {e}")
                installed = _installed_from_manifests(models)
        if kind == "lmstudio" and lm_alive is None:
            lm_alive = llm_servers._port_open(1234)
    from bench.tl_runner import availability

    return [(entry, *availability(entry, installed, lm_alive)) for entry in models]


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


def _reusable_stage1_run(chosen: list, counts: dict, data_ids: dict, asr_language: str):
    """Newest run whose stage 1 finished for every ASR on the same clips (sampling
    method, revision, clip ids) with the same forced language, or None."""
    wanted = {c.id for c in ASR_CANDIDATES}
    for run in state.RunStore.readable_runs():
        status = {t["id"][3:]: t["status"] for t in run.state.get("tasks", []) if t["id"].startswith("s1/")}
        if not wanted <= {a for a, s in status.items() if s in state.FINISHED}:
            continue
        if not any(s == "done" for s in status.values()):
            continue
        if (run.plan["settings_snapshot"] or {}).get("asr_language") != asr_language:
            continue
        old_ids = run.plan["data_ids"] or {}
        same_data = all(d in run.plan["datasets"] and run.plan["clip_counts"].get(d) == counts[d]
                        and data_ids.get(d) is not None and old_ids.get(d) == data_ids[d] for d in chosen)
        if same_data:
            return run.dir.name
    return None


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
    picked = _ask_datasets(con, stages, quick)
    if picked is None:
        return None
    chosen, counts = picked
    data_ids = {k: datasets.sample_id(k, counts[k]) for k in chosen}
    reuse = None
    if 1 not in stages:
        reuse = _reusable_stage1_run(chosen, counts, data_ids, settings.get("asr_language", "auto"))
        if not reuse:
            con.say("沒有完整的第 1 階段結果可以沿用（要 4 個辨識模型都跑完，測試句子和指定的辨識語言都相同），"
                    "必須勾選階段 1。")
            return None
        con.say(f"沿用 {reuse} 的第 1 階段結果。")

    hub = settings.get("hub", "ms")
    asr_ids = [c.id for c in ASR_CANDIDATES]
    missing = [c for c in ASR_CANDIDATES if not is_cached(c, hub)]
    download_missing = False
    if missing and 1 in stages:
        size = sum(c.download_bytes for c in missing)
        con.say("", "這些辨識模型還沒下載：" + "、".join(c.label for c in missing) + f"（約 {size / 1e9:.1f} GB）")
        download_missing = con.confirm("評測時自動下載？（選 n 會略過這些模型）", True)

    options = _model_options(settings)
    tl_opts = [Option(llm_servers.fingerprint(e), e["name"], usable, note) for e, usable, note in options]
    tl_pick = con.choose_many("要測哪些翻譯模型？", tl_opts, {o.key for o in tl_opts if o.enabled}) if 3 in stages else []
    if tl_pick is None:
        return None
    judge_pick = []
    if 3 in stages or (2 in stages and recordings):
        judge_pick = _ask_judges(con, options)
        if judge_pick is None:
            return None
    restart_ollama = _ask_restart_ollama(con, options, set(tl_pick) | set(judge_pick))

    entries = {llm_servers.fingerprint(e): e for e, _u, _n in options}
    return {
        "schema": 1, "created": datetime.now().isoformat(timespec="seconds"), "git_rev": git_rev(),
        "seed": datasets.SEED, "quick": quick, "stages": sorted(stages), "datasets": list(chosen),
        "clip_counts": counts, "data_ids": data_ids, "asr": asr_ids, "download_missing": download_missing,
        "reuse_stage1_from": reuse,
        "recordings": [{"path": str(p), "category": c, "minutes": _minutes(p)} for p, c in recordings] if 2 in stages else [],
        "tl_models": [{"fp": fp, "name": entries[fp]["name"], "model": entries[fp]["model"],
                       "kind": llm_servers.model_kind(entries[fp])} for fp in tl_pick],
        "judges": list(judge_pick), "judge_mode": "cross" if len(judge_pick) >= 2 else "single",
        "together_clips": 3 if quick else 20, "restart_ollama": restart_ollama,
        "tl_timeout_s": settings.get("timeout", 10),
        "settings_snapshot": {k: settings.get(k) for k in ("asr_language", "vad_threshold", "target_language",
                                                           "silence_mode", "silence_duration", "min_speech_duration",
                                                           "max_speech_duration", "vad_min_speech_density")},
    }


def _ask_datasets(con: Console, stages, quick: bool):
    """(chosen dataset keys, {key: clip count}), or None when cancelled."""
    ds_opts = []
    available = {}
    for key, spec in datasets.DATASETS.items():
        have = len(datasets.load_clips(key)[1])
        available[key] = have
        ds_opts.append(Option(key, spec["label"], have > 0,
                              f"已下載 {have}/{spec['n']}" if have else "尚未下載（先選 1 下載）"))
    chosen = con.choose_many("要用哪些資料集？", ds_opts, {"anime", "vtuber", "baseline"})
    if chosen is None:
        return None
    if 3 in stages and not (set(analysis.CATEGORIES) & set(chosen)):
        con.say("階段 3 需要動畫或 VTuber 資料集至少一個。")
        return None
    counts = {}
    for k in chosen:
        want = 5 if quick else datasets.DATASETS[k]["n"]
        counts[k] = min(want, available[k])
        if counts[k] < want:
            con.say(f"注意：「{datasets.DATASETS[k]['label']}」只有 {available[k]} 句（目標 {want}），會用這 {available[k]} 句。")
    return chosen, counts


def _ask_judges(con: Console, options):
    """Judge fingerprints, or None when cancelled / no model can judge."""
    judge_opts = [Option(llm_servers.fingerprint(e), e["name"], usable and judge_mod.eligible_judge(e),
                         note if judge_mod.eligible_judge(e) else "只會翻譯，不能當評審")
                  for e, usable, note in options]
    if not any(o.enabled for o in judge_opts):
        con.say("沒有可以當評審的模型（Sakura 這類只會翻譯的模型不能評分），翻譯品質無法評分。")
        return None
    picked = con.choose_many("交叉評審：要用哪些模型當評審？（至少 2 個才能互評，候選不會評自己）",
                             judge_opts, {o.key for o in judge_opts if o.enabled})
    if picked is not None and len(picked) < 2:
        con.say("只有 1 個評審：評它自己翻譯的部分會標示「自評，分數可能偏高」。")
    return picked


def _ask_restart_ollama(con: Console, options, picked: set) -> bool:
    """A tray-started Ollama has none of the launcher's fast settings: offer a restart."""
    ollama_entries = [e for e, _u, _n in options if llm_servers.fingerprint(e) in picked
                      and llm_servers.model_kind(e) == "ollama"]
    if not ollama_entries:
        return False
    import local_models

    server = llm_servers.Ollama(local_models.ollama_root(ollama_entries[0]["api_base"]))
    if server.alive() and llm_servers.pid_file_server() is None:
        con.say("", "Ollama 已經在執行，但不是啟動器以「快速設定」開的（顯示記憶體用量和速度會和實際使用不同）。")
        return con.confirm("要以快速設定重新啟動 Ollama 嗎？", True)
    return False


def _minutes(path) -> float:
    try:
        import soundfile as sf

        return sf.info(str(path)).duration / 60
    except Exception as e:  # unknown format: the estimate falls back to 1 minute
        log.warning(f"Could not read the length of {path}: {e}")
        return 1.0


def estimate_cost(plan: dict, settings: dict):
    """(US$ total, any cloud model without prices) for translations (alone, warm-up and
    together) and judging (translations, calibration, stage-2 windows)."""
    entries = settings_io.models_by_fp(settings)
    cats = [c for c in analysis.CATEGORIES if c in plan["datasets"]]
    n_tl = sum(plan["clip_counts"].get(c, 0) for c in cats)
    n_together = plan["together_clips"] * 2 * len(cats)  # 2 ASRs per category
    prompt = settings.get("system_prompt") or ""
    sample = "今日はいい天気ですね、散歩に行こうか。"
    total, unknown = 0.0, False
    for m in plan["tl_models"]:
        tin, tout = costs.translation_tokens(prompt, [sample] * (n_tl + n_together + 6))
        c = costs.call_cost(entries[m["fp"]], tin, tout) if m["fp"] in entries else 0.0
        unknown |= c is None
        total += c or 0.0
    candidates = [m["fp"] for m in plan["tl_models"]]
    assignment = judge_mod.assign_judges(candidates, plan["judges"])
    rec_cats = {r["category"] for r in plan["recordings"]}
    minutes = sum(r.get("minutes", 1.0) for r in plan["recordings"])
    windows = int(vad_runner.estimate_windows(minutes, min(3, 2 * len(rec_cats))))
    for i, j in enumerate(plan["judges"]):
        calls = len(judge_mod.CALIBRATION) + n_tl * judge_mod.assigned_count(assignment, candidates, j)
        tin, tout = costs.judge_tokens(calls)
        if i == 0 and windows:
            vin, vout = costs.vad_judge_tokens(windows)
            tin, tout = tin + vin, tout + vout
        c = costs.call_cost(entries[j], tin, tout) if j in entries else 0.0
        unknown |= c is None
        total += c or 0.0
    return total, unknown


def _summary(con: Console, plan: dict, settings: dict, store_like) -> None:
    entries = settings_io.models_by_fp(settings)
    con.say("", "======== 開始前確認 ========")
    con.say(" 階段：" + "、".join(str(s) for s in plan["stages"])
            + ("（沿用之前的階段 1）" if plan["reuse_stage1_from"] else ""))
    con.say(" 資料集：" + "、".join(f"{datasets.DATASETS[k]['label']} {plan['clip_counts'][k]} 句" for k in plan["datasets"]))
    con.say(" 辨識模型：" + "、".join(CANDIDATES_BY_ID[a].label for a in plan["asr"]))
    con.say(" 翻譯模型：" + ("、".join(m["name"] for m in plan["tl_models"]) or "（無）"))
    con.say(" 評審：" + ("、".join(entries[j]["name"] for j in plan["judges"] if j in entries) or "（無）")
            + ("（交叉評審）" if plan["judge_mode"] == "cross" else ""))
    total, unknown = estimate_cost(plan, settings)
    con.say(" 預估時間：" + fmt_duration(estimate_remaining(store_like)))
    con.say(" 預估 API 費用：" + costs.fmt_cost(total) + ("；有雲端模型沒有設定價格" if unknown else ""))
    con.say(" 執行中請不要開 LiveTranslate；可以隨時關掉視窗，之後選「3) 從上次中斷的地方繼續」。")


class _PlanView:
    """Enough of a RunStore for estimate_remaining before the run exists."""

    def __init__(self, plan):
        self.plan = {**state.PLAN_DEFAULTS, **plan}
        self.state = {"tasks": [], "frozen": {}, "timing": {}}

    def final_keys(self, kind, task_id):
        return set()


def _run(con: Console, store, settings, config, deps=None) -> int:
    if not state.acquire_lock():
        con.say("另一個評測正在執行中。")
        return 1
    fh = _attach_run_log(store.dir)
    try:
        Bench(store, settings, config, deps or Deps(app_running=settings_io.find_running_app)).run()
    except KeyboardInterrupt:
        return 130
    except Exception as e:  # Bench.run logged it with the traceback; errors before it ran did not
        if not store.state.get("error"):
            log.exception("Benchmark stopped")
        con.say("", f"評測因錯誤停止：{e}", f"詳細原因在 {store.dir / 'bench.log'}。修正後可以選「3) 從上次中斷的地方繼續」。")
        return 1
    finally:
        _detach_run_log(fh)
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
    loaded = _load_settings_config(con)
    if loaded is None:
        return 1
    settings, config = loaded
    if not quick:
        mode = con.choose_one("要跑哪一種？", [
            ("full", "完整評測：動畫 200 句、VTuber 200 句、基準 100 句（約 1.5–3 小時）"),
            ("quick", "快速試跑：每個資料集 5 句（約 10–20 分鐘），先確認整個流程正常")], 0)
        if mode is None:
            return 0
        quick = mode == "quick"
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
    if not state.acquire_lock():
        con.say("另一個評測正在執行中，請等它結束（或關掉它的視窗）再繼續。")
        return 1
    try:
        loaded = _load_settings_config(con)
        if loaded is None:
            return 1
        settings, config = loaded
        settings = _with_run_language(con, store, settings)
        failed = [t for t in store.state.get("tasks", []) if t["status"] == "failed"]
        con.say(f"繼續 {store.dir.name}，剩餘約 {describe_estimate(store)}。")
        retry = False
        if failed:
            con.say("上次失敗的項目：")
            for t in failed:
                con.say(f"  - {t['id']}：{t.get('reason')}")
            transient = all("下載" not in str(t.get("reason")) for t in failed)
            retry = con.confirm("要重試這些項目嗎？（只會重做失敗的部分）", transient)
            if retry and any(t["id"].startswith("s1/") for t in failed) and store.frozen("rankings") is not None:
                con.say("注意：辨識排名已經固定，重試的辨識結果只會更新階段 1 的表格，不會改變後面用哪個模型。")
        if not con.confirm("開始？", True):
            return 0  # nothing changed: the failure reasons stay for the report
        if retry:
            for t in failed:
                store.set_task(t["id"], status="pending", reason=None)
        # judges run again over anything new (scored items are skipped)
        for t in store.state.get("tasks", []):
            if t["id"].startswith("s3judge/") or t["id"] == "s2judge":
                store.set_task(t["id"], status="pending", reason=None)
        store.set_status("running")
    finally:
        state.release_lock()
    return _run(con, store, settings, config)


def _load_settings_config(con: Console):
    """(settings, config), or None after telling the user what is wrong."""
    try:
        return settings_io.load_settings(), settings_io.load_config()
    except settings_io.SettingsError as e:
        con.say(str(e))
        return None


def _with_run_language(con: Console, store, settings: dict) -> dict:
    """A resumed run keeps the forced ASR language it started with, so its results stay
    comparable (stage 3 translates what stage 1 recognised)."""
    started = (store.plan["settings_snapshot"] or {}).get("asr_language")
    current = settings.get("asr_language", "auto")
    if started and started != current:
        con.say(f"這次評測開始時指定的辨識語言是 {started}，目前的設定是 {current}；"
                f"繼續時沿用 {started}，結果才能和前面的比較。")
        return {**settings, "asr_language": started}
    return settings


# ---------------------------------------------------------------- report / apply

def _open(path: Path):
    try:
        os.startfile(str(path))  # noqa: S606 - opens the report in the default browser
    except (AttributeError, OSError) as e:
        log.warning(f"Could not open {path}: {e}")
        print(f"請手動打開：{path}")


CANCELLED = object()  # the user chose 0 (as opposed to "nothing to choose from")


def _pick_run(con: Console, need_recs: bool = False):
    """A RunStore of the 5 newest readable runs (with recommendations when asked), None
    when there is none, or CANCELLED."""
    runs = []
    for run in state.RunStore.readable_runs():
        if need_recs and not ((run.dir / "recommendations.json").exists() and report.load_recs(run.dir)["items"]):
            continue
        runs.append(run)
        if len(runs) == 5:
            break
    if not runs:
        con.say("還沒有評測結果。" if not need_recs else "還沒有產生推薦組合的評測（要跑完階段 3）。")
        return None
    status_names = {"done": "完成", "interrupted": "中斷", "error": "錯誤"}
    options = [(run, f"{run.dir.name}（{status_names.get(run.state.get('status'), '未完成')}）") for run in runs]
    picked = con.choose_one("選擇一次評測：", options, 0)
    return CANCELLED if picked is None else picked


def cmd_report(con: Console) -> int:
    store = _pick_run(con)
    if store is None or store is CANCELLED:
        return 0
    try:
        report.build(store)
    except Exception as e:  # show what failed instead of a traceback
        log.exception("Report failed")
        con.say(f"報告產生失敗：{e}")
        return 1
    con.say(f"報告：{store.dir / 'report.html'}")
    _open(store.dir / "report.html")
    return 0


def _rec_label(r: dict) -> str:
    vad = f"VAD {r['vad_threshold']:.2f}" if r["vad_measured"] else "VAD 維持目前設定（未實測）"
    return (f"{r['id']} {analysis.CATEGORY_LABELS.get(r['category'], r['category'])}【{'、'.join(r['kinds'])}】"
            f" {r['asr_label']} + {vad} + {r['tl_name']}" + ("" if r["recommended"] else "（不推薦）"))


def cmd_apply(con: Console) -> int:
    if not _wait_app_closed(con):
        return 0
    try:
        return _apply(con)
    except settings_io.SettingsError as e:
        con.say(str(e))
        return 1


def _apply(con: Console) -> int:
    backups = settings_io.list_backups()
    run = _pick_run(con, need_recs=True)
    if run is CANCELLED or (run is None and not backups):
        return 0
    recs = report.load_recs(run.dir) if run else {"items": [], "status": None}
    if run and recs["status"] != "done":
        con.say("這次評測沒有跑完，推薦是用不完整的結果算的。")
        if not con.confirm("還是要從這次評測套用嗎？", False):
            return 0
    options = [(r, _rec_label(r)) for r in recs["items"]]
    if backups:
        options.append(("restore", f"R) 還原上次套用前的設定（{backups[0].name}）"))
    choice = con.choose_one("要套用哪一個組合？", options, 0)
    if choice is None:
        return 0
    if choice == "restore":  # before reading the settings: a broken file can still be restored
        if con.confirm(f"還原 {backups[0].name} 的設定？（目前的設定會先備份）", False):
            settings_io.restore(backups[0])
            con.say("已還原。")
        return 0
    settings = settings_io.load_settings()
    entry = next((m for fp, m in settings_io.models_by_fp(settings).items()
                  if fp == choice["tl_fp"]), None)
    if entry is None:
        con.say(f"設定檔裡找不到翻譯模型「{choice['tl_name']}」。")
        return 1
    threshold = choice["vad_threshold"] if choice["vad_measured"] else None
    changes = settings_io.planned_changes(settings, choice["asr"], threshold, entry)
    if not changes:
        con.say("目前的設定已經是這個組合。")
        return 0
    con.say("", "會修改：")
    for key, (old, new) in changes.items():
        con.say(f"  {key}：{old} → {new}")
    if threshold is None:
        con.say("  （語音偵測閾值沒有實測，維持你目前的設定）")
    if not is_cached(CANDIDATES_BY_ID[choice["asr"]], settings.get("hub", "ms")):
        con.say("  （這個辨識模型還沒下載，LiveTranslate 啟動時會自動下載）")
    for note in settings_io.apply_notes(entry):
        con.say(f"  注意：{note}")
    if not con.confirm("確定套用？", False):
        return 0
    saved = settings_io.apply_changes(changes, entry)
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
