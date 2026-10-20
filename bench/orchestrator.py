"""Runs a benchmark plan stage by stage, resumable at sentence level.

Order (fixed so a resume continues where it stopped):
  stage 1  s1/<asr>                each ASR loaded once, all datasets
           -> rankings frozen (top 2 per category, best ASR)
  stage 2  s2asr/<asr>             top-2 ASRs on the user's recordings (VAD segments
                                   per threshold + continuous reference)
  stage 3  s3tl/<model>            each translation model alone on the best ASR's text
           s3tog/<asr>/<model>     ASR + model loaded together, real-time replay
           s3judge/<judge>         each judge alone: calibration, its assigned
                                   translations (cross judging) and, for the first judge,
                                   the stage-2 windows
  report

Every finished item is appended to results/<kind>.jsonl before the next starts. A model
that cannot be downloaded or loaded is recorded with the reason and skipped."""
import concurrent.futures
import ctypes
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field

from bench import analysis, asr_runner, datasets, gpu, judge as judge_mod, llm_servers, report, tl_runner, vad_runner
from bench.progress import Ewma, ProgressLine, fmt_duration
from bench.state import RunStore

log = logging.getLogger("LiveTranslate.Bench")

ASR_WARMUP = datasets.WARMUP_CLIPS
FAIL_FAST_ITEMS = 10      # a model failing its first 10 sentences is broken, not slow
MAX_WORKER_ERRORS = 3
TOGETHER_RELEASE_PAD_S = 0.5
DEFAULT_SECONDS = {  # per item, before any measurement
    "asr:sensevoice": 0.15, "asr:mlt-nano": 0.6, "asr:anime-whisper": 0.5, "asr:whisper-turbo": 0.35,
    "tl": 0.9, "judge": 2.0, "load": 30.0}


@dataclass
class Deps:
    """Injectable collaborators (tests pass fakes)."""
    asr_client_factory: object = None                      # config -> ASRClient-like
    translator_factory: object = None                      # (entry, settings, config) -> Translator-like
    judge_factory: object = None                           # entry -> JudgeClient-like
    ollama_factory: object = None                          # root url -> Ollama-like
    vram: object = None                                    # VramReader-like
    app_running: object = None                             # () -> [pid]
    vad_factory: object = None                             # (threshold, settings) -> VADProcessor-like
    load_audio: object = None                              # path -> float32 16 kHz
    sleep: object = time.sleep
    out: object = None                                     # text stream for the progress line
    is_cached: object = None                               # cand -> bool
    download: object = None                                # cand -> None (raises)
    together_realtime: bool = True
    extra: dict = field(default_factory=dict)


def _prevent_sleep(on: bool):
    if sys.platform != "win32":
        return
    try:
        es_continuous, es_system = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | (es_system if on else 0))
    except (AttributeError, OSError) as e:
        log.warning(f"Could not change the sleep setting: {e}")


class Bench:
    def __init__(self, store: RunStore, settings: dict, config: dict, deps: Deps | None = None):
        self.store = store
        self.settings = settings
        self.config = config
        self.d = deps or Deps()
        self.vram = self.d.vram if self.d.vram is not None else gpu.VramReader()
        self.progress = ProgressLine(out=self.d.out)
        self.forced = settings.get("asr_language", config.get("asr", {}).get("language", "auto"))
        self._entries = {llm_servers.fingerprint(m): m for m in settings.get("models", []) if isinstance(m, dict)}
        self._ollama_started_pid = None
        self._ollama_restarted = False
        self.baseline_mb = None
        self._timing = dict(store.state.get("timing") or {})

    # ------------------------------------------------------------------ helpers
    def _check_app(self):
        """Pause while the user's LiveTranslate runs (GPU and Ollama would be shared)."""
        if not self.d.app_running:
            return
        warned = False
        while self.d.app_running():
            if not warned:
                self.progress.finish("偵測到 LiveTranslate 正在執行，評測暫停。請關閉它（系統匣按「結束」），會自動繼續…")
                warned = True
            self.d.sleep(5.0)

    def _settle(self):
        result = gpu.wait_vram_settle(self.vram, self.baseline_mb, sleep=self.d.sleep)
        if result["status"] not in ("settled", "no_nvml"):
            log.warning(f"VRAM settle: {result}")
        return result

    def _audio(self, path):
        return (self.d.load_audio or datasets.load_audio_file)(path)

    def _record_timing(self, key, seconds):
        prev = self._timing.get(key)
        self._timing[key] = seconds if prev is None else 0.3 * seconds + 0.7 * prev
        self.store.state["timing"] = self._timing

    def _per_item(self, key):
        return self._timing.get(key) or DEFAULT_SECONDS.get(key) or DEFAULT_SECONDS["tl"]

    def _remaining_estimate(self) -> float:
        return estimate_remaining(self.store, self._timing)

    def _ollama(self, entry):
        import local_models

        root = local_models.ollama_root(entry["api_base"])
        return (self.d.ollama_factory or llm_servers.Ollama)(root)

    def _unload_llms(self):
        """Unload every local model (Ollama: all, LM Studio: lms unload --all)."""
        roots = set()
        for entry in self._entries.values():
            kind = llm_servers.model_kind(entry)
            if kind == "ollama":
                import local_models

                roots.add(local_models.ollama_root(entry["api_base"]))
            elif kind == "lmstudio":
                if not llm_servers.lmstudio_unload_all():
                    log.info("LM Studio has no CLI (lms); waiting for VRAM instead of unloading")
        for root in roots:
            server = (self.d.ollama_factory or llm_servers.Ollama)(root)
            if server.alive():
                server.unload_all(sleep=self.d.sleep)

    def _ensure_ollama(self, entry) -> bool:
        """Ollama answering with the launcher's fast settings. A tray-started Ollama
        (no fast settings) is restarted when the user agreed before the run."""
        server = self._ollama(entry)
        alive = server.alive()
        restart = alive and self.store.plan.get("restart_ollama") and not self._ollama_restarted
        if alive and not restart:
            return True
        if self.d.ollama_factory is not None:  # tests: no real process
            return alive
        port = int(entry["api_base"].split("://", 1)[-1].split("/", 1)[0].rsplit(":", 1)[-1])
        if restart:
            llm_servers.stop_tray_ollama()
            self._ollama_restarted = True
        try:
            self._ollama_started_pid = llm_servers.start_fast_ollama(port, self.store.dir)
            self.store.freeze("ollama_started_by_bench", self._ollama_started_pid)
            return True
        except (OSError, RuntimeError) as e:
            log.error(f"Could not start Ollama: {e}")
            return False

    def _load_llm(self, entry) -> float:
        """Load a local model alone; returns load seconds (0 for cloud / LM Studio)."""
        kind = llm_servers.model_kind(entry)
        if kind != "ollama":
            return 0.0
        if not self._ensure_ollama(entry):
            raise RuntimeError("Ollama 沒有回應")
        server = self._ollama(entry)
        installed = server.installed()
        if entry["model"] not in installed:
            raise RuntimeError(f"Ollama 裡沒有 {entry['model']}（請先在啟動視窗下載）")
        return server.load(entry["model"])

    # ------------------------------------------------------------------ run
    def run(self):
        _prevent_sleep(True)
        try:
            self._check_app()
            self._unload_llms()
            self.baseline_mb = self.vram.used_mb()
            self.store.freeze("vram_baseline_mb", self.baseline_mb)
            if self.baseline_mb is not None and self.baseline_mb > 1536:
                self.progress.finish(f"注意：開始前顯示卡已經用了 {self.baseline_mb / 1024:.1f} GB，"
                                     "其他程式（瀏覽器、動態桌布）會影響顯示記憶體的量測。")
            stages = self.store.plan.get("stages", [1, 2, 3])
            if self.store.plan.get("reuse_stage1_from"):
                self._import_stage1()
            elif 1 in stages:
                self.stage1()
            self._freeze_rankings()
            if 2 in stages and self.store.plan.get("recordings"):
                self.stage2_asr()
            if 3 in stages:
                self.stage3_translate()
                self.stage3_together()
            if 3 in stages or (2 in stages and self.store.plan.get("recordings")):
                self.stage_judges()
            self.store.state["status"] = "done"
            self.store.save()
            report.build(self.store)
            self.progress.finish("評測完成。")
        except KeyboardInterrupt:
            self.store.state["status"] = "interrupted"
            self.store.save()
            self.progress.finish("已中斷。之後在選單選「3) 從上次中斷的地方繼續」。")
            raise
        finally:
            try:
                self._unload_llms()
            except Exception as e:  # cleanup must not hide the real outcome
                log.warning(f"Unloading models at the end failed: {e}")
            if self._ollama_started_pid:
                llm_servers.stop_ollama(self._ollama_started_pid)
            _prevent_sleep(False)
            self.store.save()

    # ------------------------------------------------------------------ stage 1
    def _import_stage1(self):
        src = RunStore(self.store.dir.parent / self.store.plan["reuse_stage1_from"])
        if self.store.read("asr"):
            return
        for kind in ("asr", "asr_meta"):
            for r in src.read(kind):
                self.store.append(kind, r)
        self.store.set_task("s1/imported", status="done", reason=f"沿用 {src.dir.name}")

    def _freeze_rankings(self):
        rankings = analysis.asr_rankings(self.store)
        self.store.freeze("rankings", rankings)
        top2 = {cat: ids[:2] for cat, ids in rankings.items()}
        self.store.freeze("top2", top2)

    def stage1(self):
        plan = self.store.plan
        for asr_id in plan["asr"]:
            cand = asr_runner.CANDIDATES_BY_ID[asr_id]
            task_id = f"s1/{asr_id}"
            task = self.store.task(task_id)
            if task["status"] in ("done", "failed"):
                continue
            self._check_app()
            self._run_asr_task(cand, task_id)

    def _clips(self, dataset):
        warm, clips = datasets.load_clips(dataset, limit=self.store.plan.get("clip_counts", {}).get(dataset))
        return warm, clips

    def _ensure_asr_model(self, cand, task_id) -> bool:
        hub = self.settings.get("hub", "ms")
        cached = (self.d.is_cached or (lambda c: asr_runner.is_cached(c, hub)))(cand)
        if cached:
            return True
        if not self.store.plan.get("download_missing"):
            self._asr_failed(cand, task_id, "模型尚未下載（開始前選擇不下載）")
            return False
        self.progress.finish(f"下載 {cand.label}…")
        try:
            (self.d.download or (lambda c: asr_runner.download(c, hub, self.settings.get("download_proxy", "system"))))(cand)
            return True
        except Exception as e:  # any download error: record and skip this model
            self._asr_failed(cand, task_id, f"下載失敗：{e}")
            return False

    def _asr_failed(self, cand, task_id, reason):
        log.error(f"{cand.label}: {reason}")
        self.store.append("asr_meta", {"task": task_id, "asr": cand.id, "status": "failed", "reason": str(reason)[:300]})
        self.store.set_task(task_id, status="failed", reason=str(reason)[:300])
        self.progress.finish(f"略過 {cand.label}：{reason}")

    def _open_session(self, cand, task_id):
        """Loaded AsrSession, or None after a second failed load (recorded)."""
        config = asr_runner.build_worker_config(cand, self.settings, self.config)
        for attempt in (1, 2):
            session = asr_runner.AsrSession(cand, config, client_factory=self.d.asr_client_factory)
            try:
                session.load()
                return session
            except Exception as e:  # load errors are model-specific: retry once, then skip
                log.warning(f"{cand.label} failed to load (attempt {attempt}): {e}")
                self._settle()
                if attempt == 2:
                    self._asr_failed(cand, task_id, f"載入失敗：{e}")
        return None

    def _run_asr_task(self, cand, task_id):
        plan = self.store.plan
        if not self._ensure_asr_model(cand, task_id):
            return
        self._settle()
        hint_missing = asr_runner.hint_missing_for(cand, self.forced)
        done = self.store.done_keys("asr", task_id)
        total = sum(len(self._clips(ds)[1]) for ds in plan["datasets"])
        self.store.set_task(task_id, status="running", items_total=total, attempts=self.store.task(task_id)["attempts"] + 1)
        ewma = Ewma()
        errors = 0
        with gpu.VramSampler(self.vram) as sampler:
            session = self._open_session(cand, task_id)
            if session is None:
                return
            try:
                warm, _ = self._clips(plan["datasets"][0])
                for clip in warm[:ASR_WARMUP]:
                    session.transcribe(self._audio(clip.wav))
                count = len(done)
                for ds in plan["datasets"]:
                    for clip in self._clips(ds)[1]:
                        key = f"{ds}/{clip.id}"
                        if key in done:
                            continue
                        try:
                            result, ms = session.transcribe(self._audio(clip.wav))
                        except Exception as e:  # worker died twice in a row
                            errors += 1
                            log.error(f"{cand.label} failed on {key}: {e}")
                            if errors >= MAX_WORKER_ERRORS:
                                raise
                            session = self._open_session(cand, task_id)
                            if session is None:
                                return
                            continue
                        raw = (result or {}).get("text", "") if result else ""
                        label = (result or {}).get("language", "") if result else ""
                        shown, reason = asr_runner.displayed_text(raw, label, clip.duration, self.forced, hint_missing)
                        self.store.append("asr", {"task": task_id, "key": key, "asr": cand.id, "dataset": ds,
                                                  "clip": clip.id, "ref": clip.ref, "raw": raw, "label": label,
                                                  "display": shown, "reason": reason, "ms": ms,
                                                  "duration": clip.duration})
                        count += 1
                        avg = ewma.add(ms)
                        self._record_timing(f"asr:{cand.id}", avg / 1000)
                        self.progress.show(f"階段 1/3 辨識｜{cand.label}", f"{ds}", count, total, avg,
                                           self._remaining_estimate())
            except Exception as e:  # incomplete: keep what was measured, do not rank it
                self.store.append("asr_meta", {"task": task_id, "asr": cand.id, "status": "incomplete",
                                               "reason": str(e)[:300], "load_s": session.load_s if session else None,
                                               "vram_peak_mb": sampler.peak_mb, "vram_base_mb": self.baseline_mb})
                self.store.set_task(task_id, status="failed", reason=f"結果不完整：{e}")
                return
            finally:
                if session:
                    session.close()
        self.store.append("asr_meta", {"task": task_id, "asr": cand.id, "status": "ok", "load_s": session.load_s,
                                       "restarts": session.restarts, "vram_peak_mb": sampler.peak_mb,
                                       "vram_base_mb": self.baseline_mb})
        self.store.set_task(task_id, status="done", items_done=total)
        self.progress.finish(f"{cand.label} 完成")
        self._settle()

    # ------------------------------------------------------------------ stage 2
    def _top2_union(self, categories=None):
        top2 = self.store.frozen("top2", {})
        out = []
        for cat in (categories or analysis.CATEGORIES):
            for a in top2.get(cat, []):
                if a not in out:
                    out.append(a)
        return out

    def stage2_asr(self):
        recordings = self.store.plan.get("recordings", [])
        cats = sorted({r["category"] for r in recordings})
        for asr_id in self._top2_union(cats):
            cand = asr_runner.CANDIDATES_BY_ID[asr_id]
            task_id = f"s2asr/{asr_id}"
            if self.store.task(task_id)["status"] in ("done", "failed"):
                continue
            self._check_app()
            self._settle()
            done = self.store.done_keys("vad_raw", task_id)
            hint_missing = asr_runner.hint_missing_for(cand, self.forced)
            session = self._open_session(cand, task_id)
            if session is None:
                continue
            try:
                for rec_no, rec in enumerate(recordings, 1):
                    if asr_id not in self.store.frozen("top2", {}).get(rec["category"], []):
                        continue
                    audio = self._audio(rec["path"])
                    minutes = len(audio) / vad_runner.SR / 60
                    key = f"{rec['path']}/continuous"
                    if key not in done:
                        windows = []
                        for s, e in vad_runner.continuous_windows(audio):
                            result, _ms = session.transcribe(audio[s:e])
                            windows.append({"start": s / vad_runner.SR, "end": e / vad_runner.SR,
                                            "text": (result or {}).get("text", "") if result else ""})
                        self.store.append("vad_raw", {"task": task_id, "key": key, "asr": asr_id,
                                                      "recording": rec["path"], "category": rec["category"],
                                                      "threshold": None, "windows": windows, "minutes": minutes})
                    for th in vad_runner.THRESHOLDS:
                        key = f"{rec['path']}/{th}"
                        if key in done:
                            continue
                        vad = (self.d.vad_factory or vad_runner.make_vad)(th, self.settings)
                        lines = []
                        speech_s = 0.0
                        for seg in vad_runner.segment_recording(audio, vad):
                            speech_s += seg["end"] - seg["start"]
                            result, _ms = session.transcribe(seg["audio"])
                            raw = (result or {}).get("text", "") if result else ""
                            label = (result or {}).get("language", "") if result else ""
                            shown, _reason = asr_runner.displayed_text(raw, label, seg["end"] - seg["start"],
                                                                       self.forced, hint_missing)
                            if shown:
                                lines.append({"start": seg["start"], "end": seg["end"], "text": shown})
                        self.store.append("vad_raw", {"task": task_id, "key": key, "asr": asr_id,
                                                      "recording": rec["path"], "category": rec["category"],
                                                      "threshold": th, "lines": lines, "segments": len(lines),
                                                      "speech_s": speech_s, "minutes": minutes})
                        self.progress.show("階段 2/3 語音偵測", f"{cand.label}｜閾值 {th}｜錄音",
                                           rec_no, len(recordings), None, self._remaining_estimate(), force=True)
            except Exception as e:  # one recording / ASR failing does not stop the run
                log.error(f"Stage 2 with {cand.label} failed: {e}")
                self.store.set_task(task_id, status="failed", reason=str(e)[:300])
                continue
            finally:
                session.close()
            self.store.set_task(task_id, status="done")
        self._settle()

    # ------------------------------------------------------------------ stage 3
    def _tl_inputs(self):
        """{category: [(clip, displayed text of the best ASR)]}"""
        rankings = self.store.frozen("rankings", {})
        records = {(r["asr"], r["dataset"], r["clip"]): r for r in self.store.read("asr")}
        out = {}
        for cat in analysis.CATEGORIES:
            if cat not in self.store.plan["datasets"] or not rankings.get(cat):
                continue
            best = rankings[cat][0]
            rows = []
            for clip in self._clips(cat)[1]:
                r = records.get((best, cat, clip.id))
                if r is not None:
                    rows.append((clip, r.get("display") or ""))
            out[cat] = rows
        return out

    def _translator(self, entry):
        return (self.d.translator_factory or tl_runner.build_translator)(entry, self.settings, self.config)

    def stage3_translate(self):
        inputs = self._tl_inputs()
        for m in self.store.plan["tl_models"]:
            entry = self._entries.get(m["fp"])
            task_id = f"s3tl/{m['fp']}"
            if self.store.task(task_id)["status"] in ("done", "failed"):
                continue
            if entry is None:
                self.store.set_task(task_id, status="failed", reason="設定檔裡已沒有這個模型")
                continue
            self._check_app()
            self._unload_llms()
            self._settle()
            done = self.store.done_keys("tl", task_id)
            total = sum(len(v) for v in inputs.values())
            with gpu.VramSampler(self.vram) as sampler:
                try:
                    load_s = self._load_llm(entry)
                except Exception as e:  # missing / unloadable model: skip it
                    self._tl_failed(m, task_id, f"載入失敗：{e}")
                    continue
                translator = self._translator(entry)
                for text in _warmup_sentences():
                    tl_runner.translate_timed(translator, text)
                count = len(done)
                tried = failed = 0
                ewma = Ewma()
                broken = None
                for cat, rows in inputs.items():
                    for clip, source in rows:
                        key = f"{cat}/{clip.id}"
                        if key in done:
                            continue
                        if not source:
                            rec = {"ok": False, "text": "", "ttft_ms": None, "total_ms": None, "error": "asr_empty"}
                        else:
                            rec = tl_runner.translate_timed(translator, source)
                            tried += 1
                            failed += 0 if rec["ok"] else 1
                            if rec["ok"]:
                                self._record_timing("tl", ewma.add(rec["total_ms"]) / 1000)
                        self.store.append("tl", {"task": task_id, "key": key, "model": m["fp"], "category": cat,
                                                 "clip": clip.id, "ref": clip.ref, "source": source, **rec})
                        count += 1
                        self.progress.show(f"階段 3/3 翻譯｜{m['name']}", cat, count, total, ewma.value,
                                           self._remaining_estimate())
                        if tried == FAIL_FAST_ITEMS and failed == tried:
                            broken = rec.get("error")
                            break
                    if broken:
                        break
            if broken:
                self._tl_failed(m, task_id, f"前 {FAIL_FAST_ITEMS} 句全部失敗（{broken}）", peak=sampler.peak_mb)
                continue
            self.store.append("tl_meta", {"task": task_id, "model": m["fp"], "status": "ok", "load_s": load_s,
                                          "vram_peak_mb": sampler.peak_mb})
            self.store.set_task(task_id, status="done", items_done=total)
            self.progress.finish(f"{m['name']} 翻譯完成")
        self._unload_llms()
        self._settle()

    def _tl_failed(self, m, task_id, reason, peak=None):
        log.error(f"{m['name']}: {reason}")
        self.store.append("tl_meta", {"task": task_id, "model": m["fp"], "status": "failed", "reason": reason,
                                      "vram_peak_mb": peak})
        self.store.set_task(task_id, status="failed", reason=reason)
        self.progress.finish(f"略過 {m['name']}：{reason}")

    def stage3_together(self):
        top2 = self.store.frozen("top2", {})
        n_clips = self.store.plan.get("together_clips", 20)
        failed_models = {t["id"].split("/", 1)[1] for t in self.store.state["tasks"]
                         if t["id"].startswith("s3tl/") and t["status"] == "failed"}
        for asr_id in self._top2_union():
            cand = asr_runner.CANDIDATES_BY_ID[asr_id]
            cats = [c for c in analysis.CATEGORIES if asr_id in top2.get(c, []) and c in self.store.plan["datasets"]]
            for m in self.store.plan["tl_models"]:
                entry = self._entries.get(m["fp"])
                task_id = f"s3tog/{asr_id}/{m['fp']}"
                if self.store.task(task_id)["status"] in ("done", "failed"):
                    continue
                if entry is None or m["fp"] in failed_models:
                    self.store.append("together_meta", {"task": task_id, "asr": asr_id, "model": m["fp"],
                                                        "status": "failed", "reason": "翻譯模型無法使用"})
                    self.store.set_task(task_id, status="failed", reason="翻譯模型無法使用")
                    continue
                self._check_app()
                self._unload_llms()
                self._settle()
                self._run_together(cand, m, entry, cats, n_clips, task_id)
        self._unload_llms()
        self._settle()

    def _run_together(self, cand, m, entry, cats, n_clips, task_id):
        hint_missing = asr_runner.hint_missing_for(cand, self.forced)
        with gpu.VramSampler(self.vram) as sampler:
            session = self._open_session(cand, task_id)
            if session is None:
                self.store.append("together_meta", {"task": task_id, "asr": cand.id, "model": m["fp"],
                                                    "status": "failed", "reason": "ASR 載入失敗"})
                self.store.set_task(task_id, status="failed", reason="ASR 載入失敗")
                return
            try:
                self._load_llm(entry)
                translator = self._translator(entry)
                warm, _ = self._clips(cats[0]) if cats else ([], [])
                for clip in warm[:ASR_WARMUP]:
                    result, _ms = session.transcribe(self._audio(clip.wav))
                    tl_runner.translate_timed(translator, (result or {}).get("text") or "テスト")
                for cat in cats:
                    clips = self._clips(cat)[1][:n_clips]
                    done = self.store.done_keys("together", task_id)
                    if all(f"{cat}/{c.id}" in done for c in clips):
                        continue
                    for rec in self._replay(session, translator, clips, hint_missing):
                        key = f"{cat}/{rec['clip']}"
                        self.store.append("together", {"task": task_id, "key": key, "asr": cand.id,
                                                       "model": m["fp"], "category": cat, **rec})
            except Exception as e:  # this pair fails; others continue
                log.error(f"Together {cand.label} + {m['name']} failed: {e}")
                self.store.append("together_meta", {"task": task_id, "asr": cand.id, "model": m["fp"],
                                                    "status": "failed", "reason": str(e)[:300],
                                                    "vram_peak_mb": sampler.peak_mb})
                self.store.set_task(task_id, status="failed", reason=str(e)[:300])
                return
            finally:
                session.close()
        self.store.append("together_meta", {"task": task_id, "asr": cand.id, "model": m["fp"], "status": "ok",
                                            "vram_peak_mb": sampler.peak_mb, "vram_base_mb": self.baseline_mb})
        self.store.set_task(task_id, status="done")
        self.progress.finish(f"同時載入：{cand.label} + {m['name']} 完成")

    def _replay(self, session, translator, clips, hint_missing):
        """Real-time replay: clip i is handed over when it would have finished playing
        (+0.5 s for the VAD to notice the pause). ASR runs in order on this thread;
        translations run on 2 workers like the app. Returns one record per clip."""
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="bench-tl")
        t0 = time.perf_counter()
        release = []
        acc = 0.0
        for c in clips:
            acc += c.duration
            release.append(acc + TOGETHER_RELEASE_PAD_S)
        jobs = []
        try:
            for i, clip in enumerate(clips):
                if self.d.together_realtime:
                    wait = t0 + release[i] - time.perf_counter()
                    if wait > 0:
                        self.d.sleep(wait)
                released = t0 + release[i] if self.d.together_realtime else time.perf_counter()
                start = time.perf_counter()
                result, asr_ms = session.transcribe(self._audio(clip.wav))
                raw = (result or {}).get("text", "") if result else ""
                label = (result or {}).get("language", "") if result else ""
                shown, _reason = asr_runner.displayed_text(raw, label, clip.duration, self.forced, hint_missing)
                base = {"clip": clip.id, "queue_ms": max(0.0, (start - released) * 1000), "asr_ms": asr_ms,
                        "text": shown}
                if not shown:
                    jobs.append((base, None, released))
                    continue
                jobs.append((base, pool.submit(tl_runner.translate_timed, translator, shown), released))
                self.progress.show("階段 3/3 同時載入", f"第 {i + 1}/{len(clips)} 句", i + 1, len(clips), asr_ms,
                                   self._remaining_estimate())
            out = []
            for base, fut, released in jobs:
                if fut is None:
                    out.append({**base, "ok": False, "error": "asr_empty", "ttft_ms": None, "tl_ms": None, "e2e_ms": None})
                    continue
                res = fut.result()
                done_at = time.perf_counter()
                out.append({**base, "ok": res["ok"], "error": res["error"], "ttft_ms": res["ttft_ms"],
                            "tl_ms": res["total_ms"], "e2e_ms": (done_at - released) * 1000 if res["ok"] else None,
                            "translation": res["text"]})
            return out
        finally:
            pool.shutdown(wait=True)

    # ------------------------------------------------------------------ judges
    def stage_judges(self):
        plan = self.store.plan
        judges = plan.get("judges") or []
        candidates = [m["fp"] for m in plan["tl_models"]]
        assignment = judge_mod.assign_judges(candidates, judges)
        tl_records = [r for r in self.store.read("tl") if r.get("ok") and r.get("source")]
        for jfp in judges:
            entry = self._entries.get(jfp)
            task_id = f"s3judge/{jfp}"
            if self.store.task(task_id)["status"] in ("done", "failed"):
                continue
            if entry is None:
                self.store.set_task(task_id, status="failed", reason="設定檔裡已沒有這個評審模型")
                continue
            self._check_app()
            self._unload_llms()
            self._settle()
            try:
                self._load_llm(entry)
            except Exception as e:  # judge unavailable: its scores stay missing
                self.store.set_task(task_id, status="failed", reason=f"載入失敗：{e}")
                self.progress.finish(f"評審 {entry.get('name')} 無法載入：{e}")
                continue
            client = (self.d.judge_factory or judge_mod.JudgeClient)(entry)
            done = self.store.done_keys("judge", task_id)
            items = [(f"cal/{cid}", "anime", ja, tl, None, False) for cid, ja, tl, _exp in judge_mod.CALIBRATION]
            for r in tl_records:
                for j, is_self in assignment.get(r["model"], []):
                    if j == jfp:
                        items.append((f"{r['model']}/{r['category']}/{r['clip']}", r["category"], r["ref"],
                                      r["text"], r["model"], is_self))
            total = len(items)
            count = len(done)
            ewma = Ewma()
            for item, cat, ja, tl, cand_fp, is_self in items:
                if item in done:
                    continue
                t0 = time.perf_counter()
                scores = client.score(cat, ja, tl)
                self._record_timing("judge", ewma.add(time.perf_counter() - t0))
                self.store.append("judge", {"task": task_id, "key": item, "judge": jfp, "item": item,
                                            "candidate": cand_fp, "category": cat, "scores": scores, "self": is_self})
                count += 1
                self.progress.show(f"評審｜{entry.get('name')}", "翻譯評分", count, total,
                                   (ewma.value or 0) * 1000, self._remaining_estimate())
            if jfp == judges[0]:
                self._judge_vad(client, task_id)
            self.store.set_task(task_id, status="done", items_done=total)
            self.progress.finish(f"評審 {entry.get('name')} 完成")
        self._unload_llms()
        self._settle()

    def _judge_vad(self, client, task_id):
        raws = self.store.read("vad_raw")
        continuous = {(r["asr"], r["recording"]): r for r in raws if r.get("threshold") is None}
        done = {r["key"] for r in self.store.read("vad")}
        for r in raws:
            if r.get("threshold") is None:
                continue
            key = f"{r['asr']}/{r['recording']}/{r['threshold']}"
            if key in done:
                continue
            ref = continuous.get((r["asr"], r["recording"]))
            if not ref:
                continue
            hall = missed = judged = 0
            for w in ref["windows"]:
                lines = [x["text"] for x in vad_runner.lines_in_window(r["lines"], w["start"], w["end"])]
                result = client.vad(w["text"], lines)
                if result is None:
                    continue
                judged += 1
                hall += len(result["hallucinated"])
                missed += len(result["missed"])
            self.store.append("vad", {"task": task_id, "key": key, "asr": r["asr"], "recording": r["recording"],
                                      "category": r["category"], "threshold": r["threshold"], "minutes": r["minutes"],
                                      "segments": r["segments"], "speech_s": r["speech_s"], "hallucinated": hall,
                                      "missed": missed, "judged_windows": judged})


def _warmup_sentences():
    from benchmark import BENCH_SENTENCES

    return BENCH_SENTENCES["ja"][:tl_runner.WARMUP_SENTENCES]


# ---------------------------------------------------------------------- estimates

def estimate_remaining(store, timing: dict | None = None) -> float:
    """Seconds left for the unfinished part of a plan (stage 3 counts assume 2 ASRs per
    category until stage 1 has frozen the rankings)."""
    timing = timing or {}
    plan = store.plan
    done_tasks = {t["id"] for t in store.state.get("tasks", []) if t["status"] in ("done", "failed")}

    def per(key):
        return timing.get(key) or DEFAULT_SECONDS.get(key, DEFAULT_SECONDS["tl"])

    counts = plan.get("clip_counts", {})
    total = 0.0
    stages = plan.get("stages", [1, 2, 3])
    if 1 in stages and not plan.get("reuse_stage1_from"):
        n = sum(counts.get(ds, 0) for ds in plan.get("datasets", []))
        for asr in plan.get("asr", []):
            if f"s1/{asr}" not in done_tasks:
                done = len(store.done_keys("asr", f"s1/{asr}"))
                total += DEFAULT_SECONDS["load"] + max(0, n - done) * per(f"asr:{asr}")
    cats = [c for c in analysis.CATEGORIES if c in plan.get("datasets", [])]
    top2 = (store.state.get("frozen") or {}).get("top2")
    asr_union = len({a for c in cats for a in (top2 or {}).get(c, [])}) if top2 else min(3, 2 * len(cats))
    models = plan.get("tl_models", [])
    if 2 in stages and plan.get("recordings"):
        minutes = sum(r.get("minutes", 1.0) for r in plan["recordings"])
        total += asr_union * (DEFAULT_SECONDS["load"] + minutes * 60 * 0.5)
        total += minutes * 2 * len(vad_runner.THRESHOLDS) * per("judge")
    if 3 in stages:
        n_tl = sum(counts.get(c, 0) for c in cats)
        for m in models:
            if f"s3tl/{m['fp']}" not in done_tasks:
                total += DEFAULT_SECONDS["load"] + n_tl * per("tl")
        together_s = plan.get("together_clips", 20) * 6.0 + 60.0
        total += asr_union * len(models) * together_s
        judges = plan.get("judges", [])
        assignment = judge_mod.assign_judges([m["fp"] for m in models], judges)
        for j in judges:
            if f"s3judge/{j}" not in done_tasks:
                assigned = sum(1 for m in models if any(x == j for x, _ in assignment[m["fp"]]))
                total += DEFAULT_SECONDS["load"] + (len(judge_mod.CALIBRATION) + assigned * n_tl) * per("judge")
    return total


def describe_estimate(store) -> str:
    return fmt_duration(estimate_remaining(store, store.state.get("timing")))


def git_rev() -> str:
    try:
        from bench import paths

        out = subprocess.run(["git", "-C", str(paths.ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "?"
    except (OSError, subprocess.TimeoutExpired):
        return "?"


