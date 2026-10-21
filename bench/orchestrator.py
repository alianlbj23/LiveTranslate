"""Runs a benchmark plan stage by stage.

Order (fixed so a resume continues where it stopped):
  stage 1  s1/<asr>                each ASR loaded once, all datasets
           -> rankings frozen once (top 2 per category, best ASR)
  stage 2  s2asr/<asr>             top-2 ASRs on the user's recordings (VAD segments
                                   per threshold + continuous reference)
  stage 3  s3tl/<model>            each translation model alone on the best ASR's text
           s3tog/<asr>/<model>     ASR + model loaded together, real-time replay
           s3judge/<judge>         each judge alone: calibration and its assigned
                                   translations (cross judging)
           s2judge                 stage-2 windows, by the first judge that loads and
                                   has not failed its translation scoring
  report

Records are appended to results/<kind>.jsonl as they finish: per sentence in stage 1,
translation alone and judging; per category in the real-time replay; per (ASR,
recording, threshold) in stage 2 and its judging. A resume skips tasks already done or
failed (state.py has the rules); inside a task that runs again only items that are not
final (state.is_final) are redone, so failed sentences of a finished task stay failed.
A model that cannot be downloaded, built or loaded is recorded with the reason and
skipped; failures are logged to bench.log and shown on the console."""
import concurrent.futures
import ctypes
import logging
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from bench import analysis, asr_runner, datasets, gpu, judge as judge_mod, llm_servers, paths, report, tl_runner, vad_runner
from bench.progress import Ewma, ProgressLine, fmt_duration
from bench.state import RunStore

log = logging.getLogger("LiveTranslate.Bench")

FAIL_STREAK = 10          # this many failures in a row: the model / judge is broken, not slow
APP_CHECK_EVERY = 10      # items between checks that LiveTranslate was not opened meanwhile
TL_WORKERS = 8            # like the app's translation executor (Ollama itself runs 2 at a time)
MAX_WORKER_ERRORS = 3     # ASR errors in a row: the task stops, incomplete and unranked
TOGETHER_RELEASE_PAD_S = 0.5
DEFAULT_SECONDS = {"tl": 0.9, "judge": 2.0, "load": 30.0}  # per item before any measurement; ASRs: AsrCandidate


@dataclass
class Deps:
    """Injectable collaborators (tests pass fakes)."""
    asr_client_factory: Callable[[dict], Any] | None = None         # config -> ASRClient-like
    translator_factory: Callable[[dict, dict, dict], Any] | None = None  # (entry, settings, config) -> Translator-like
    judge_factory: Callable[[dict], Any] | None = None              # entry -> JudgeClient-like
    ollama_factory: Callable[[str], Any] | None = None              # root url -> Ollama-like
    vram: Any = None                                                # VramReader-like
    app_running: Callable[[], list] | None = None                   # () -> [pid]
    vad_factory: Callable[[float, dict], Any] | None = None         # (threshold, settings) -> VADProcessor-like
    load_audio: Callable[[Any], Any] | None = None                  # path -> float32 16 kHz
    sleep: Callable[[float], None] = time.sleep
    out: Any = None                                                 # text stream for the progress line
    is_cached: Callable[[Any], bool] | None = None                  # cand -> bool
    download: Callable[[Any], None] | None = None                   # cand -> None (raises)
    together_realtime: bool = True
    manage_ollama: bool = True                                      # start / restart Ollama processes


def _prevent_sleep(on: bool) -> bool:
    if sys.platform != "win32":
        return True
    try:
        es_continuous, es_system = 0x80000000, 0x00000001
        ok = ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | (es_system if on else 0)) != 0
    except (AttributeError, OSError) as e:
        log.warning(f"Could not change the sleep setting: {e}")
        return False
    if not ok:
        log.warning("SetThreadExecutionState failed; the computer may go to sleep during the run")
    return ok


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
        self._servers = {}
        self.baseline_mb = None
        self._timing = dict(store.state.get("timing") or {})
        self._eta = (0.0, None)

    # ------------------------------------------------------------------ helpers
    def _say(self, message: str, level=logging.INFO):
        log.log(level, message)
        self.progress.finish(message)

    def _check_app(self):
        """Pause while the user's LiveTranslate runs (GPU and Ollama would be shared)."""
        if not self.d.app_running:
            return
        warned = False
        while self.d.app_running():
            if not warned:
                self._say("偵測到 LiveTranslate 正在執行，評測暫停。請關閉它（系統匣按「結束」），會自動繼續…")
                warned = True
            self.d.sleep(5.0)

    def _settle(self, label: str = ""):
        result = gpu.wait_vram_settle(self.vram, self.baseline_mb, sleep=self.d.sleep)
        if result["status"] in ("timeout", "stable_high"):
            issues = self.store.state.setdefault("settle_issues", [])
            issues.append({"after": label, **result})
            self.store.save()
            log.warning(f"VRAM did not fall back after {label}: {result}")
        return result

    def _audio(self, path):
        return (self.d.load_audio or datasets.load_audio_file)(path)

    def _record_timing(self, key, seconds):
        prev = self._timing.get(key)
        self._timing[key] = seconds if prev is None else 0.3 * seconds + 0.7 * prev
        self.store.state["timing"] = self._timing

    def _remaining_estimate(self) -> float:
        """Recomputed at most every 5 s (it rereads the result logs)."""
        now = time.monotonic()
        if self._eta[1] is None or now - self._eta[0] > 5.0:
            self._eta = (now, estimate_remaining(self.store, self._timing))
        return self._eta[1]

    def _maybe_check_app(self, count: int):
        if count % APP_CHECK_EVERY == 0:
            self._check_app()

    def _ollama_at(self, root: str):
        """One client per server for the whole run (each holds an HTTP connection pool)."""
        if root not in self._servers:
            self._servers[root] = (self.d.ollama_factory or llm_servers.Ollama)(root)
        return self._servers[root]

    def _ollama(self, entry):
        import local_models

        return self._ollama_at(local_models.ollama_root(entry["api_base"]))

    def _unload_llms(self) -> bool:
        """Unload every local model (Ollama: all, LM Studio: lms unload --all). Failures
        are logged; the next measurement waits for VRAM anyway."""
        import local_models

        roots = set()
        kinds = set()
        for entry in self._entries.values():
            kind = llm_servers.model_kind(entry)
            kinds.add(kind)
            if kind == "ollama":
                roots.add(local_models.ollama_root(entry["api_base"]))
        ok = "lmstudio" not in kinds or llm_servers.lmstudio_unload_all()
        for root in roots:
            try:
                server = self._ollama_at(root)
                if server.alive() and not server.unload_all(sleep=self.d.sleep):
                    ok = False
            except Exception as e:  # httpx / JSON errors from a busy server: log and go on
                ok = False
                log.warning(f"Could not unload the models of {root}: {e}")
        return ok

    def _clear_gpu(self, label: str):
        """Unload every local LLM and wait for VRAM to fall back."""
        self._unload_llms()
        return self._settle(label)

    def _ensure_ollama(self, entry) -> bool:
        """Ollama answering with the launcher's fast settings. A tray-started Ollama
        (no fast settings) is restarted when the user agreed before the run."""
        server = self._ollama(entry)
        alive = server.alive()
        restart = alive and self.store.plan["restart_ollama"] and not self._ollama_restarted
        if alive and not restart:
            return True
        if not self.d.manage_ollama:
            return alive
        from urllib.parse import urlsplit

        port = urlsplit(entry["api_base"]).port or 11434
        if restart:
            llm_servers.stop_tray_ollama()
            self._ollama_restarted = True
        self._ollama_started_pid = llm_servers.start_fast_ollama(port, self.store.dir)
        self.store.freeze("ollama_started_by_bench", self._ollama_started_pid)
        return True

    def _load_llm(self, entry) -> float:
        """Load a local model alone; returns load seconds (0 for cloud / LM Studio).
        Raises with a readable reason."""
        if llm_servers.model_kind(entry) != "ollama":
            return 0.0
        if not self._ensure_ollama(entry):
            raise RuntimeError("Ollama 沒有回應")
        server = self._ollama(entry)
        if entry["model"] not in server.installed():
            raise RuntimeError(f"Ollama 裡沒有 {entry['model']}（請先在啟動視窗下載）")
        return server.load(entry["model"])

    def _fail_task(self, task_id, reason, label=None):
        self.store.set_task(task_id, status="failed", reason=str(reason)[:300])
        self._say(f"略過 {label or task_id}：{reason}", logging.ERROR)

    # ------------------------------------------------------------------ run
    def run(self):
        if not _prevent_sleep(True):
            self.progress.finish("注意：無法阻止電腦睡眠，請在評測期間保持電腦開著。")
        try:
            self._check_app()
            self._unload_llms()
            self.baseline_mb = self.vram.used_mb()
            self.store.freeze("vram_baseline_mb", self.baseline_mb)
            if self.baseline_mb is None:
                self._say("讀不到顯示卡的記憶體用量（NVML），顯示記憶體欄位會空白，11 GB 的檢查也不會做。", logging.WARNING)
            elif self.baseline_mb > 1536:
                self.progress.finish(f"注意：開始前顯示卡已經用了 {self.baseline_mb / 1024:.1f} GB，"
                                     "其他程式（瀏覽器、動態桌布）會影響顯示記憶體的量測。")
            self._check_clips()
            plan = self.store.plan
            stages = plan["stages"]
            if plan["reuse_stage1_from"]:
                self._import_stage1()
            elif 1 in stages:
                self.stage1()
            self._freeze_rankings()
            recordings = plan["recordings"]
            if 2 in stages and recordings:
                self.stage2_asr()
            if 3 in stages:
                self.stage3_translate()
                self.stage3_together()
            if 3 in stages or (2 in stages and recordings):
                self.stage_judges()
            self.store.set_status("done")
            report.build(self.store)
            self.progress.finish("評測完成。")
        except KeyboardInterrupt:
            self.store.set_status("interrupted")
            self.progress.finish("已中斷。之後在選單選「3) 從上次中斷的地方繼續」。")
            raise
        except Exception as e:
            log.exception("Benchmark stopped by an error")
            self.store.set_status("error", f"{type(e).__name__}: {e}")
            raise
        finally:
            self._cleanup()

    def _cleanup(self):
        """Each step on its own: one failing must not skip the others."""
        try:
            self._unload_llms()
        except Exception as e:  # cleanup must not hide the real outcome
            log.warning(f"Unloading models at the end failed: {e}")
        if self._ollama_started_pid:
            try:
                llm_servers.stop_ollama(self._ollama_started_pid)
            except Exception as e:  # the watchdog stops it when this process ends
                log.warning(f"Could not stop the Ollama started for the benchmark: {e}")
        _prevent_sleep(False)
        try:
            self.store.save()
        except OSError as e:
            log.error(f"Could not save the run state: {e}")

    def _check_clips(self):
        """Stop before measuring when test clips of the plan are gone: with one missing
        every ASR would be unranked ("199/200"), with an unreadable manifest every task
        would finish with 0 sentences."""
        for ds in self.store.plan["datasets"]:
            want = self.store.plan["clip_counts"].get(ds, 0)
            have = len(self._clips(ds)[1])
            if have < want:
                raise RuntimeError(f"「{datasets.DATASETS[ds]['label']}」只找到 {have}/{want} 句測試資料"
                                   "（音檔被刪掉或 manifest.json 讀不到，缺哪些檔見 bench.log）。"
                                   "請先在選單選 1 補下載，再開始新的評測")

    # ------------------------------------------------------------------ stage 1
    def _import_stage1(self):
        if self.store.read("asr"):
            return
        src = RunStore(self.store.dir.parent / self.store.plan["reuse_stage1_from"])
        for kind in ("asr", "asr_meta"):
            for r in src.read(kind):
                self.store.append(kind, r)
        self.store.set_task("s1/imported", status="done", reason=f"沿用 {src.dir.name}")

    def _freeze_rankings(self):
        if self.store.frozen("rankings") is not None:
            return
        rankings = analysis.asr_rankings(self.store)
        self.store.freeze("rankings", rankings)
        self.store.freeze("top2", {cat: ids[:2] for cat, ids in rankings.items()})
        for cat in analysis.CATEGORIES:
            if cat in self.store.plan["datasets"] and not rankings.get(cat):
                self._say(f"沒有任何辨識模型完成「{analysis.CATEGORY_LABELS[cat]}」的測試，"
                          "這一類不會有翻譯結果和推薦（原因見上面的訊息與 bench.log）。", logging.ERROR)

    def stage1(self):
        for asr_id in self.store.plan["asr"]:
            cand = asr_runner.CANDIDATES_BY_ID[asr_id]
            task_id = f"s1/{asr_id}"
            if self.store.finished(task_id):
                continue
            self._check_app()
            self._run_asr_task(cand, task_id)

    def _clips(self, dataset):
        return datasets.load_clips(dataset, limit=self.store.plan["clip_counts"].get(dataset))

    def _ensure_asr_model(self, cand, task_id) -> bool:
        hub = self.settings.get("hub", "ms")
        if self.d.is_cached is not None:
            cached = self.d.is_cached(cand)
        else:
            cached = asr_runner.is_cached(cand, hub)
        if cached:
            return True
        if not self.store.plan["download_missing"]:
            self._asr_failed(cand, task_id, "模型尚未下載（開始前選擇不下載）")
            return False
        self.progress.finish(f"下載 {cand.label}…")
        try:
            if self.d.download is not None:
                self.d.download(cand)
            else:
                asr_runner.download(cand, hub, self.settings.get("download_proxy", "system"))
            return True
        except Exception as e:  # any download error: record and skip this model
            self._asr_failed(cand, task_id, f"下載失敗：{e}")
            return False

    def _asr_failed(self, cand, task_id, reason):
        if task_id.startswith("s1/"):  # stage-1 meta only: later stages must not override it
            self.store.append("asr_meta", {"task": task_id, "asr": cand.id, "status": "failed",
                                           "reason": str(reason)[:300]})
        self._fail_task(task_id, reason, cand.label)

    def _open_session(self, cand, task_id):
        """Loaded AsrSession, or None after a second failed load (recorded)."""
        try:
            config = asr_runner.build_worker_config(cand, self.settings, self.config)
        except Exception as e:  # e.g. the Whisper turbo folder is gone
            self._asr_failed(cand, task_id, f"設定錯誤：{e}")
            return None
        for attempt in (1, 2):
            session = asr_runner.AsrSession(cand, config, client_factory=self.d.asr_client_factory)
            try:
                session.load()
                return session
            except Exception as e:  # load errors are model-specific: retry once, then skip
                log.warning(f"{cand.label} failed to load (attempt {attempt}): {e}")
                self._settle(f"{cand.label} load failure")
                if attempt == 2:
                    self._asr_failed(cand, task_id, f"載入失敗：{e}")
        return None

    def _run_asr_task(self, cand, task_id):
        from asr_client import ASRWorkerError

        plan = self.store.plan
        if not self._ensure_asr_model(cand, task_id):
            return
        self._settle("start")
        hint_missing = asr_runner.hint_missing_for(cand, self.forced)
        done = self.store.final_keys("asr", task_id)
        clips = [(ds, clip) for ds in plan["datasets"] for clip in self._clips(ds)[1]]
        total = len(clips)
        self.store.set_task(task_id, status="running", items_total=total,
                            attempts=self.store.task(task_id)["attempts"] + 1)
        ewma = Ewma()
        errors = streak = 0
        session = None
        with gpu.VramSampler(self.vram) as sampler:
            try:
                session = self._open_session(cand, task_id)
                if session is None:
                    return
                warm, _ = self._clips(plan["datasets"][0])
                for clip in warm[:datasets.WARMUP_CLIPS]:
                    session.transcribe(self._audio(clip.wav))
                count = len(done)
                for ds, clip in clips:
                    key = clip.key
                    if key in done:
                        continue
                    base = {"task": task_id, "key": key, "asr": cand.id, "dataset": ds, "clip": clip.id,
                            "ref": clip.ref, "duration": clip.duration}
                    try:
                        audio = self._audio(clip.wav)
                    except Exception as e:  # unreadable clip: a data problem, not the ASR's
                        log.warning(f"Cannot read {clip.wav}: {e}")
                        self.store.append("asr", {**base, "raw": "", "display": "", "error": f"audio: {e}",
                                                  "skip": True, "ms": None})
                        continue
                    try:
                        result, ms = session.transcribe(audio)
                    except Exception as e:  # this sentence failed: it counts as all wrong
                        errors += 1
                        streak += 1
                        alive = isinstance(e, ASRWorkerError) and e.recoverable
                        log.error(f"{cand.label} {'failed' if alive else 'worker died'} on {key}: {e}")
                        self.store.append("asr", {**base, "raw": "", "display": "",
                                                  "error": (str(e) if alive else f"worker: {e}")[:200], "ms": None})
                        if streak >= MAX_WORKER_ERRORS:
                            raise
                        if not alive:  # start a fresh worker for the next sentence
                            session.close()
                            session = self._open_session(cand, task_id)
                            if session is None:
                                return
                        continue
                    streak = 0
                    raw, label = asr_runner.result_text(result)
                    shown, reason = asr_runner.displayed_text(raw, label, clip.duration, self.forced, hint_missing)
                    self.store.append("asr", {**base, "raw": raw, "label": label, "display": shown,
                                              "reason": reason, "ms": ms})
                    count += 1
                    self._maybe_check_app(count)
                    avg = ewma.add(ms)
                    self._record_timing(f"asr:{cand.id}", avg / 1000)
                    self.progress.show(f"階段 1/3 辨識｜{cand.label}", ds, count, total, avg,
                                       self._remaining_estimate())
            except Exception as e:  # incomplete: keep what was measured, do not rank it
                log.exception(f"{cand.label} stopped")
                self.store.append("asr_meta", {"task": task_id, "asr": cand.id, "status": "incomplete",
                                               "reason": str(e)[:300], "load_s": session.load_s if session else None,
                                               "vram_peak_mb": sampler.peak_mb, "vram_base_mb": self.baseline_mb})
                self._fail_task(task_id, f"結果不完整：{e}", cand.label)
                return
            finally:
                if session:
                    session.close()
        self.store.append("asr_meta", {"task": task_id, "asr": cand.id, "status": "ok", "load_s": session.load_s,
                                       "restarts": session.restarts, "errors": errors,
                                       "vram_peak_mb": sampler.peak_mb, "vram_base_mb": self.baseline_mb})
        self.store.set_task(task_id, status="done", items_done=total)
        self.progress.finish(f"{cand.label} 完成" + (f"（{errors} 句辨識出錯）" if errors else ""))
        self._settle(cand.label)

    # ------------------------------------------------------------------ stage 2
    def _top2(self) -> dict:
        return self.store.frozen("top2") or {}

    def _top2_union(self, categories=None):
        top2 = self._top2()
        out = []
        for cat in (categories or analysis.CATEGORIES):
            for a in top2.get(cat, []):
                if a not in out:
                    out.append(a)
        return out

    def stage2_asr(self):
        recordings = self.store.plan["recordings"]
        asr_ids = self._top2_union(sorted({r["category"] for r in recordings}))
        if not asr_ids:
            self._say("沒有可用的辨識模型，第 2 階段（語音偵測閾值）略過。", logging.ERROR)
            return
        top2 = self._top2()
        for asr_id in asr_ids:
            cand = asr_runner.CANDIDATES_BY_ID[asr_id]
            task_id = f"s2asr/{asr_id}"
            if self.store.finished(task_id):
                continue
            self._check_app()
            self._settle("stage 2")
            done = self.store.final_keys("vad_raw", task_id)
            hint_missing = asr_runner.hint_missing_for(cand, self.forced)
            session = self._open_session(cand, task_id)
            if session is None:
                continue
            failures = []
            try:
                for rec_no, rec in enumerate(recordings, 1):
                    if asr_id not in top2.get(rec["category"], []):
                        continue
                    try:
                        if session.client is None:  # the worker died on an earlier recording
                            session.load()
                        self._stage2_recording(session, cand, rec, rec_no, len(recordings), task_id, done, hint_missing)
                    except Exception as e:  # one recording failing does not stop the others
                        log.exception(f"Stage 2: {rec['path']} with {cand.label} failed")
                        failures.append(f"{rec['path']}: {e}")
            finally:
                session.close()
            if failures:
                self._fail_task(task_id, "；".join(failures)[:300], f"{cand.label} 第 2 階段")
            else:
                self.store.set_task(task_id, status="done")
        self._settle("stage 2")

    def _stage2_recording(self, session, cand, rec, rec_no, n_recs, task_id, done, hint_missing):
        audio = self._audio(rec["path"])
        minutes = len(audio) / vad_runner.SR / 60
        base = {"task": task_id, "asr": cand.id, "recording": rec["path"], "category": rec["category"],
                "minutes": minutes}
        key = f"{rec['path']}/continuous"
        if key not in done:
            windows = []
            for s, e in vad_runner.continuous_windows(audio):
                result, _ms = session.transcribe(audio[s:e])
                windows.append({"start": s / vad_runner.SR, "end": e / vad_runner.SR,
                                "text": asr_runner.result_text(result)[0]})
            self.store.append("vad_raw", {**base, "key": key, "threshold": None, "windows": windows})
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
                raw, label = asr_runner.result_text(result)
                shown, _reason = asr_runner.displayed_text(raw, label, seg["end"] - seg["start"], self.forced, hint_missing)
                if shown:
                    lines.append({"start": seg["start"], "end": seg["end"], "text": shown})
            self.store.append("vad_raw", {**base, "key": key, "threshold": th, "lines": lines,
                                          "segments": len(lines), "speech_s": speech_s})
            self.progress.show("階段 2/3 語音偵測", f"{cand.label}｜閾值 {th}｜錄音", rec_no, n_recs, None,
                               self._remaining_estimate(), force=True)

    # ------------------------------------------------------------------ stage 3
    def _tl_inputs(self):
        """{category: [(clip, displayed text of the best ASR)]}. A clip without a record of
        that ASR (stage 1 reused from a run on other clips) is logged and left out."""
        rankings = self.store.frozen("rankings") or {}
        records = {}
        for r in self.store.read("asr"):
            if not r.get("skip"):
                records[(r["asr"], r["dataset"], r["clip"])] = r
        out = {}
        for cat in analysis.CATEGORIES:
            if cat not in self.store.plan["datasets"] or not rankings.get(cat):
                continue
            best = rankings[cat][0]
            clips = self._clips(cat)[1]
            rows = [(clip, records[(best, cat, clip.id)].get("display") or "")
                    for clip in clips if (best, cat, clip.id) in records]
            if len(rows) < len(clips):
                log.warning(f"{cat}: {len(clips) - len(rows)} of {len(clips)} test sentences have no {best} "
                            "result in stage 1; they are not translated")
            if rows:
                out[cat] = rows
        return out

    def _translator(self, entry):
        return (self.d.translator_factory or tl_runner.build_translator)(entry, self.settings, self.config)

    def stage3_translate(self):
        inputs = self._tl_inputs()
        no_input = None
        if not inputs:
            ranked = any((self.store.frozen("rankings") or {}).values())
            no_input = ("沒有可用的辨識結果（第 1 階段沒有任何辨識模型完成）" if not ranked
                        else "第 1 階段的辨識結果和目前的測試句子對不上（測試資料重新抽樣過？請勾選階段 1 重跑）")
        for m in self.store.plan["tl_models"]:
            task_id = f"s3tl/{m['fp']}"
            if self.store.finished(task_id):
                continue
            entry = self._entries.get(m["fp"])
            if entry is None:
                self._tl_failed(m, task_id, "設定檔裡已沒有這個模型")
                continue
            if no_input:
                self._tl_failed(m, task_id, no_input)
                continue
            self._check_app()
            settle = self._clear_gpu(m["name"])
            self._translate_task(m, entry, inputs, task_id, settle)
        self._clear_gpu("stage 3 translation")

    def _translate_task(self, m, entry, inputs, task_id, settle):
        done = self.store.final_keys("tl", task_id)
        todo = [(cat, clip, source) for cat, rows in inputs.items() for clip, source in rows
                if clip.key not in done]
        total = sum(len(v) for v in inputs.values())
        errors = {}
        broken = None
        with gpu.VramSampler(self.vram) as sampler:
            try:
                load_s = self._load_llm(entry)
                translator = self._translator(entry)
                for text in _warmup_sentences():
                    tl_runner.translate_timed(translator, text)
            except Exception as e:  # missing / unloadable / misconfigured model: skip it
                log.exception(f"{m['name']} could not start")
                self._tl_failed(m, task_id, f"無法開始：{e}")
                return
            count = len(done)
            streak = 0
            ewma = Ewma()
            for cat, clip, source in todo:
                key = clip.key
                if not source:
                    rec = {"ok": False, "text": "", "ttft_ms": None, "total_ms": None, "error": asr_runner.ASR_EMPTY}
                else:
                    rec = tl_runner.translate_timed(translator, source)
                    if rec["ok"]:
                        streak = 0
                        self._record_timing("tl", ewma.add(rec["total_ms"]) / 1000)
                    else:
                        streak += 1
                        errors[rec["error"]] = errors.get(rec["error"], 0) + 1
                        log.warning(f"{m['name']} failed on {key}: {rec['error']} {rec.get('detail', '')}")
                self.store.append("tl", {"task": task_id, "key": key, "model": m["fp"], "category": cat,
                                         "clip": clip.id, "ref": clip.ref, "source": source, **rec})
                count += 1
                self._maybe_check_app(count)
                self.progress.show(f"階段 3/3 翻譯｜{m['name']}", cat, count, total, ewma.value,
                                   self._remaining_estimate())
                if streak >= FAIL_STREAK:
                    broken = f"連續 {FAIL_STREAK} 句失敗（{rec['error']} {rec.get('detail') or ''}".strip() + "）"
                    break
        meta = {"task": task_id, "model": m["fp"], "load_s": load_s, "vram_peak_mb": sampler.peak_mb,
                "errors": errors, "vram_settle": settle.get("status")}
        if broken:
            self.store.append("tl_meta", {**meta, "status": "failed", "reason": broken})
            self._fail_task(task_id, broken, m["name"])
            return
        self.store.append("tl_meta", {**meta, "status": "ok"})
        self.store.set_task(task_id, status="done", items_done=total)
        summary = "、".join(f"{k} {v} 句" for k, v in errors.items())
        self._say(f"{m['name']} 翻譯完成" + (f"（失敗：{summary}）" if summary else ""))

    def _tl_failed(self, m, task_id, reason):
        self.store.append("tl_meta", {"task": task_id, "model": m["fp"], "status": "failed", "reason": reason,
                                      "vram_peak_mb": None})
        self._fail_task(task_id, reason, m["name"])

    def stage3_together(self):
        top2 = self._top2()
        n_clips = self.store.plan["together_clips"]
        failed_models = {t["id"].split("/", 1)[1] for t in self.store.state["tasks"]
                         if t["id"].startswith("s3tl/") and t["status"] == "failed"}
        for asr_id in self._top2_union():
            cand = asr_runner.CANDIDATES_BY_ID[asr_id]
            cats = [c for c in analysis.CATEGORIES if asr_id in top2.get(c, []) and c in self.store.plan["datasets"]]
            for m in self.store.plan["tl_models"]:
                entry = self._entries.get(m["fp"])
                task_id = f"s3tog/{asr_id}/{m['fp']}"
                if self.store.finished(task_id):
                    continue
                if entry is None or m["fp"] in failed_models:
                    self.store.append("together_meta", {"task": task_id, "asr": asr_id, "model": m["fp"],
                                                        "status": "failed", "reason": "翻譯模型無法使用"})
                    self.store.set_task(task_id, status="failed", reason="翻譯模型無法使用")
                    continue
                self._check_app()
                self._clear_gpu(f"{cand.label} + {m['name']}")
                self._run_together(cand, m, entry, cats, n_clips, task_id)
        self._clear_gpu("stage 3 together")

    def _run_together(self, cand, m, entry, cats, n_clips, task_id):
        label = f"{cand.label} + {m['name']}"
        done = self.store.final_keys("together", task_id)
        # a category is replayed as a whole (its records are written together at its end)
        todo = {cat: [c for c in self._clips(cat)[1][:n_clips] if c.key not in done] for cat in cats}
        todo = {cat: clips for cat, clips in todo.items() if clips}
        hint_missing = asr_runner.hint_missing_for(cand, self.forced)
        errors = {}
        session = None
        with gpu.VramSampler(self.vram) as sampler:
            try:
                if todo:
                    session = self._open_session(cand, task_id)
                    if session is None:
                        self.store.append("together_meta", {"task": task_id, "asr": cand.id, "model": m["fp"],
                                                            "status": "failed", "reason": "ASR 載入失敗"})
                        return
                    self._load_llm(entry)
                    translator = self._translator(entry)
                    self._warm_up_together(session, translator, cats)
                for cat, clips in todo.items():
                    for clip, rec in zip(clips, self._replay(session, translator, clips, hint_missing)):
                        key = clip.key
                        if not rec["ok"] and rec["error"] != asr_runner.ASR_EMPTY:
                            errors[rec["error"]] = errors.get(rec["error"], 0) + 1
                            log.warning(f"Together {label} failed on {key}: {rec['error']} {rec.get('detail') or ''}")
                        self.store.append("together", {"task": task_id, "key": key, "asr": cand.id,
                                                       "model": m["fp"], "category": cat, **rec})
            except Exception as e:  # this pair fails; others continue
                log.exception(f"Together {label} failed")
                self.store.append("together_meta", {"task": task_id, "asr": cand.id, "model": m["fp"],
                                                    "status": "failed", "reason": str(e)[:300],
                                                    "vram_peak_mb": sampler.peak_mb})
                self._fail_task(task_id, e, label)
                return
            finally:
                if session:
                    session.close()
        self.store.append("together_meta", {"task": task_id, "asr": cand.id, "model": m["fp"], "status": "ok",
                                            "errors": errors, "vram_peak_mb": sampler.peak_mb,
                                            "vram_base_mb": self.baseline_mb})
        self.store.set_task(task_id, status="done")
        summary = "、".join(f"{k} {v} 句" for k, v in errors.items())
        self._say(f"同時載入：{label} 完成" + (f"（失敗：{summary}）" if summary else ""))

    def _warm_up_together(self, session, translator, cats):
        """Warm both models up; raises when every warm-up translation failed (typically
        not enough memory with both loaded), instead of replaying into certain failures."""
        asr_runner.warm_up_filters(self.forced)
        warm, _ = self._clips(cats[0]) if cats else ([], [])
        results = []
        for clip in warm[:datasets.WARMUP_CLIPS]:
            result, _ms = session.transcribe(self._audio(clip.wav))
            results.append(tl_runner.translate_timed(translator, asr_runner.result_text(result)[0] or "テスト"))
        if results and not any(r["ok"] for r in results):
            last = results[-1]
            raise RuntimeError(f"暖機翻譯全部失敗：{last['error']} {last.get('detail') or ''}".strip())

    def _replay(self, session, translator, clips, hint_missing):
        """Real-time replay: clip i is handed over when it would have finished playing
        (+0.5 s for the VAD to notice the pause). ASR runs in order on this thread;
        translations run on a pool like the app's executor, and each one's end-to-end
        time is taken on its worker thread when it finishes. One record per clip."""
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=TL_WORKERS, thread_name_prefix="bench-tl")
        t0 = time.perf_counter()
        release = []
        acc = 0.0
        for c in clips:
            acc += c.duration
            release.append(acc + TOGETHER_RELEASE_PAD_S)
        jobs = []

        def timed(text, released):
            res = tl_runner.translate_timed(translator, text)
            return res, time.perf_counter() - released

        try:
            for i, clip in enumerate(clips):
                if self.d.together_realtime:
                    released = t0 + release[i]
                    wait = released - time.perf_counter()
                    if wait > 0:
                        self.d.sleep(wait)
                else:
                    released = time.perf_counter()
                start = time.perf_counter()
                result, asr_ms = session.transcribe(self._audio(clip.wav))
                raw, label = asr_runner.result_text(result)
                shown, _reason = asr_runner.displayed_text(raw, label, clip.duration, self.forced, hint_missing)
                base = {"clip": clip.id, "queue_ms": max(0.0, (start - released) * 1000), "asr_ms": asr_ms,
                        "source": shown}
                jobs.append((base, pool.submit(timed, shown, released) if shown else None))
                # no time-remaining estimate here: rereading the logs would delay the next ASR
                self.progress.show("階段 3/3 同時載入", f"第 {i + 1}/{len(clips)} 句", i + 1, len(clips), asr_ms, None)
            out = []
            for base, fut in jobs:
                if fut is None:
                    out.append({**base, "ok": False, "error": asr_runner.ASR_EMPTY, "ttft_ms": None, "tl_ms": None,
                                "e2e_ms": None})
                    continue
                res, e2e_s = fut.result()
                out.append({**base, "ok": res["ok"], "error": res["error"], "detail": res.get("detail"),
                            "ttft_ms": res["ttft_ms"], "tl_ms": res["total_ms"], "e2e_ms": e2e_s * 1000,
                            "translation": res["text"]})
            return out
        finally:
            pool.shutdown(wait=True)

    # ------------------------------------------------------------------ judges
    def stage_judges(self):
        plan = self.store.plan
        judges = plan["judges"]
        candidates = [m["fp"] for m in plan["tl_models"]]
        assignment = judge_mod.assign_judges(candidates, judges)
        vad_pending = bool(plan["recordings"]) and 2 in plan["stages"] and not self.store.finished("s2judge")
        for jfp in judges:
            entry = self._entries.get(jfp)
            task_id = f"s3judge/{jfp}"
            scored = self.store.finished(task_id)
            if scored and not vad_pending:
                continue
            if entry is None:
                self._fail_task(task_id, "設定檔裡已沒有這個評審模型")
                continue
            self._check_app()
            self._clear_gpu(f"judge {entry.get('name')}")
            try:
                self._load_llm(entry)
                client = (self.d.judge_factory or judge_mod.JudgeClient)(entry)
            except Exception as e:  # judge unavailable: its scores stay missing (reported as low coverage)
                log.exception(f"Judge {entry.get('name')} could not start")
                self._fail_task(task_id, f"無法開始：{e}", f"評審 {entry.get('name')}")
                continue
            if not scored:
                self._judge_translations(client, entry, jfp, assignment, task_id)
            if vad_pending and self.store.task(task_id)["status"] != "failed":
                vad_pending = not self._judge_vad(client, entry)  # stopped: the next judge takes over
        if vad_pending and self.store.task("s2judge")["status"] != "failed":
            self._fail_task("s2judge", "沒有可用的評審，第 2 階段的結果沒有評分", "語音偵測評分")
        self._clear_gpu("judges")

    def _judge_translations(self, client, entry, jfp, assignment, task_id):
        tl_records = {}
        for r in self.store.read("tl"):
            tl_records[(r["model"], r["category"], r["clip"])] = r
        done = self.store.final_keys("judge", task_id)
        items = [(judge_mod.cal_key(cid), "anime", ja, tl, None, False) for cid, ja, tl, _exp in judge_mod.CALIBRATION]
        for r in tl_records.values():
            if not (r.get("ok") and r.get("source")):
                continue
            for j, is_self in assignment.get(r["model"], []):
                if j == jfp:
                    items.append((judge_mod.item_key(r["model"], r["category"], r["clip"]), r["category"], r["ref"],
                                  r["text"], r["model"], is_self))
        total = len(items)
        count = len(done)
        ewma = Ewma()
        streak = cal_fail = 0
        for item, cat, ja, tl, cand_fp, is_self in items:
            if item in done:
                continue
            t0 = time.perf_counter()
            scores = client.score(cat, ja, tl)
            self._record_timing("judge", ewma.add(time.perf_counter() - t0))
            self.store.append("judge", {"task": task_id, "key": item, "judge": jfp, "item": item,
                                        "candidate": cand_fp, "category": cat, "scores": scores, "self": is_self})
            count += 1
            self._maybe_check_app(count)
            self.progress.show(f"評審｜{entry.get('name')}", "翻譯評分", count, total,
                               (ewma.value or 0) * 1000, self._remaining_estimate())
            if scores is None:
                streak += 1
                cal_fail += item.startswith(judge_mod.CAL_PREFIX)
            else:
                streak = 0
            why = None
            if cal_fail == len(judge_mod.CALIBRATION):
                why = "校準題全部無法評分"
            elif streak >= FAIL_STREAK:
                why = f"連續 {FAIL_STREAK} 題無法評分"
            if why:
                self._fail_task(task_id, f"{why}（{_judge_error(client)}）", f"評審 {entry.get('name')}")
                return
        self.store.set_task(task_id, status="done", items_done=total)
        self._say(f"評審 {entry.get('name')} 完成")

    def _judge_vad(self, client, entry) -> bool:
        """Judge every (ASR, recording, threshold) not fully judged yet. False when the
        judge failed FAIL_STREAK windows in a row (s2judge is marked failed)."""
        raws = self.store.read("vad_raw")
        continuous = {(r["asr"], r["recording"]): r for r in raws if r.get("threshold") is None}
        latest = {r["key"]: r for r in self.store.read("vad")}
        done = {k for k, r in latest.items() if r.get("judged_windows") == r.get("windows")}
        todo = [r for r in raws if r.get("threshold") is not None]
        streak = 0
        for n, r in enumerate(todo, 1):
            key = f"{r['asr']}/{r['recording']}/{r['threshold']}"
            if key in done:
                continue
            ref = continuous.get((r["asr"], r["recording"]))
            if not ref:
                log.error(f"No continuous reference for {key}; cannot judge it")
                continue
            hall = missed = judged = 0
            judged_minutes = 0.0
            for w in ref["windows"]:
                lines = [x["text"] for x in vad_runner.lines_in_window(r["lines"], w["start"], w["end"])]
                result = client.vad(w["text"], lines)
                if result is None:
                    streak += 1
                    if streak >= FAIL_STREAK:
                        self._fail_task("s2judge", f"連續 {FAIL_STREAK} 段無法評分（{_judge_error(client)}）",
                                        f"評審 {entry.get('name')} 的語音偵測評分")
                        return False
                    continue
                streak = 0
                judged += 1
                judged_minutes += (w["end"] - w["start"]) / 60
                hall += len(result["hallucinated"])
                missed += len(result["missed"])
            self.store.append("vad", {"task": "s2judge", "key": key, "asr": r["asr"], "recording": r["recording"],
                                      "category": r["category"], "threshold": r["threshold"], "minutes": r["minutes"],
                                      "judged_minutes": judged_minutes, "windows": len(ref["windows"]),
                                      "segments": r["segments"], "speech_s": r["speech_s"], "hallucinated": hall,
                                      "missed": missed, "judged_windows": judged})
            self.progress.show(f"評審｜{entry.get('name')}", "語音偵測", n, len(todo), None, self._remaining_estimate())
        self.store.set_task("s2judge", status="done", reason=None)
        return True


def _judge_error(client) -> str:
    """The judge's last error, for task reasons (the console shows each one as it happens)."""
    return getattr(client, "last_error", None) or "回答不是規定的 JSON，或模型沒有回應"


def _warmup_sentences():
    from benchmark import BENCH_SENTENCES

    return BENCH_SENTENCES["ja"][:tl_runner.WARMUP_SENTENCES]


# ---------------------------------------------------------------------- estimates

def estimate_remaining(store, timing: dict | None = None) -> float:
    """Seconds left for the unfinished part of a plan. Until stage 1 has frozen the
    rankings, later stages assume 2 ASRs per category and at most 3 in total."""
    from bench.state import FINISHED

    timing = timing or {}
    plan = store.plan
    done_tasks = {t["id"] for t in store.state.get("tasks", []) if t["status"] in FINISHED}

    def per(key):
        if timing.get(key):
            return timing[key]
        if key.startswith("asr:"):
            cand = asr_runner.CANDIDATES_BY_ID.get(key[4:])
            return cand.default_item_s if cand else DEFAULT_SECONDS["tl"]
        return DEFAULT_SECONDS[key]

    counts = plan["clip_counts"]
    total = 0.0
    stages = plan["stages"]
    if 1 in stages and not plan["reuse_stage1_from"]:
        n = sum(counts.get(ds, 0) for ds in plan["datasets"])
        for asr in plan["asr"]:
            if f"s1/{asr}" not in done_tasks:
                done = len(store.final_keys("asr", f"s1/{asr}"))
                total += DEFAULT_SECONDS["load"] + max(0, n - done) * per(f"asr:{asr}")
    cats = [c for c in analysis.CATEGORIES if c in plan["datasets"]]
    top2 = (store.state.get("frozen") or {}).get("top2")
    asr_union = len({a for c in cats for a in (top2 or {}).get(c, [])}) if top2 else min(3, 2 * len(cats))
    models = plan["tl_models"]
    if 2 in stages and plan["recordings"]:
        minutes = sum(r.get("minutes", 1.0) for r in plan["recordings"])
        total += asr_union * (DEFAULT_SECONDS["load"] + minutes * 60 * 0.5)
        total += vad_runner.estimate_windows(minutes, asr_union) * per("judge")
    if 3 in stages:
        n_tl = sum(counts.get(c, 0) for c in cats)
        for m in models:
            if f"s3tl/{m['fp']}" not in done_tasks:
                total += DEFAULT_SECONDS["load"] + n_tl * per("tl")
        together_s = plan["together_clips"] * 6.0 + 60.0
        total += asr_union * len(models) * together_s
        judges = plan["judges"]
        candidates = [m["fp"] for m in models]
        assignment = judge_mod.assign_judges(candidates, judges)
        for j in judges:
            if f"s3judge/{j}" not in done_tasks:
                assigned = judge_mod.assigned_count(assignment, candidates, j)
                total += DEFAULT_SECONDS["load"] + (len(judge_mod.CALIBRATION) + assigned * n_tl) * per("judge")
    return total


def describe_estimate(store) -> str:
    return fmt_duration(estimate_remaining(store, store.state.get("timing")))


def git_rev() -> str:
    try:
        out = subprocess.run(["git", "-C", str(paths.ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "?"
    except (OSError, subprocess.TimeoutExpired):
        return "?"
