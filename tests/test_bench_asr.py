"""bench.asr_runner / tl_runner / vad_runner / llm_servers against the real app code.

The benchmark copies a few pieces of main.py that cannot be imported on their own (the
worker config, the Translator arguments, the subtitle filters). These tests compare the
copies with main.py so a change there cannot silently skew the benchmark."""
import ast
import re
from pathlib import Path

import numpy as np
import pytest

from bench import asr_runner, llm_servers, tl_runner, vad_runner

ROOT = Path(__file__).resolve().parents[1]
MAIN = (ROOT / "main.py").read_text(encoding="utf-8")


def _method(name):
    tree = ast.parse(MAIN)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in main.py")


# ---------------------------------------------------------------- worker config

def test_worker_config_has_exactly_the_keys_main_builds():
    fn = _method("_switch_asr_engine")
    keys = None
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "worker_config" for t in node.targets):
            keys = {k.value for k in node.value.keys}
    assert keys, "worker_config dict not found"
    settings = {"hub": "hf", "asr_language": "ja", "asr_device": "cuda:0", "whisper_model_size": "medium"}
    config = {"asr": {"compute_type": "float16", "beam_size": 1, "whisper": {"x": 1}}}
    for cand in asr_runner.ASR_CANDIDATES:
        cfg = asr_runner.build_worker_config(cand, settings, config)
        assert set(cfg) == keys, cand.id
        assert cfg["engine_type"] == cand.engine_type
        assert cfg["language"] == "ja" and cfg["hub"] == "hf" and cfg["beam_size"] == 1


def test_worker_config_per_engine():
    settings = {"hub": "hf", "asr_language": "ja", "sensevoice_pad_seconds": 0.4, "whisper_pad_seconds": 0.3}
    config = {"asr": {"compute_type": "float16"}}
    by_id = {c.id: asr_runner.build_worker_config(c, settings, config) for c in asr_runner.ASR_CANDIDATES}
    assert by_id["sensevoice"]["funasr_model"] == "sensevoice-small" and by_id["sensevoice"]["pad_seconds"] == 0.4
    assert by_id["mlt-nano"]["model_size"] == "funasr-mlt-nano-2512"
    assert by_id["anime-whisper"]["pad_seconds"] is None
    assert by_id["whisper-turbo"]["pad_seconds"] == 0.3


# ---------------------------------------------------------------- translator arguments

def test_translator_gets_the_same_arguments_as_main(monkeypatch):
    fn = _method("_on_model_changed")
    main_kwargs = None
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Translator":
            main_kwargs = {k.arg for k in node.keywords}
    assert main_kwargs

    import translator as translator_mod

    seen = {}

    class FakeTranslator:
        def __init__(self, **kw):
            seen.update(kw)
            self.turns = None

        def set_context_turns(self, n):
            self.turns = n

    monkeypatch.setattr(translator_mod, "Translator", FakeTranslator)
    entry = {"name": "Sakura", "model": "hf.co/SakuraLLM/x", "api_base": "http://127.0.0.1:11434/v1",
             "api_key": "ollama", "prompt_style": "sakura", "thinking_style": "off", "context_turns": 3,
             "overrides": {"temperature": 0.3}}
    t = tl_runner.build_translator(entry, {"target_language": "zh", "timeout": 10},
                                   {"translation": {"max_tokens": 256, "temperature": 0.3, "zh_tw_postprocess": True}})
    assert set(seen) == main_kwargs
    assert seen["prompt_style"] == "sakura" and seen["zh_tw_postprocess"] is True
    assert t.turns == 0  # unrelated test sentences: no context


# ---------------------------------------------------------------- subtitle filters vs main.py

SPLIT_SAMPLES = ["今日はいい天気ですね。明日も晴れるといいな", "本当に？うん、そうだよ！",
                 "えーっと、それはね、たぶん来週の水曜日あたりになると思うんだけど、まだわからない",
                 "はい。본론으로 들어가자 친구들。", "Hello there. How are you?"]


@pytest.mark.parametrize("text", SPLIT_SAMPLES)
def test_sentence_splitter_matches_main(text):
    import main as main_module

    app = main_module.LiveTranslateApp.__new__(main_module.LiveTranslateApp)
    for lang in ("ja", "en"):
        assert asr_runner.split_sentences(text, lang) == app._split_sentences(text, lang)


FILTER_SAMPLES = [
    ("本当にありがとうございます。위여 가자 빨리 가자 친구들。", "ja", 1.0),
    ("위여.", "ja", 1.0),
    ("本当に", "ja", 1.0),
    ("OK、行こう", "ja", 1.0),
    ("你好。我们走吧。", "zh", 1.0),   # engine label says another language
    ("本当に", "auto", 1.0),          # no label: judged by text
    ("。、", "ja", 1.0),               # punctuation only
    ("あ", "ja", 3.0),                 # noise: long segment, <= 3 characters
    ("はい。본론으로 들어가자 친구들。", "ja", 1.0),
]


@pytest.mark.parametrize("text, label, seconds", FILTER_SAMPLES)
def test_displayed_text_matches_what_main_commits(text, label, seconds):
    from test_language_filter import _Recorder
    from test_live_preview import _Panel, _app

    app = _app()
    app._panel = _Panel("ja")
    app._transcript = _Recorder()
    app._run_asr = lambda audio, kind, **kw: ({"text": text, "language": label}, 10.0)
    app._process_segment(np.zeros(int(16000 * seconds), dtype=np.float32))
    shown, _reason = asr_runner.displayed_text(text, label, seconds, "ja")
    assert app._transcript.lines == ([shown] if shown else [])


def test_hint_missing_matches_main():
    nano = asr_runner.AsrCandidate("nano", "Fun-ASR-Nano", "funasr", funasr_model="funasr-nano-2512")
    assert asr_runner.hint_missing_for(nano, "ko") is True
    assert asr_runner.hint_missing_for(nano, "ja") is False
    assert asr_runner.hint_missing_for(asr_runner.CANDIDATES_BY_ID["mlt-nano"], "ko") is False
    assert asr_runner.hint_missing_for(asr_runner.CANDIDATES_BY_ID["sensevoice"], "ko") is False
    assert asr_runner.hint_missing_for(asr_runner.CANDIDATES_BY_ID["mlt-nano"], "auto") is False


# ---------------------------------------------------------------- ASR session

class _FakeClient:
    instances = []

    def __init__(self, config, plan):
        self.config = config
        self.plan = plan
        self.closed = False
        _FakeClient.instances.append(self)

    def start(self):
        if self.plan.get("fail_start"):
            raise RuntimeError("cannot load")

    def wait_ready(self):
        return {}

    def transcribe(self, audio):
        from asr_client import ASRWorkerExited

        if self.plan.get("die"):
            self.plan["die"] -= 1
            raise ASRWorkerExited("worker died")
        return {"text": "はい", "language": "ja"}

    def shutdown(self):
        self.closed = True


def test_session_restarts_a_dead_worker_once():
    _FakeClient.instances.clear()
    plan = {"die": 1}
    s = asr_runner.AsrSession(asr_runner.CANDIDATES_BY_ID["sensevoice"], {}, client_factory=lambda c: _FakeClient(c, plan))
    s.load()
    result, ms = s.transcribe(np.zeros(160))
    assert result["text"] == "はい" and s.restarts == 1 and _FakeClient.instances[0].closed


def test_session_gives_up_after_two_deaths_and_load_failure_closes():
    from asr_client import ASRWorkerExited

    plan = {"die": 5}
    s = asr_runner.AsrSession(asr_runner.CANDIDATES_BY_ID["sensevoice"], {}, client_factory=lambda c: _FakeClient(c, plan))
    s.load()
    with pytest.raises(ASRWorkerExited):
        s.transcribe(np.zeros(160))
    _FakeClient.instances.clear()
    bad = asr_runner.AsrSession(asr_runner.CANDIDATES_BY_ID["sensevoice"], {},
                                client_factory=lambda c: _FakeClient(c, {"fail_start": True}))
    with pytest.raises(RuntimeError):
        bad.load()
    assert _FakeClient.instances[0].closed and bad.client is None


# ---------------------------------------------------------------- translation timing

class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.1
        return self.t


class _Tr:
    def __init__(self, parts=None, error=None):
        self.parts = parts or []
        self.error = error
        self.closed = False

    def translate_iter(self, text, source_language="ja"):
        try:
            for p in self.parts:
                yield p
            if self.error:
                raise self.error
        finally:
            self.closed = True


def test_translate_timed_measures_first_token_and_total():
    tr = _Tr(["今", "今天", "今天天氣很好。"])
    r = tl_runner.translate_timed(tr, "今日はいい天気", clock=_Clock())
    assert r["ok"] and r["text"] == "今天天氣很好。"
    # the clock is read at the start, at the first non-empty partial and at the end
    assert r["ttft_ms"] == pytest.approx(100) and r["total_ms"] == pytest.approx(200)
    assert tr.closed


@pytest.mark.parametrize("parts, error, kind", [
    (["今天"], TimeoutError("slow"), "timeout"),
    ([], ConnectionError("refused"), "connection"),
    ([""], None, "empty"),
    (["今日はいい天気ですね"], None, "untranslated"),
])
def test_translate_timed_classifies_failures(parts, error, kind):
    tr = _Tr(parts, error)
    r = tl_runner.translate_timed(tr, "x", clock=_Clock())
    assert not r["ok"] and r["error"] == kind
    assert tr.closed  # the stream is always closed (pool deadlock otherwise)


def test_repetition_is_its_own_failure_kind():
    from translator import RepetitionError

    r = tl_runner.translate_timed(_Tr(["哈"], RepetitionError("哈哈")), "x", clock=_Clock())
    assert r["error"] == "repetition"


def test_model_availability():
    ollama = {"model": "qwen3.5:9b", "api_base": "http://127.0.0.1:11434/v1", "api_key": "ollama"}
    assert tl_runner.availability(ollama, {"qwen3.5:9b"}, None) == (True, "已下載")
    assert tl_runner.availability(ollama, set(), None)[0] is False
    assert tl_runner.availability(ollama, None, None)[0] is False
    lm = {"model": "x", "api_base": "http://127.0.0.1:1234/v1", "api_key": "lm"}
    assert tl_runner.availability(lm, None, False)[0] is False
    cloud = {"model": "x", "api_base": "https://api.example.com/v1", "api_key": ""}
    assert tl_runner.availability(cloud, None, None) == (False, "沒有 API key")


# ---------------------------------------------------------------- Ollama settings in sync

def test_fast_ollama_settings_match_the_launcher():
    script = (ROOT / "launcher" / "start_jp_subtitles.ps1").read_text(encoding="utf-8-sig")
    block = script[script.index("function Start-WindowsOllama"):script.index("Start-Process -FilePath $Ollama")]
    launcher = dict(re.findall(r'\$env:(OLLAMA_\w+)\s*=\s*"([^"]+)"', block))
    ours = llm_servers.fast_env(11434)
    assert set(launcher) == set(ours)
    for key, value in launcher.items():
        if key != "OLLAMA_HOST":
            assert ours[key] == value, key


def test_model_kind_and_fingerprint():
    assert llm_servers.model_kind({"api_base": "http://127.0.0.1:11434/v1", "api_key": "ollama"}) == "ollama"
    assert llm_servers.model_kind({"api_base": "http://127.0.0.1:1234/v1", "api_key": "x"}) == "lmstudio"
    assert llm_servers.model_kind({"api_base": "http://localhost:8000/v1", "api_key": "x"}) == "local"
    assert llm_servers.model_kind({"api_base": "https://api.openai.com/v1", "api_key": "x"}) == "cloud"
    fp = llm_servers.fingerprint({"name": "N", "model": "m", "api_base": "b", "api_key": "secret"})
    assert "secret" not in fp


def test_ollama_manifest_path_matches_the_launcher_rule(tmp_path):
    from bench.cli import manifest_path

    assert manifest_path(tmp_path, "qwen3.5:9b") == tmp_path / "registry.ollama.ai" / "library" / "qwen3.5" / "9b"
    assert manifest_path(tmp_path, "hf.co/SakuraLLM/Sakura-GalTransl-7B-v3:IQ4_XS") == \
        tmp_path / "hf.co" / "SakuraLLM" / "Sakura-GalTransl-7B-v3" / "IQ4_XS"
    assert manifest_path(tmp_path, "user/model") == tmp_path / "registry.ollama.ai" / "user" / "model" / "latest"


# ---------------------------------------------------------------- VAD stage

class _EnergyVad:
    """Stand-in with the parts of VADProcessor segment_recording uses."""

    def __init__(self, threshold):
        self.threshold = threshold
        self._buf = []
        self._speech_samples = 0
        self._silence = 0

    def _get_effective_silence_limit(self):
        return 5

    def process_chunk(self, chunk):
        loud = float(np.abs(chunk).mean()) > self.threshold
        if loud:
            self._buf.append(chunk)
            self._speech_samples += len(chunk)
            self._silence = 0
        elif self._buf:
            self._silence += 1
            self._buf.append(chunk)
            self._speech_samples += len(chunk)
            if self._silence >= 5:
                return self.flush()
        return None

    def flush(self):
        if not self._buf:
            return None
        seg = np.concatenate(self._buf)
        self._buf, self._speech_samples, self._silence = [], 0, 0
        return seg


def test_segment_recording_timestamps():
    sr = vad_runner.SR
    audio = np.zeros(sr * 6, dtype=np.float32)
    audio[sr * 1:sr * 2] = 0.5   # speech 1.0-2.0 s
    audio[sr * 4:int(sr * 4.5)] = 0.5  # speech 4.0-4.5 s, file ends 1.5 s later
    segs = vad_runner.segment_recording(audio, _EnergyVad(0.1))
    assert len(segs) == 2
    assert segs[0]["start"] == pytest.approx(1.0, abs=0.05)
    assert segs[1]["start"] == pytest.approx(4.0, abs=0.05)


def test_continuous_windows_cut_at_the_quietest_point():
    sr = vad_runner.SR
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(sr * 70) * 0.3).astype(np.float32)
    audio[int(sr * 24.0):int(sr * 24.1)] = 0.0
    windows = vad_runner.continuous_windows(audio)
    assert windows[0][0] == 0 and abs(windows[0][1] / sr - 24.0) < 0.05
    assert windows[-1][1] == len(audio)
    assert all(b - a <= 30 * sr for a, b in windows)


def test_best_threshold_prefers_fewer_errors_then_the_default():
    assert vad_runner.best_threshold({0.3: 2.0, 0.4: 1.0, 0.6: 1.0}) == 0.4
    assert vad_runner.best_threshold({0.4: 1.0, 0.6: 1.0}) in (0.4, 0.6)
    assert vad_runner.best_threshold({0.3: 1.0, 0.5: 1.0}) == 0.5
    assert vad_runner.best_threshold({}) == 0.5


def test_make_vad_uses_the_user_vad_settings(monkeypatch):
    import vad_processor

    created = {}

    class FakeVAD:
        def __init__(self, sr, **kw):
            created["init"] = kw

        def update_settings(self, s):
            created["settings"] = s

    monkeypatch.setattr(vad_processor, "VADProcessor", FakeVAD)
    vad_runner.make_vad(0.4, {"silence_mode": "fixed", "silence_duration": 0.5, "min_speech_duration": 0.6,
                              "vad_threshold": 0.27, "unrelated": 1})
    assert created["settings"]["vad_threshold"] == 0.4 and created["settings"]["silence_mode"] == "fixed"
    assert "unrelated" not in created["settings"] and created["init"]["min_speech_duration"] == 0.6
