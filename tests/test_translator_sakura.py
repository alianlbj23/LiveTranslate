"""SakuraLLM needs its own prompt: a fixed Simplified-Chinese system prompt and a user
message with 历史翻译 / 术语表 / input (model card of Sakura-GalTransl-7B-v3). Other
models keep the user's system prompt and multi-turn context."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import translator as tr_mod  # noqa: E402
from translator import Translator  # noqa: E402


def _tr(**kw):
    return Translator(api_base="http://127.0.0.1:11434/v1", api_key="ollama", model="m",
                      target_language="zh", streaming=True, system_prompt="翻譯成{target_lang}：{source_lang}", **kw)


def test_sakura_messages_follow_the_model_card():
    t = _tr(prompt_style="sakura")
    msgs = t._build_messages(t._build_system_prompt("ja"), "えっ、マジで？")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == tr_mod.SAKURA_SYSTEM_PROMPT
    user = msgs[1]["content"]
    assert "参考以下术语表（可为空，格式为src->dst #备注）：" in user
    assert user.endswith("将下面的文本从日文翻译成简体中文：\nえっ、マジで？")
    assert "历史翻译" not in user, "no history yet"


def test_sakura_history_goes_into_the_user_message():
    t = _tr(prompt_style="sakura")
    t.set_context_turns(2)
    t._append_history("一", "第一句")
    t._append_history("二", "第二句")
    t._append_history("三", "第三句")
    msgs = t._build_messages(t._build_system_prompt("ja"), "四")
    assert len(msgs) == 2, "history is text in the prompt, not extra chat turns"
    assert msgs[1]["content"].startswith("历史翻译：第二句\n第三句\n")


def test_live_preview_copy_keeps_the_sakura_prompt():
    t = _tr(prompt_style="sakura").with_target_language("zh")
    msgs = t._build_messages(t._build_system_prompt("ja"), "テスト")
    assert msgs[0]["content"] == tr_mod.SAKURA_SYSTEM_PROMPT


def test_sakura_prompt_only_for_chinese_targets():
    """Error path: Sakura only translates into Chinese; other targets keep the normal prompt."""
    t = _tr(prompt_style="sakura")
    t.set_target_language("en")
    msgs = t._build_messages(t._build_system_prompt("ja"), "テスト")
    assert msgs[0]["content"] != tr_mod.SAKURA_SYSTEM_PROMPT
    assert msgs[-1]["content"] == "テスト"


def test_default_prompt_style_unchanged():
    t = _tr()
    t.set_context_turns(1)
    t._append_history("一", "第一句")
    msgs = t._build_messages(t._build_system_prompt("ja"), "二")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[-1]["content"] == "二"


def test_unknown_prompt_style_falls_back_to_default(caplog):
    with caplog.at_level("WARNING"):
        t = _tr(prompt_style="nonsense")
    assert t._prompt_style is None
    assert "nonsense" in caplog.text
