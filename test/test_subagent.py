import types

import app.tool.subagent as subagent_mod
from app.tool.tools import TOOLS, TOOL_HANDLERS


def _text_block(text):
    return types.SimpleNamespace(type="text", text=text)


def _tool_use_block(name, input_, bid="tu_1"):
    return types.SimpleNamespace(type="tool_use", id=bid, name=name, input=input_)


def test_child_tools_exclude_subagent():
    """CHILD_TOOLS 必须不含 subagent，结构上不可嵌套递归；其余工具保留。"""
    child_names = [t["name"] for t in subagent_mod.CHILD_TOOLS]
    parent_names = [t["name"] for t in TOOLS]

    assert "subagent" in parent_names          # 父 Agent 拥有 subagent
    assert "subagent" not in child_names       # 子 Agent 不拥有 subagent
    assert set(child_names) == set(parent_names) - {"subagent"}


def test_run_subagent_isolates_session_and_returns_answer(monkeypatch):
    """子 Agent 工具调用注入 sub_session_id（非父），最终返回子 Agent 文本，沙箱被清理。"""
    parent_session_id = "parent-sess-123"

    seen_bash_session_ids = []
    destroyed_ids = []

    def fake_bash(**kw):
        seen_bash_session_ids.append(kw.get("session_id"))
        return "bash-ok"

    # 两次 LLM 调用：第一次 tool_use(bash)，第二次纯文本 "DONE"
    responses = [
        types.SimpleNamespace(
            content=[_tool_use_block("bash", {"command": "echo hi"})],
            stop_reason="tool_use",
        ),
        types.SimpleNamespace(
            content=[_text_block("DONE")],
            stop_reason="end_turn",
        ),
    ]

    def fake_messages_create(**kwargs):
        return responses.pop(0)

    fake_client = types.SimpleNamespace(messages_create=fake_messages_create)

    monkeypatch.setattr(subagent_mod, "client", fake_client)
    monkeypatch.setitem(TOOL_HANDLERS, "bash", fake_bash)
    monkeypatch.setattr(subagent_mod, "destroy_sandbox", lambda sid: destroyed_ids.append(sid))

    answer = subagent_mod.run_subagent(parent_session_id, "调研 A")

    # 最终答案为子 Agent 第二次返回的文本
    assert answer == "DONE"

    # bash 被调用一次，且 session_id 是 sub_session_id（以父 id + __sub_ 前缀开头），不是父 id
    assert len(seen_bash_session_ids) == 1
    sub_sid = seen_bash_session_ids[0]
    assert sub_sid.startswith(f"{parent_session_id}__sub_")
    assert sub_sid != parent_session_id

    # 沙箱被清理一次，且用的是同一个 sub_session_id
    assert destroyed_ids == [sub_sid]
