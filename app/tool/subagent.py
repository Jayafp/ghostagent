"""子 Agent 工具：父 Agent 委派子任务给独立上下文的子 Agent 执行。

子 Agent 拥有独立 session_id（沙箱 / 任务图 / 消息上下文均隔离），内部跑精简
非流式 ReAct 循环，完成后只把最终结论文本作为 tool_result 返回父 Agent。
子 Agent 工具集合剔除 subagent，结构上不可嵌套递归。

注意循环依赖：本模块模块级导入 react_agent 与 tools 是安全的，因为 tools.py
对 run_subagent 采用懒导入（_subagent_handler 内部才 import），故导入图无环。
"""

import os
from uuid import uuid4

from app.llm.react_agent import client, MODEL
from app.tool.tools import TOOLS, TOOL_HANDLERS, destroy_sandbox
from app.log.logger import LOG


# 子 Agent 可用工具：全工具减 subagent，防止递归嵌套
CHILD_TOOLS = [t for t in TOOLS if t["name"] != "subagent"]

# 子 Agent 专用精简 system prompt（不注入 skill 列表，保持最小）
_SUB_SYSTEM_PROMPT = (
    "你是一个子 Agent，负责独立完成父 Agent 委派给你的单个任务。\n"
    "你有完整的工具可用（除再次委派子 Agent 外），请善用工具完成任务。\n"
    "你与父 Agent 上下文隔离，看不到父会话历史，仅依据本任务说明和你的工具结果工作。\n"
    "完成后，用简洁中文回复最终结论 / 产出，供父 Agent 决策下一步。"
)

MAX_ITERATIONS = 50
MAX_ANSWER_CHARS = 20000


def run_subagent(parent_session_id: str, task: str, context: str = "") -> str:
    """启动子 Agent 完成委派任务，阻塞直到完成，返回最终答案文本。

    任何异常都返回 ``[子Agent错误] ...`` 字符串，不向父 Agent 抛出；
    无论成功失败都在 finally 中调用 destroy_sandbox 清理子沙箱。
    """
    sub_session_id = f"{parent_session_id}__sub_{uuid4().hex[:8]}"
    LOG.info(f"subagent 启动: parent={parent_session_id} sub={sub_session_id} task={task[:200]}")

    task_prompt = "【委派任务】\n" f"{task}\n"
    if context and context.strip():
        task_prompt += f"\n【补充上下文】\n{context}\n"
    task_prompt += "\n请独立完成上述任务，善用工具。完成后用简洁中文回复最终结论 / 产出。"

    messages = [{"role": "user", "content": task_prompt}]

    try:
        final_text = ""
        hit_limit = False
        for iteration in range(1, MAX_ITERATIONS + 1):
            resp = client.messages_create(
                model=MODEL,
                system=_SUB_SYSTEM_PROMPT,
                messages=messages,
                tools=CHILD_TOOLS,
                max_tokens=128000,
                thinking={
                    "type": os.getenv("LLM_THINKING_TYPE", "disabled"),
                    "budget_tokens": int(os.getenv("budget_tokens", 4096)),
                },
            )
            messages.append({"role": "assistant", "content": resp.content})
            LOG.info(f"subagent 迭代 {iteration}: sub={sub_session_id} stop_reason={resp.stop_reason}")

            tool_use_blocks = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            if resp.stop_reason != "tool_use" or not tool_use_blocks:
                final_text = _extract_text(resp.content)
                break

            # 串行执行 tool_use 块，组装 tool_result 列表回灌
            tool_results = []
            for block in tool_use_blocks:
                name = block.name
                handler = TOOL_HANDLERS.get(name)
                try:
                    if handler is None:
                        output = f"Unknown tool: {name}"
                    else:
                        output = handler(**block.input, session_id=sub_session_id)
                except Exception as e:
                    LOG.exception(f"subagent 子工具 '{name}' 抛出异常: {e}")
                    output = f"Tool '{name}' execution error: {e}"
                if not isinstance(output, str):
                    output = str(output)
                LOG.info(f"subagent 子工具: {name}[{block.input}] -> {output[:200].replace(chr(10), '. ')}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })
            messages.append({"role": "user", "content": tool_results})
        else:
            # 达到迭代上限：取已产出的最后一条 assistant 文本 + 提示
            hit_limit = True
            final_text = _extract_last_assistant_text(messages)

        if hit_limit:
            final_text = (final_text or "") + f"\n\n[子Agent提示] 已达最大迭代次数 {MAX_ITERATIONS}，停止执行。"
            LOG.warning(f"subagent 达到迭代上限: sub={sub_session_id} max={MAX_ITERATIONS}")

        if len(final_text) > MAX_ANSWER_CHARS:
            final_text = final_text[:MAX_ANSWER_CHARS]
        return final_text
    except Exception as e:
        LOG.exception(f"subagent 执行异常: sub={sub_session_id} err={e}")
        return f"[子Agent错误] {type(e).__name__}: {e}"
    finally:
        destroy_sandbox(sub_session_id)
        LOG.info(f"subagent 沙箱已清理: sub={sub_session_id}")


def _extract_text(content) -> str:
    """从一条 assistant 消息的 content 中拼接所有 text 块文本。"""
    parts = []
    for block in content or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "".join(parts)


def _extract_last_assistant_text(messages) -> str:
    """从 messages 倒序找最后一条 assistant 消息，提取其 text 块文本。"""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return _extract_text(msg.get("content"))
    return ""
