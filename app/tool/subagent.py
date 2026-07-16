"""子 Agent 工具：父 Agent 委派子任务给独立上下文的子 Agent 执行。

子 Agent 拥有独立 session_id（沙箱 / 任务图 / 消息上下文均隔离），内部跑精简
非流式 ReAct 循环，完成后只把最终结论文本作为 tool_result 返回父 Agent。
子 Agent 工具集合剔除 subagent，结构上不可嵌套递归。

注意循环依赖：本模块模块级导入 react_agent 与 tools 是安全的，因为 tools.py
对 run_subagent 采用懒导入（_subagent_handler 内部才 import），故导入图无环。
"""

import os
import time
from uuid import uuid4

import anthropic

from app.llm.react_agent import client, MODEL
from app.llm.context_compact import micro_compact
from app.llm.context_optimizer import optimize_thinking_for_llm
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
# 与父循环对齐：429 / 超时最多重试 10 次，每次间隔 10s
_LLM_MAX_RETRIES = 10
_LLM_RETRY_SLEEP = 10


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
        last_real_text = ""  # 迭代上限时取最后一条含真实文本的产出（审查 #4）
        hit_limit = False
        for iteration in range(1, MAX_ITERATIONS + 1):
            # 每轮调用 LLM 前：与父循环一致的 thinking 精简 + 上下文压缩，防窗口溢出（审查 #2）
            messages = optimize_thinking_for_llm(messages)
            messages = micro_compact(messages)

            resp = _messages_create_with_retry(messages)

            messages.append({"role": "assistant", "content": resp.content})
            LOG.info(f"subagent 迭代 {iteration}: sub={sub_session_id} stop_reason={resp.stop_reason}")

            # 记录本轮真实文本产出，供迭代上限路径使用（审查 #4）
            round_text = _extract_text(resp.content)
            if round_text:
                last_real_text = round_text

            # max_tokens 截断不当最终答案：并入消息 + 续轮提示，继续下一轮（审查 #6）
            if resp.stop_reason == "max_tokens":
                messages.append({"role": "user", "content": "（上一条回复因长度被截断，请继续完成你的结论）"})
                continue

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
            # 达到迭代上限：取最后一条含真实 text 块的产出 + 提示（审查 #4）
            hit_limit = True
            final_text = last_real_text

        if hit_limit:
            final_text = (final_text or "") + f"\n\n[子Agent提示] 已达最大迭代次数 {MAX_ITERATIONS}，停止执行。"
            LOG.warning(f"subagent 达到迭代上限: sub={sub_session_id} max={MAX_ITERATIONS}")

        # 截断到上限并追加截断标记，让父 Agent 感知被截断（审查 #8）
        if len(final_text) > MAX_ANSWER_CHARS:
            orig_len = len(final_text)
            final_text = final_text[:MAX_ANSWER_CHARS] + (
                f"\n\n...[子 Agent 最终答案已截断至 {MAX_ANSWER_CHARS} 字符，原长度 {orig_len} 字符]"
            )
        return final_text
    except Exception as e:
        LOG.exception(f"subagent 执行异常: sub={sub_session_id} err={e}")
        return f"[子Agent错误] {type(e).__name__}: {e}"
    finally:
        destroy_sandbox(sub_session_id)
        LOG.info(f"subagent 沙箱已清理: sub={sub_session_id}")


def _messages_create_with_retry(messages):
    """调用子 Agent LLM，对 429 / 超时按与父循环一致的次数重试（审查 #5）。

    重试耗尽才抛出，由 run_subagent 外层捕获并返回 ``[子Agent错误]``。
    """
    last_err = None
    for attempt in range(1, _LLM_MAX_RETRIES + 1):
        try:
            return client.messages_create(
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
        except (anthropic.RateLimitError, anthropic.APITimeoutError) as e:
            last_err = e
            LOG.exception(f"subagent LLM 限流/超时，重试中... ({attempt}/{_LLM_MAX_RETRIES})")
            time.sleep(_LLM_RETRY_SLEEP)
    raise last_err


def _extract_text(content) -> str:
    """从一条 assistant 消息的 content 中拼接所有 text 块文本。"""
    parts = []
    for block in content or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "".join(parts)
