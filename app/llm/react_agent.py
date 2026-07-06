# !/usr/bin/env python3
# Harness: the loop -- the model's first connection to the real world.
import os
import time
import json
import atexit
from concurrent.futures import ThreadPoolExecutor
from typing import Generator, Dict, Tuple, Optional, List
from enum import Enum

import anthropic
from pathlib import Path

from app.llm.human_perception import get_human_perception_as_message_fmt, async_generate_perception_ifneed
from app.log.logger import LOG

from app.llm.anthropic_logging import LoggingAnthropic, serialize_messages
from app.skill.skill_manager import SKILL_MANAGER
from app.tool.tools import TOOLS, TOOL_HANDLERS, PARALLEL_SAFE_TOOLS, destroy_sandbox, TASK_TOOLS
from app.llm.context_compact import micro_compact, smart_compact, TOKEN_SOFT_LIMIT, TOKEN_HARD_LIMIT, LLM_MAX_WINDOW
from app.llm.utils import estimate_tokens, usage_tokens
from app.llm.session_manager import session
from app.llm.memory_manager import append_message, KEEP_RECENT_ROUNDS, INIT_RECENT_ROUNDS, MEMORY_DIR
from app.llm.memory_retrieval import get_retriever, clear_retriever_cache
from app.llm.context_optimizer import optimize_tool_results_for_memory, optimize_tool_results_for_llm, \
    optimize_thinking_for_llm

client = LoggingAnthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"), api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = os.environ["MODEL_ID"]


def _atexit_cleanup():
    """进程退出时清理所有沙箱容器"""
    try:
        from app.tool.tools import _container_manager
        _container_manager.destroy_all()
    except Exception:
        pass


atexit.register(_atexit_cleanup)

# 缓存的 System Prompt 模板（懒加载）
_PROMPT_TEMPLATE_CACHE = None

# Session 级别的停止标志 {session_id: bool}
_stop_flags: Dict[str, bool] = {}


def stop_agent(session_id: str) -> bool:
    """
    请求终止指定 session 的 agent 运行

    在下一轮迭代时，agent 会检查此标志并优雅地停止

    Args:
        session_id: 会话唯一标识符

    Returns:
        bool: True - 成功设置停止标志
              False - session_id 为空
    """
    if session_id:
        _stop_flags[session_id] = True
        LOG.info(f"已请求终止 session: {session_id}")
        return True
    return False


def _is_stopped(session_id: str) -> bool:
    """
    检查指定 session 是否被请求终止

    Args:
        session_id: 会话唯一标识符

    Returns:
        bool: True - 已请求终止
              False - 未请求终止或 session 不存在
    """
    return _stop_flags.get(session_id, False)


def _clear_stop_flag(session_id: str) -> None:
    """
    清除停止标志

    在 agent_loop 开始时调用，确保新对话能正常开始

    Args:
        session_id: 会话唯一标识符
    """
    if session_id in _stop_flags:
        del _stop_flags[session_id]


def _load_prompt_template() -> str:
    """
    加载 System Prompt 模板

    使用缓存机制，只读取一次文件

    Returns:
        str: Prompt 模板内容

    Raises:
        FileNotFoundError: 如果模板文件不存在
        其他异常: 读取失败时

    Note:
        模板文件路径：app/prompts/system_prompt_default.md
    """
    global _PROMPT_TEMPLATE_CACHE
    if _PROMPT_TEMPLATE_CACHE is not None:
        return _PROMPT_TEMPLATE_CACHE

    prompt_file = Path(__file__).parent.parent / "prompts" / "system_prompt_default.md"

    try:
        _PROMPT_TEMPLATE_CACHE = prompt_file.read_text(encoding='utf-8')
        LOG.info(f"已加载 prompt 模板: {prompt_file}")
    except FileNotFoundError:
        LOG.error(f"Prompt 文件未找到: {prompt_file}")
        raise
    except Exception as e:
        LOG.exception(f"加载 prompt 模板失败: {e}")
        raise

    return _PROMPT_TEMPLATE_CACHE


def get_session_special_info(session_id: str):
    session_special_file = MEMORY_DIR / Path(f"{session_id}/special_prompt.md")
    if not session_special_file.exists():
        return ""

    special_prompt = ""
    with open(session_special_file, 'r', encoding="utf-8") as f:
        special_prompt = f.read()

    special_prompt = f"\n### 用户的特定要求\n\n⚠️ **这是用户最重要的要求，请记住并遵循它。**\n\n{special_prompt}\n\n"
    return special_prompt


def build_system_prompt(session_id: str, user_message: str = "") -> str:
    """
    构建动态 system prompt

    加载模板并注入变量（当前年份、skill 列表等）

    Args:
        session_id: 会话唯一标识符（保留参数）
        user_message: 当前用户输入（保留参数，历史记忆不再直接注入）

    Returns:
        str: 完整的 system prompt

    Note:
        历史记忆不再直接注入 system prompt，而是通过 memory_search 工具让模型自主检索
    """
    # # 历史记忆不再直接注入system prompt, 而是通过工具让模型自己决策需要检索
    # memories_section = "(无历史记忆信息)"
    #
    # # 只有在有用户输入且 session 有历史记录时才检索
    # if user_message and len(user_message.strip()) > 2:
    #     try:
    #         retriever = get_retriever(session_id)
    #         if retriever:
    #             results = retriever.search(
    #                 user_message,
    #                 top_k=int(os.getenv("MEMORY_TOP_K", "3")),
    #                 bm25_weight=0.5,
    #                 vector_weight=0.5
    #             )
    #             if results:
    #                 memories_section = retriever.format_for_prompt(results, max_length=800)
    #                 # 添加详细的日志信息
    #                 result_line = ""
    #                 for i,result in enumerate(results): result_line += f'\n{i+1}. 内容: {result.content[:200].replace("\n", " ")}\n{result.bm25_score:.3f} | {result.vector_score:.3f} | {result.final_score:.3f}'
    #                 LOG.info(f"为用户查询检索到 {len(results)} 条相关记忆: {result_line}")
    #     except Exception as e:
    #         LOG.warning(f"检索历史记忆失败: {e}")

    now = time.localtime()
    system_prompt = _load_prompt_template().format(
        # 不在 system prompt 写入当前时间, 优化 prompt cache
        # current_time=time.strftime("%Y-%m-%d %H:%M:%S", now),
        current_year=time.strftime("%Y", now),
        skill_list=SKILL_MANAGER.get_all_skill_desc(),
        special_prompt=get_session_special_info(session_id),
    )

    LOG.debug(f"System Prompt:\n{system_prompt}")
    return system_prompt


class StreamEvent(str, Enum):
    """
    流式事件类型枚举

    用于区分不同类型的输出，前端据此进行差异化展示：

    - PROCESS: 过程内容（工具调用、中间步骤）
    - ANSWER: 最终答案（完整的助手回复）
    - TEXT: 文本增量（流式输出的文本片段）

    前端展示建议：
    - PROCESS：显示为灰色/辅助信息区域
    - ANSWER：显示为主要对话内容
    - TEXT：实时拼接显示
    - TASK：任务图状态变更，UI 据此刷新任务进度面板
    """
    PROCESS = "process"
    ANSWER = "answer"
    TEXT = "text"
    TASK = "task"


# -- The core pattern: a while loop that calls tools until the model stops --
def exec_cmd(cmd: str, session_id: str) -> Tuple[bool, Optional[str]]:
    """
    执行特殊命令

    处理以 / 开头的特殊命令，这些命令不经过 LLM 处理

    Args:
        cmd: 命令字符串（如 "/clear", "/status"）
        session_id: 会话唯一标识符

    Returns:
        Tuple[bool, Optional[str]]: (是否匹配命令, 响应消息)
            - (True, str): 命令已处理，返回响应
            - (False, None): 不是特殊命令，继续正常处理

    Supported Commands:
        /clear: 清空会话上下文和 memory 缓存
        /context: 查看当前上下文的 JSON 表示
        /skill: 列出所有已加载的 SKILL
        /status: 显示会话统计信息（token 数、消息数、压缩状态）
        /compact: 手动触发上下文压缩
        /reload: 重新从 memory 加载历史消息
    """
    messages = session.get_messages(session_id, rounds=INIT_RECENT_ROUNDS, force_compact=True)

    if cmd == '/clear':
        messages.clear()
        # 同时清除检索器缓存，确保下次重新加载
        clear_retriever_cache(session_id)
        LOG.info('模型上下文已清空...')
        return True, '上下文已清空~！'

    if cmd == '/context':
        real_messages = messages
        human_perception = get_human_perception_as_message_fmt(session_id)
        if human_perception:
            real_messages = [human_perception, *messages]
        return True, f"\n```\n{json.dumps(serialize_messages(real_messages), indent=2, ensure_ascii=False)}\n```\n"

    if cmd == '/skill':
        skills = SKILL_MANAGER.get_all_skils()
        if not skills:
            return True, "📝 **当前没有加载任何 SKILL**\n\n请检查 `SKILL_PATH` 环境变量配置。"

        skill_list = []
        for skill in skills:
            name = skill.get('name', 'N/A')
            desc = skill.get('description', '无描述')
            path = skill.get('path', 'N/A')
            skill_list.append(f"• **{name}**\n  └─ {desc}\n  └─ 路径: `{path}`")

        report = f"""🛠️ **已加载的 SKILL ({len(skills)} 个)**

{chr(10).join(skill_list)}

---

**路径来源:** `{os.getenv('SKILL_PATH', '未配置')}`
"""
        return True, report

    if cmd == '/status':
        token_count = estimate_tokens(messages)
        message_count = len(messages)

        # 计算状态
        if token_count < TOKEN_SOFT_LIMIT:
            status = "✅ 正常"
            progress = token_count / TOKEN_SOFT_LIMIT
        elif token_count < TOKEN_HARD_LIMIT:
            status = "⚠️ 警告"
            progress = token_count / TOKEN_HARD_LIMIT
        else:
            status = "❌ 即将压缩"
            progress = 1.0

        # 可视化进度条
        bar_length = 20
        filled = int(progress * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)

        report = f"""
📊 **会话状态**

├─ Session ID: `{session_id}`
├─ 消息数量: {message_count} 条
├─ 估算 Tokens: {token_count:,} / {LLM_MAX_WINDOW}
├─ 状态: {status}
├─ 进度: [{bar}] {progress * 100:.1f}%
│
├─ 软限制阈值: {TOKEN_SOFT_LIMIT:,} ({TOKEN_SOFT_LIMIT / LLM_MAX_WINDOW * 100:.0f}%)
├─ 硬限制阈值: {TOKEN_HARD_LIMIT:,} ({TOKEN_HARD_LIMIT / LLM_MAX_WINDOW * 100:.1f}%)
│
└─ 压缩策略: 保留最近10轮完整对话，更早内容生成摘要

**可用命令:**
- `/context` - 查看完整上下文
- `/clear` - 清空会话
- `/compact` - 手动压缩上下文
- `/reload` - 重新从 memory 加载记忆消息
- `/status` - 显示会话状态
- `/skill` - 列出已加载的 SKILL
- `/btw <消息>` - By the way 模式，本次对话不记录到上下文和文件
"""
        return True, report

    if cmd == '/compact':
        original_count = len(messages)
        original_tokens = estimate_tokens(messages)

        # 强制压缩上下文
        compacted = smart_compact(messages, session_id=session_id, force=True)
        compacted_count = len(compacted)
        compacted_tokens = estimate_tokens(compacted)

        # 更新 session 的消息列表（使用切片赋值避免引用问题）
        messages[:] = compacted

        # 生成报告
        report = f"""📋 **上下文压缩完成**

```
压缩前: {original_count} 条消息, 约 {original_tokens:,} tokens
压缩后: {compacted_count} 条消息, 约 {compacted_tokens:,} tokens
减少: {original_count - compacted_count} 条消息, 约 {original_tokens - compacted_tokens:,} tokens
```

**压缩后上下文:**

```json
{json.dumps(serialize_messages(messages), indent=2, ensure_ascii=False)}
```
"""
        return True, report

    if cmd == '/reload':
        session.clear_session(session_id)
        LOG.info(f"上下文已重新加载 [session_id={session_id}]")
        return True, "上下文已重新加载"

    return False, None


def resp_text(messages: List[Dict]) -> str:
    """
    从消息列表中提取最后一条消息的文本内容

    Args:
        messages: 消息字典列表

    Returns:
        str: 最后一条消息的文本内容
             如果为空或无法提取，返回 "(None Output)"
    """
    response_content = messages[-1]["content"]
    if isinstance(response_content, list):
        for block in response_content:
            if hasattr(block, "text"):
                return block.text
    return "(None Output)"


def agent_loop(message: str, session_id: str) -> Generator[Dict, None, None]:
    """
    执行 ReAct Agent 主循环，支持流式输出

    核心功能：
    1. 处理特殊命令（/clear, /context, /status 等）
    2. 管理对话上下文（加载、压缩、优化）
    3. 支持 /btw（By The Way）模式，不记录到历史
    4. 调用 LLM 并处理响应（文本或工具调用）
    5. 执行工具并返回结果
    6. 处理 API 限流和超时错误

    输出规则：
    - 工具调用信息 → StreamEvent.PROCESS
    - 工具执行结果 → StreamEvent.PROCESS
    - 流式文本增量 → StreamEvent.TEXT
    - 最终答案 → StreamEvent.ANSWER

    Args:
        message: 用户输入文本
        session_id: 会话唯一标识符

    Yields:
        Dict: 流式事件
            {
                "type": StreamEvent,
                "content": str
            }

    Processing Flow:
        1. 清除停止标志
        2. 检查是否为特殊命令
        3. 加载会话历史
        4. 优化上下文（tool_result 压缩、thinking 删除）
        5. 构建 system prompt
        6. 循环调用 LLM 直到没有 tool_use
           - 检查停止标志
           - 上下文压缩（如果需要）
           - 添加人类感知（如果存在）
           - 流式接收响应
           - 如果有 tool_use，执行工具并继续循环
        7. 处理错误（限流、超时等）

    Special Commands:
        /clear: 清空会话历史
        /context: 查看原始上下文
        /status: 查看会话状态（token 数、消息数）
        /compact: 手动压缩上下文
        /skill: 列出已加载的技能
        /reload: 重新从 memory 加载
        /btw <message>: By The Way 模式（不记录）

    Error Handling:
        - 限流/超时：自动重试，最多 10 次
        - 其他错误：记录日志并返回错误信息
    """
    # 清除该 session 的停止标志，确保新对话能正常开始
    _clear_stop_flag(session_id)

    is_cmd, cmd_result = exec_cmd(message, session_id)
    if is_cmd:
        yield {"type": StreamEvent.ANSWER, "content": cmd_result}
        return

    # 生成感知信息
    async_generate_perception_ifneed(session_id)

    # 检查是否是 /btw (by the way) 模式 - 不记录到上下文和 memory
    btw_mode = message.strip().lower().startswith('/btw')
    if btw_mode:
        # 去掉 /btw 前缀，获取实际消息
        actual_message = message.strip()[4:].strip()
        LOG.info(f"BTW 模式: 本次对话不记录到上下文 [session_id={session_id}]")
    else:
        actual_message = message

    error_count = 0
    messages = session.get_messages(session_id, rounds=INIT_RECENT_ROUNDS, force_compact=True)

    # 优化二：为 LLM 上下文优化历史 tool_result（超过 N 轮的 huge result 进行压缩）
    messages = optimize_tool_results_for_llm(messages)

    # 优化三：删除超过 N 轮的历史 thinking 块
    messages = optimize_thinking_for_llm(messages)

    # 构建动态 system prompt（包含相关历史记忆）
    dynamic_system = build_system_prompt(session_id, actual_message)

    # 初始化当前使用的消息列表
    if btw_mode:
        # btw 模式下创建临时消息列表，不影响原始上下文
        active_messages = messages.copy()
        active_messages.append({"role": "user", "content": actual_message})
    else:
        messages.append({"role": "user", "content": actual_message})
        # 写入 user 消息到 memory
        append_message(session_id, "user", actual_message)
        active_messages = messages

    while True:
        # 检查是否被请求终止
        if _is_stopped(session_id):
            destroy_sandbox(session_id)
            yield {"type": StreamEvent.PROCESS, "content": "\n\n⚠️ 任务已被用户终止\n"}
            LOG.info(f"agent loop 被用户终止 [session_id={session_id}]")
            return

        # btw 模式下不进行上下文压缩（因为是临时的）
        if not btw_mode:
            # 智能上下文压缩，传递 session_id 用于保存 memories
            compact_result = micro_compact(messages, session_id=session_id)
            if len(compact_result) < len(messages):
                yield {"type": StreamEvent.PROCESS, "content": "\n📋 [上下文已压缩，历史对话已保存]\n"}
            # 使用切片赋值更新 session 中的实际列表（避免仅修改局部变量）
            messages[:] = compact_result

        if error_count >= 10:
            yield {"type": StreamEvent.ANSWER, "content": '\n\n⚠️ 错误次数过多，停止运行\n'}
            return
        try:
            human_perception = get_human_perception_as_message_fmt(session_id)
            if human_perception:
                input_messages = [
                    human_perception,
                    *active_messages
                ]
            else:
                input_messages = active_messages

            with client.messages_stream(
                    model=MODEL,
                    system=dynamic_system,
                    messages=input_messages,
                    tools=TOOLS,
                    max_tokens=128000,
                    thinking={
                        "type": os.getenv("LLM_THINKING_TYPE", "disabled"),
                        "budget_tokens": int(os.getenv("budget_tokens", 4096)),
                    },
            ) as stream:
                for event in stream:
                    if event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            yield {"type": StreamEvent.TEXT, "content": delta.text}

                    elif event.type == "content_block_start":
                        if event.content_block.type == "tool_use":
                            # 工具调用确定是过程
                            pass

                final_response = stream.get_final_message()

            response_content_blocks = list(final_response.content)
            active_messages.append({"role": "assistant", "content": response_content_blocks})

            # 只有在非 btw 模式下才写入 memory
            if not btw_mode:
                # 写入 assistant 消息到 memory
                append_message(session_id, "assistant", response_content_blocks)

            if final_response.stop_reason != "tool_use":
                LOG.info(f'agent stop loop, stop_reason={final_response.stop_reason}, usage_tokens: {usage_tokens(final_response)}')
                destroy_sandbox(session_id)
                return

            # 还有工具调用，执行工具
            tool_use_blocks = [b for b in response_content_blocks if b.type == "tool_use"]
            parallel_enabled = os.getenv("PARALLEL_TOOL_CALLS", "true").lower() == "true"

            def _exec_tool(block):
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    output = handler(**block.input, session_id=session_id) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    LOG.exception(f"tool '{block.name}' raised: {e}")
                    output = f"Tool '{block.name}' execution error: {e}"
                if not isinstance(output, str):
                    output = str(output)
                return output

            def _emit_result(block, output, tag=""):
                LOG.info(f">>>tool_call: {block.name}[{block.input}] → {output[:200].replace(chr(10), '. ')}")
                return {"type": StreamEvent.PROCESS,
                        "content": f"\n👉🏻 工具结果{tag}: {output[:200].replace(chr(10), ' ')}{'...' if len(output) > 200 else ''}\n"}

            results = []
            # 按原顺序处理，把"连续的并行安全工具"打包成一组并发执行；
            # 遇到非并行安全工具则立即串行执行，保证副作用顺序不变。
            i = 0
            n = len(tool_use_blocks)
            while i < n:
                block = tool_use_blocks[i]
                is_safe = parallel_enabled and block.name in PARALLEL_SAFE_TOOLS

                if is_safe:
                    # 收集连续的并行安全工具
                    batch = []
                    while i < n and tool_use_blocks[i].name in PARALLEL_SAFE_TOOLS:
                        batch.append(tool_use_blocks[i])
                        i += 1

                    if len(batch) == 1:
                        # 只有 1 个，没必要起线程池
                        b = batch[0]
                        yield {"type": StreamEvent.PROCESS, "content": f"\n🔧 调用工具: {b.name} → {b.input}"}
                        output = _exec_tool(b)
                        yield _emit_result(b, output)
                        results.append({"type": "tool_result", "tool_use_id": b.id, "content": output})
                    else:
                        for b in batch:
                            yield {"type": StreamEvent.PROCESS, "content": f"\n🔧 调用工具(并行): {b.name} → {b.input}"}
                        pool_size = min(len(batch), 8)
                        with ThreadPoolExecutor(max_workers=pool_size) as ex:
                            futures = [ex.submit(_exec_tool, b) for b in batch]
                            outputs = [f.result() for f in futures]
                        for b, output in zip(batch, outputs):
                            yield _emit_result(b, output)
                            results.append({"type": "tool_result", "tool_use_id": b.id, "content": output})
                else:
                    # 非并行安全工具：串行执行
                    yield {"type": StreamEvent.PROCESS, "content": f"\n🔧 调用工具: {block.name} → {block.input}"}
                    output = _exec_tool(block)
                    yield _emit_result(block, output)
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
                    i += 1

            # 任务管理工具执行后会改变任务图状态，通知 UI 刷新任务进度面板
            if any(b.name in TASK_TOOLS for b in tool_use_blocks):
                yield {"type": StreamEvent.TASK, "content": ""}

            # 优化一：对过大的 tool_result 进行截断（写入 memory 前）
            optimized_results = optimize_tool_results_for_memory(results)

            active_messages.append({"role": "user", "content": optimized_results})

            # 只有在非 btw 模式下才写入 tool_result 到 memory
            if not btw_mode:
                append_message(session_id, "user", optimized_results)
            error_count = 0

        except (anthropic.RateLimitError, anthropic.APITimeoutError) as e:
            error_count += 1
            LOG.exception(f'anthropic api error: {e}')
            yield {"type": StreamEvent.PROCESS, "content": f"\n⚠️ API限流或超时，重试中... ({error_count}/10)\n"}
            time.sleep(10)
        except BaseException as e:
            LOG.exception(f'agent loop error: {e}')
            if 'maximum context window' in str(e):
                error_count += 2
                yield {"type": StreamEvent.PROCESS, "content": f"\n❌ 上下文窗口已满，压缩后重试...({error_count}/10)\n"}
                smart_compact(active_messages, session_id, force=True)
            else:
                destroy_sandbox(session_id)
                yield {"type": StreamEvent.PROCESS, "content": f"\n❌ 错误: {str(e)}\n"}
                return
        except IndexError as e:
            # SDK streaming 解析错误，通常是模型返回了空内容块
            error_count += 1
            LOG.exception(f"SDK streaming parse error (empty content block): {e}")
            yield {"type": StreamEvent.PROCESS, "content": f"\n⚠️ 模型返回了空响应，正在重试... ({error_count}/10)\n"}
            time.sleep(2)
        except Exception as e:
            error_count += 2
            LOG.exception(f"execute error: {e}")
            destroy_sandbox(session_id)
            yield {"type": StreamEvent.PROCESS, "content": f"\n❌ 错误: {str(e)}\n"}
            return
