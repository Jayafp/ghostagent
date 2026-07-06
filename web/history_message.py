from app.llm.memory_manager import load_recent_messages
from app.llm.utils import is_pure_user_message
from app.log.logger import LOG
from web.render import render_segments

priority = {'thinking': 0, 'text': 1}


def _append_segment(segments, kind, text):
    """把一段文本归入段落：与末段同类型则合并，否则新建。"""
    if segments and segments[-1]["kind"] == kind:
        segments[-1]["text"] += text
    else:
        segments.append({"kind": kind, "text": text})


def _blocks_to_segments(content, segments):
    """把一条消息的 content 块列表追加进 segments（thinking 不回显）。

    块内按 priority 排序，保证 text 在 tool_use 之前（与实时流式输出顺序一致）。
    """
    blocks = sorted(content, key=lambda x: priority.get(x.get("type", ""), 2))
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")
        if btype == "thinking":
            continue
        elif btype == "text":
            _append_segment(segments, "text", block.get("text", "") + "\n")
        elif btype == "tool_use":
            _append_segment(segments, "process", f"🔧 调用工具: {block.get('input')}\n")
        elif btype == "tool_result":
            tr = block.get("content", "")
            if not isinstance(tr, str):
                tr = str(tr)
            if len(tr) > 200:
                tr = tr[:200].replace("\n", " ") + "..."
            else:
                tr = tr.replace("\n", " ")
            _append_segment(segments, "process", f"👉🏻 工具结果:{tr}\n")


def format_history_from_memory(session_id: str, rounds: int = 10) -> list:
    """
    从 memory 加载历史对话并格式化为 Gradio Chatbot 的 messages 格式。

    每个用户回合渲染为一条 assistant 消息：工具过程(tool_use/tool_result)用代码块、
    正文(text)用 Markdown，按出现顺序交错——复用 render_segments，避免正文被困在
    代码块内（与实时流式 chat() 的渲染保持一致）。thinking 不回显。

    Args:
        session_id: 会话唯一标识符
        rounds: 要加载的轮数（每轮 = user + 后续 assistant/tool_result 直到下一个纯 user）

    Returns:
        list: [{"role": "user"|"assistant", "content": str}, ...]
    """
    try:
        messages = load_recent_messages(session_id, rounds=rounds)
        if not messages:
            return []

        history = []
        user_msg = None
        segments = []

        def flush():
            """收尾当前回合：写入 user + assistant(渲染后) 并重置"""
            nonlocal user_msg, segments
            if user_msg is not None:
                history.append({"role": "user", "content": user_msg})
                history.append({"role": "assistant", "content": render_segments(segments)})
            user_msg = None
            segments = []

        for msg in messages:
            if is_pure_user_message(msg):
                # 新回合开始：先收尾上一回合
                flush()
                user_msg = msg.get("content", "")
                continue

            content = msg.get("content", "")
            if isinstance(content, list):
                _blocks_to_segments(content, segments)

        flush()
        return history
    except Exception as e:
        LOG.exception(f"加载历史信息失败, {e}")
        return []


if __name__ == "__main__":
    session_id = "main"
    print(format_history_from_memory(session_id))
