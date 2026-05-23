from typing import List, Dict, Any, Tuple


def get_tool_use_id(block: Dict) -> str:
    """
    从工具结果块中提取 tool_use_id

    Args:
        block: 工具结果块字典，包含 tool_use_id 等信息

    Returns:
        str: tool_use_id 字符串
        None: 如果 block 为空或不存在 tool_use_id
    """
    if not block:
        return None
    return block.get("tool_use_id", None)


def is_pure_user_message(message: Dict) -> bool:
    """
    判断是否为纯用户消息（新一轮对话的开始）

    纯用户消息的定义：
    - role 为 "user"
    - content 为字符串类型（不包含 tool_result）

    Args:
        message: 消息字典，包含 role 和 content 字段

    Returns:
        bool: 是否为纯用户消息

    Example:
        >>> is_pure_user_message({"role": "user", "content": "你好"})
        True
        >>> is_pure_user_message({"role": "user", "content": [{"type": "tool_result"}]})
        False
    """
    return message.get('role') == 'user' and isinstance(message.get('content'), str)


def is_answer_message(message: Dict) -> bool:
    """
    判断消息是否为对用户的最终回答（而非工具调用）

    Args:
        message: 消息字典，包含 content 字段

    Returns:
        bool: 是否为回答用户的消息
        False: 消息为空或包含工具调用/工具结果
    """
    content = message.get('content', None)
    if not content:
        return False
    return is_answer_content(content)


def is_answer_content(content: Any) -> bool:
    """
    判断内容块列表是否为回答用户的内容（不包含 tool_use 或 tool_result）

    Args:
        content: 内容块列表，每个块是包含 type 字段的字典

    Returns:
        bool: True - 纯回答内容（只包含 text/thinking）
              False - 包含工具调用或工具结果

    Note:
        如果内容不是列表，或包含 type 为 "tool_use" 或 "tool_result" 的块，
        则返回 False
    """
    if not content or not isinstance(content, list):
        return False

    for block in content:
        if not isinstance(block, dict):
            # 忽略非字典类型的块（正常情况下不应出现）
            continue
        if 'type' in block and (block['type'] == 'tool_use' or block['type'] == 'tool_result'):
            return False
    return True


def estimate_tokens(messages: List[Dict]) -> int:
    """
    估算消息列表的 token 数量

    使用粗略估算方法：每 4 个字符约 1 个 token
    这是 LLM tokenization 的经验估算值

    Args:
        messages: 消息字典列表

    Returns:
        int: 估算的 token 数量

    Note:
        此估算为粗略值，实际 token 数量可能因模型和 tokenizer 不同而有差异
        用于上下文长度管理和压缩决策，不需要精确值
    """
    return len(str(messages)) // 4


def usage_tokens(message) -> str:
    try:
        if not message:
            return ""

        usage = getattr(message, 'usage')
        if not usage:
            return ""

        cache_read_input_tokens = getattr(usage, 'cache_read_input_tokens', 0)
        input_tokens = getattr(usage, 'input_tokens', 0)
        output_tokens = getattr(usage, 'output_tokens', 0)

        return f'cache_tokens={cache_read_input_tokens}, input_tokens={input_tokens}, output_tokens={output_tokens}'
    except:
        return ""