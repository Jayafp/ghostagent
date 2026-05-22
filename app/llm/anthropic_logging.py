import datetime
import json
from typing import Generator, Any, List, Dict, Union

from anthropic import Anthropic
from app.log.logger import LLM_LOG


class LoggingAnthropic(Anthropic):
    """
    带日志记录的 Anthropic 客户端包装类

    功能：
    1. 包装标准 messages.create 调用，自动记录输入输出到日志
    2. 包装流式 messages.stream 调用，自动记录输入到日志
    3. 序列化消息内容，使其可 JSON 序列化

    继承自 Anthropic 类，保持原有的所有功能

    Example:
        client = LoggingAnthropic(api_key="your_key")

        # 标准调用
        response = client.messages_create(model="claude-3", messages=[...])

        # 流式调用
        with client.messages_stream(model="claude-3", messages=[...]) as stream:
            for event in stream:
                ...
    """

    def messages_create(self, *args, **kwargs):
        """
        创建消息（非流式），并记录输入输出到日志

        Args:
            *args: 传递给 Anthropic.messages.create 的位置参数
            **kwargs: 传递给 Anthropic.messages.create 的关键字参数
                       通常包含：model, messages, max_tokens, system, tools 等

        Returns:
            Message: Anthropic Message 对象，包含生成的内容和元信息

        Note:
            会自动将 messages 参数序列化并记录到 LLM_LOG（DEBUG级别）
        """
        if 'messages' in kwargs:
            msgs = kwargs['messages']
            LLM_LOG.debug(f">>>Input: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                         f"{json.dumps(serialize_messages(kwargs['messages']), indent=2, ensure_ascii=False)}")

        response = super().messages.create(*args, **kwargs)

        LLM_LOG.debug(f"<<< Response: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                     f"{json.dumps(serialize_content(response), indent=2, ensure_ascii=False)}")

        return response

    def messages_stream(self, *args, **kwargs):
        """
        创建流式消息，并记录输入到日志

        Args:
            *args: 传递给 Anthropic.messages.stream 的位置参数
            **kwargs: 传递给 Anthropic.messages.stream 的关键字参数
                       通常包含：model, messages, max_tokens, system, tools 等

        Returns:
            MessageStreamManager: 流式响应上下文管理器

        Usage:
            with client.messages_stream(model="claude-3", messages=[...]) as stream:
                for event in stream:
                    if event.type == "content_block_delta":
                        print(event.delta.text)

        Note:
            仅记录输入消息，不记录流式输出（由调用方处理）
        """
        if 'messages' in kwargs:
            LLM_LOG.debug(f">>>Stream Input: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                         f"{json.dumps(serialize_messages(kwargs['messages']), indent=2, ensure_ascii=False)}")

        # 直接返回原始的 stream 上下文管理器
        return super().messages.stream(*args, **kwargs)


def serialize_content(content: Any) -> Union[str, List, Dict]:
    """
    递归序列化内容对象，使其可 JSON 序列化

    Args:
        content: 内容对象，可能是以下类型：
            - str: 直接返回
            - list: 递归序列化每个元素
            - dict: 递归序列化每个值
            - Pydantic model (有 model_dump 方法): 调用 model_dump()
            - 其他: 转为字符串

    Returns:
        Union[str, List, Dict]: 可 JSON 序列化的对象

    Example:
        >>> serialize_content([{"type": "text", "text": "hello"}])
        [{"type": "text", "text": "hello"}]
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return [
            item.model_dump() if hasattr(item, 'model_dump')
            else serialize_content(item)
            for item in content
        ]
    elif isinstance(content, dict):
        return {
            k: serialize_content(v) if hasattr(v, 'model_dump') else v
            for k, v in content.items()
        }
    elif hasattr(content, 'model_dump'):
        return content.model_dump()
    else:
        return str(content)


def serialize_messages(messages: List[Dict]) -> List[Dict]:
    """
    序列化消息列表为可 JSON 序列化的格式

    Args:
        messages: 消息字典列表，每个字典包含：
            - role: str, 消息角色（"user"/"assistant"）
            - content: Any, 消息内容（字符串或内容块列表）

    Returns:
        List[Dict]: 序列化后的消息列表

    Example:
        >>> msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        >>> serialize_messages(msgs)
        [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    """
    return [
        {"role": msg["role"], "content": serialize_content(msg["content"])}
        for msg in messages
    ]


def resp_to_dict(response: Any) -> Dict:
    """
    将 Anthropic Response 对象转换为可序列化的字典

    Args:
        response: Anthropic Message 对象，包含以下属性：
            - id: str, 消息唯一标识
            - model: str, 使用的模型
            - role: str, 角色
            - stop_reason: str, 停止原因
            - stop_sequence: str, 停止序列
            - usage: Usage 对象，包含 input_tokens/output_tokens
            - content: List[ContentBlock], 内容块列表

    Returns:
        Dict: 包含完整响应信息的字典
            {
                "id": str,
                "model": str,
                "role": str,
                "stop_reason": str,
                "stop_sequence": str,
                "usage": {"input_tokens": int, "output_tokens": int},
                "content": [{"type": str, ...}]
            }

    Note:
        支持的内容块类型：text, thinking, tool_use
    """
    return {
        "id": response.id,
        "model": response.model,
        "role": response.role,
        "stop_reason": response.stop_reason,
        "stop_sequence": response.stop_sequence,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
        "content": [
            {
                "type": block.type,
                **(
                    {"text": block.text} if block.type == "text"
                    else {"thinking": getattr(block, 'thinking', None)} if block.type == "thinking"
                    else {
                        "id": block.id,
                        "name": block.name,
                        "input": block.input
                    }
                )
            }
            for block in response.content
        ]
    }
