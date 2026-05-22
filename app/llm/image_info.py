import base64
import os
from pathlib import Path
from typing import Optional

from anthropic import Anthropic

image_model_id = os.getenv("IMAGE_MODEL_ID", "kimi-k2.5")
client = Anthropic(
    api_key=os.getenv('ANTHROPIC_API_KEY'),
    base_url=os.getenv('ANTHROPIC_BASE_URL'),
)


def get_media_type(image_path: str) -> str:
    """
    根据文件扩展名返回对应的 MIME type

    Args:
        image_path: 图片文件路径

    Returns:
        str: MIME type 字符串
            - .png → image/png
            - .jpg/.jpeg → image/jpeg
            - .webp → image/webp
            - .gif → image/gif
            - 其他 → image/jpeg（默认值）
    """
    ext = Path(image_path).suffix.lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.webp': 'image/webp',
        '.gif': 'image/gif',
    }
    return mime_types.get(ext, 'image/jpeg')  # 默认 jpeg


def get_image_info(image_path: str) -> Optional[str]:
    """
    分析图片并返回描述文本

    使用多模态 LLM 分析图片内容并生成描述

    Args:
        image_path: 图片文件路径

    Returns:
        str: 图片内容描述
        None: 如果分析失败或无有效返回

    Implementation:
        1. 将图片转为 base64 编码
        2. 调用多模态模型（如 kimi-k2.5）
        3. 返回模型生成的描述

    Note:
        使用 IMAGE_MODEL_ID 环境变量指定的模型，默认 kimi-k2.5
        支持格式：png, jpg, jpeg, webp, gif
    """
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    message = client.messages.create(
        model=image_model_id,
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": get_media_type(image_path),
                            "data": image_data,
                        }
                    },
                    {"type": "text", "text": "请描述这张图片的内容。"}
                ]
            }
        ]
    )

    for msg in message.content:
        if msg.type == "text" and hasattr(msg, "text"):
            return msg.text
    return None