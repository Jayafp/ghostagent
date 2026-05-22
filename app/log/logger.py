import sys
import os
from loguru import logger


def llm_filter(record: dict) -> bool:
    """
    过滤器：只保留 LLM 相关的日志

    Args:
        record: 日志记录字典

    Returns:
        bool: True 如果记录是 LLM 日志（extra.name == "llm"）
    """
    return record["extra"].get("name") == "llm"


def log_filter(record: dict) -> bool:
    """
    过滤器：排除 LLM 相关的日志

    Args:
        record: 日志记录字典

    Returns:
        bool: True 如果记录不是 LLM 日志
    """
    return record["extra"].get("name") != "llm"


# 移除 loguru 默认的处理器
logger.remove()

# 从环境变量读取日志级别，默认 INFO
DEFAULT_LOG_LEVEL = os.getenv('LOGURU_LEVEL', 'INFO')

# 默认日志记录器（非 LLM）
LOG = logger.bind(name="default")

# LLM 专用日志记录器
LLM_LOG = logger.bind(name="llm")

# 添加标准输出处理器（排除 LLM 日志）
logger.add(
    sys.stdout,
    level=f"{DEFAULT_LOG_LEVEL}",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    filter=log_filter
)

# 添加 LLM 专用文件处理器
logger.add(
    os.getenv('log_dir') + "/llm_log_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    rotation="00:00",  # 每天轮转
    retention="7 days",  # 保留7天
    compression="zip",  # 压缩旧日志
    encoding="utf-8",
    filter=llm_filter
)
