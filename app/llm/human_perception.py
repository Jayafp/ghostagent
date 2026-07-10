import os
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

from anthropic import Anthropic

from app.llm.context_compact import format_message_for_summary
from app.log.logger import LOG
from app.llm.utils import is_answer_content
from pathlib import Path

from app.llm.memory_manager import get_session_memory_dir, list_session_memories

# 文件操作锁，保证线程安全
FILE_LOCK = threading.Lock()

# 一次最多加载的消息长度（字符数）
MAX_MESSAGE_LENGHT = 200000

# 最小生成轮数阈值，小于此轮数不生成感知摘要
MIN_ROUNDS = 30

# 生成间隔时间（秒），默认 24 小时
INTERVAL_TIME = 24 * 3600

# 线程池执行器，用于异步生成摘要
executor = ThreadPoolExecutor(max_workers=3)

# 摘要生成使用的模型
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "GLM-5")

# Anthropic API 配置
api_key = os.getenv("ANTHROPIC_API_KEY")
base_url = os.getenv("ANTHROPIC_BASE_URL")
client = Anthropic(api_key=api_key, base_url=base_url)

# 根据 session_id 缓存人类感知结果（内存缓存）
_human_perceptions: Dict[str, Dict] = {}

# 根据 session_id 标记是否正在生成（防止重复生成）
_generate_is_running: Dict[str, bool] = {}

def write_to_disk(perception_result: Dict) -> None:
    """
    将感知摘要结果写入磁盘

    功能：
    1. 将结果保存为 JSON 文件到 session 的 memory 目录
    2. 清除内存缓存，强制下次从磁盘读取

    Args:
        perception_result: 感知结果字典，必须包含 session_id 字段
            {
                "session_id": str,
                "summary": str,
                "date": str,
                "start_line": int,
                "generate_time": int,
                ...
            }

    Returns:
        None

    Note:
        使用 FILE_LOCK 保证线程安全
        文件保存路径：{MEMORY_DIR}/{session_id}/perception.json
    """
    with FILE_LOCK:
        session_id = perception_result.get('session_id', None)
        if not session_id:
            LOG.error(f'session_id not found')
            return

        memdir = get_session_memory_dir(session_id)
        perception_file: Path = memdir / 'perception.json'

        with open(perception_file, 'w', encoding='utf-8') as f:
            json.dump(perception_result, f, ensure_ascii=False, indent=4)

        # 清除内存缓存
        if _human_perceptions.get(session_id, None):
            del _human_perceptions[session_id]

    LOG.info(f'write to disk success, session_id={session_id}, '
             f'perception_result={perception_result}')


def read_from_disk(session_id: str) -> Optional[Dict]:
    """
    从磁盘读取感知摘要结果

    Args:
        session_id: 会话唯一标识符

    Returns:
        Dict: 感知结果字典，包含以下字段：
            - session_id: str
            - summary: str, 摘要内容
            - date: str, 最后处理的日期
            - start_line: int, 最后处理的行号
            - generate_time: int, 生成时间戳
        None: 如果文件不存在

    Note:
        文件路径：{MEMORY_DIR}/{session_id}/perception.json
    """
    with FILE_LOCK:
        memdir = get_session_memory_dir(session_id)
        perception_file: Path = memdir / 'perception.json'

        if not perception_file.exists():
            return None

        with open(perception_file, 'r', encoding='utf-8') as f:
            return json.load(f)


def get_human_perception(session_id: str) -> Optional[Dict]:
    """
    获取人类感知摘要（优先从内存缓存）

    查找顺序：
    1. 内存缓存 (_human_perceptions)
    2. 磁盘文件 (perception.json)

    Args:
        session_id: 会话唯一标识符

    Returns:
        Dict: 感知结果字典
        None: 如果缓存和磁盘都不存在
    """
    # 先查内存缓存
    if session_id in _human_perceptions:
        return _human_perceptions[session_id]

    # 再查磁盘
    perception_result = read_from_disk(session_id)
    if perception_result:
        _human_perceptions[session_id] = perception_result
        return perception_result

    return None


def get_human_perception_as_message_fmt(session_id: str) -> Optional[Dict]:
    """
    获取人类感知摘要，格式化为 Anthropic 消息格式

    将感知摘要包装为 assistant 角色的消息，可以插入到对话上下文中

    Args:
        session_id: 会话唯一标识符

    Returns:
        Dict: Anthropic 消息格式
            {
                "role": "assistant",
                "content": str
            }
        None: 如果不存在感知摘要

    Usage:
        >>> perception_msg = get_human_perception_as_message_fmt("user_123")
        >>> if perception_msg:
        ...     messages.insert(0, perception_msg)
    """
    human_perception = get_human_perception(session_id)
    if not human_perception:
        return None

    per_summary = human_perception.get("summary", None)
    if not per_summary:
        return None

    return {
        'role': 'assistant',
        'content': f'这是基于AI和用户历史的对话记录，总结的摘要信息:\n{per_summary}'
    }


def get_memory_messages(session_id: str, date: str = None, start_line: int = 0) -> Optional[Dict]:
    """
    从 memory 文件中加载指定范围内的消息

    用于增量加载历史消息，支持分页读取（基于文件和行号）

    Args:
        session_id: 会话唯一标识符
        date: 起始日期（格式：YYYY-MM-DD），None 表示从最早文件开始
        start_line: 起始行号（在第一个文件中跳过的行数）

    Returns:
        Dict: 成功时返回消息包
            {
                "message": str,              # 格式化的消息内容
                "org_message_len": int,      # 原始消息长度（字符数）
                "date": str,                 # 最后读取的日期
                "start_line": int,           # 下次读取的起始行号
                "has_next": bool,            # 是否还有更多消息
                "rounds": int                # 包含的对话轮数
            }
        None: 如果找不到起始文件

    Processing Flow:
        1. 根据 date 找到起始文件
        2. 从 start_line 开始读取
        3. 累积消息直到达到 MAX_MESSAGE_LENGHT 或文件末尾
        4. 如果遇到完整的对话轮次（is_answer_content），可以截断
        5. 继续读取下一个文件直到条件满足或没有更多文件
    """
    jsonl_files = list_session_memories(session_id)

    date_file = f'{date}.jsonl' if date else jsonl_files[0].name

    # 找到起始文件索引
    start_file_idx = -1
    for i, file in enumerate(jsonl_files):
        if file.name >= date_file:
            start_file_idx = i
            break

    if start_file_idx == -1:
        LOG.warning(f"记忆文件不存在, session_id={session_id}, date={date}")
        return None

    message = ""
    message_len = 0
    read_line = start_line
    has_next = True
    rounds = 0

    while True:
        with open(jsonl_files[start_file_idx], 'r', encoding="utf-8") as f:
            for i, line in enumerate(f):
                # 跳过起始行之前的行
                if i < start_line:
                    continue

                # 读取完成后重置 start_line
                start_line = -1
                read_line = i

                if not line or len(line) == 0:
                    continue

                msg_dict = json.loads(line)
                fmt_msg = format_message_for_summary(msg_dict)

                message += fmt_msg + "\n"
                message_len += len(fmt_msg)

                # 检查是否是完整的回答轮次
                is_ans = is_answer_content(msg_dict['content'])
                if is_ans:
                    rounds += 1

                # 如果达到长度限制且是完整轮次，可以截断返回
                if message_len >= MAX_MESSAGE_LENGHT and is_ans:
                    return {
                        'message': message,
                        'org_message_len': len(message),
                        'date': jsonl_files[start_file_idx].name.split('.')[0],
                        'start_line': i + 1,
                        'has_next': True,
                        'rounds': rounds,
                    }

        # 移动到下一个文件
        start_file_idx += 1
        if start_file_idx >= len(jsonl_files):
            has_next = False
            break

    return {
        'message': message,
        'org_message_len': len(message),
        'date': jsonl_files[-1].name.split('.')[0],
        'start_line': read_line + 1,
        'has_next': has_next,
        'rounds': rounds,
    }



def generate_summary(history_messages: str) -> Optional[str]:
    """
    使用 LLM 生成对话历史摘要

    将历史对话内容总结为结构化的摘要，包含：
    - 讨论的主题和决策
    - 用户画像（职业、偏好、行为模式等）
    - 当前状态和待办事项

    Args:
        history_messages: 格式化的历史消息字符串

    Returns:
        str: 生成的摘要文本
        None: 如果生成失败

    Note:
        使用 SUMMARY_MODEL 配置的模型进行生成
        temperature=0.3 降低随机性
    """
    summary_prompt = f"""你是一个对话总结专家。请将以下对话内容总结成一段简洁的摘要。

**要求：**
1. 总结必须包含以下三个方面：
   - 聚焦事实性内容：讨论了什么、做出了哪些决策、当前状态如何。
   - 基于对话内容，理解跟你对话用户的画像，包含：
     * 基础属性：职业、背景、专业领域（如有提及）
     * 偏好特征：爱好、语言偏好、沟通风格、表达方式等
     * 行为模式：提问方式、决策风格、对新事物的接受度、 collaboration习惯
     * 需求特点：频繁出现的任务类型、关注重点、痛点或挑战
     * 人际关系：提及的重要他人（同事、上级、家人等）及其角色
   - 保持必需的摘要结构和章节标题不变。

2. 保持客观，不要添加对话中没有的信息。

3. 保留所有不透明标识符的原始写法（不得缩短或重构），包括 UUID、哈希值、ID、令牌、API 密钥、主机名、IP 地址、端口、URL 和文件名。

**对话内容：**
{history_messages}

**请输出简洁的摘要（不超过800字）：**"""

    # 调用模型生成摘要
    response = client.messages.create(
        model=SUMMARY_MODEL,
        max_tokens=2048,
        system="你是一个专业的对话总结助手，擅长提取关键信息并生成结构化的摘要。",
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0.3  # 降低随机性，保持摘要的稳定性
    )

    summary = None
    for block in response.content:
        if block and hasattr(block, 'text'):
            summary = block.text
            break

    return summary



def merge_summary(prev_summary: str, new_summary: str) -> Optional[str]:
    """
    合并两个摘要（旧摘要 + 新摘要）

    将早期的历史摘要与近期的摘要合并为一个连贯的摘要

    合并规则：
    1. 如果内容冲突，以 new_summary（近期）为准
    2. 保持 new_summary 的格式风格
    3. 保留用户画像等长期信息

    Args:
        prev_summary: 之前的摘要（较早的历史）
        new_summary: 新的摘要（近期的历史）

    Returns:
        str: 合并后的摘要
        None: 如果合并失败

    Note:
        使用 SUMMARY_MODEL 配置的模型进行合并
    """
    merge_summary_prompt = f"""你是一个摘要信息整合专家，请将以下 2 个历史摘要信息合并成一个连贯的摘要。

**要求：**

1. **摘要内容一** 是更早的摘要信息，如果和 **摘要内容二** 有冲突的部分，以 **摘要内容二** 的信息为准。

2. 新的摘要信息保持 **摘要内容二** 相似的内容格式。

3. 必须保留：
   - 讨论了什么、做出了哪些决策、有哪些待办事项、当前状态如何、
   - 基于对话内容，理解跟你对话用户的画像，包含：
     * 基础属性：职业、背景、专业领域（如有提及）
     * 偏好特征：爱好、语言偏好、沟通风格、表达方式等
     * 行为模式：提问方式、决策风格、对新事物的接受度、 collaboration习惯
     * 需求特点：频繁出现的任务类型、关注重点、痛点或挑战
     * 人际关系：提及的重要他人（同事、上级、家人等）及其角色

**摘要内容一**
{prev_summary}

**摘要内容二**
{new_summary}

**请输出简洁的摘要（不超过800字）：**"""

    # 调用模型生成摘要
    response = client.messages.create(
        model=SUMMARY_MODEL,
        max_tokens=2048,
        system="你是一个专业的对话总结助手，擅长提取关键信息并生成结构化的摘要。",
        messages=[{"role": "user", "content": merge_summary_prompt}],
        temperature=0.3  # 降低随机性，保持摘要的稳定性
    )

    summary = None
    for block in response.content:
        if block and hasattr(block, 'text'):
            summary = block.text
            break

    return summary


def generate_perception(session_id: str, date: str = None, start_line: int = 0) -> Dict:
    """
    生成用户感知摘要

    增量式生成感知摘要：
    1. 读取之前保存的摘要（如果存在）
    2. 加载指定范围内的历史消息
    3. 如果消息量达标，生成新摘要
    4. 如果存在旧摘要，合并新旧摘要
    5. 保存到磁盘

    Args:
        session_id: 会话唯一标识符
        date: 起始日期，None 表示从最早开始
        start_line: 起始行号

    Returns:
        Dict: 生成结果
            {
                "success": bool,           # 是否成功生成
                "session_id": str,         # 会话ID
                "summary": str,            # 摘要内容（成功时）
                "date": str,               # 处理到的日期
                "start_line": int,         # 处理到的行号
                "generate_time": int       # 生成时间戳
            }

    Processing Flow:
        1. 读取之前的感知结果
        2. 循环读取历史消息（分页）
        3. 检查消息量是否满足 MIN_ROUNDS 且足够长
        4. 生成新摘要，与旧摘要合并
        5. 保存到磁盘
        6. 继续处理下一批（has_next）
    """
    LOG.info(f'generate_perception start, session_id={session_id}')

    prev_summary = None
    prev_perception = read_from_disk(session_id)
    if prev_perception:
        prev_summary = prev_perception.get('summary', None)
        LOG.info(f'read prev perception success, session_id={session_id}')

    result = {'success': False}
    has_next = True
    while has_next:
        history_messages = get_memory_messages(session_id, date, start_line)
        if not history_messages:
            return result

        rounds = history_messages.get('rounds', 0)
        org_message_len = history_messages.get('org_message_len', 0)
        if rounds <= MIN_ROUNDS and org_message_len < MAX_MESSAGE_LENGHT:
            LOG.debug(f"rounds too shot, skip generate perception. rounds={rounds}, org_message_len={org_message_len}")
            return result

        new_summary = generate_summary(history_messages['message'])
        LOG.info(f"generate summary success, session_id={session_id}, date={date}, start_line={start_line}")
        if not new_summary:
            LOG.error(f'generate summary failed, session_id={session_id}')
            return result

        summary = new_summary
        if prev_summary:
            # merge summary
            summary = merge_summary(prev_summary, new_summary)
            LOG.info(f"merge summary success, session_id={session_id}, date={date}, start_line={start_line}, "
                     f"prev_summary={prev_summary[:100]}, new_summary={new_summary[:100]}")

        if not summary:
            LOG.error(f'merge summary failed, session_id={session_id}')
            return result

        result = {
            'success': True,
            'session_id': session_id,
            'summary': summary,
            'date': history_messages['date'],
            'start_line': history_messages['start_line'],
            'generate_time': int(time.time()),
        }
        write_to_disk(result)
        prev_summary = summary

        has_next = history_messages.get('has_next', False)
        if has_next:
            date = history_messages.get('date')
            start_line = history_messages.get('start_line')

        LOG.info(f'generate_perception end, session_id={session_id}')
    return result



def async_generate_perception_ifneed(session_id: str) -> None:
    """
    条件触发生成用户感知摘要（异步）

    检查是否需要生成新的感知摘要，如果需要则在后台线程中执行

    触发条件：
    1. 当前没有正在进行的生成任务
    2. 不存在之前的感知结果，或距离上次生成超过 INTERVAL_TIME

    Args:
        session_id: 会话唯一标识符

    Returns:
        None

    Note:
        - 使用线程池异步执行，不阻塞主线程
        - 使用 _generate_is_running 防止并发生成
        - 如果存在之前的感知结果，会在此基础上增量更新

    Processing Flow:
        1. 检查是否有进行中的生成任务
        2. 获取之前的感知结果
        3. 检查是否满足生成间隔
        4. 提交异步任务到线程池
        5. 任务完成后自动释放锁
    """
    date = None
    start_line = 0

    with FILE_LOCK:
        if _generate_is_running.get(session_id, False):
            LOG.debug(f'generate is running, skip generate_perception. session_id={session_id}')
            return

    try:
        _generate_is_running[session_id] = True

        human_perception = get_human_perception(session_id)
        if human_perception:
            genereate_time = human_perception.get('generate_time', 0)
            if int(time.time()) - genereate_time < INTERVAL_TIME:
                LOG.debug(f'距上一次生成时间太短, skip generate_perception. session_id={session_id}')
                return
            date = human_perception.get('date', None)
            start_line = human_perception.get('start_line', 0)

        # 提交异步任务到线程池
        executor.submit(generate_perception, session_id, date, start_line)

    finally:
        _generate_is_running[session_id] = False
        LOG.info(f'generate_perception end, session_id={session_id}')