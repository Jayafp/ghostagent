import os
import subprocess
import asyncio
import threading
from pathlib import Path
from typing import Optional
from app.skill.skill_manager import SKILL_MANAGER
from app.browser.browser_fetch import fetch_content
from app.browser.web_search import search
from app.log.logger import LOG
from app.llm.memory_retrieval import get_retriever
from app.sandbox.config import SandboxConfig
from app.sandbox.container_manager import ContainerManager
from app.sandbox.sandbox_executor import SandboxExecutor

# 每个线程独立的 event loop
_local = threading.local()


def _get_event_loop() -> asyncio.AbstractEventLoop:
    """
    获取或创建当前线程的事件循环

    为每个线程维护独立的 event loop，避免线程安全问题。
    如果当前线程没有 event loop，或 loop 已关闭，则创建新的。

    Returns:
        asyncio.AbstractEventLoop: 当前线程的事件循环
    """
    if not hasattr(_local, 'loop') or _local.loop is None:
        try:
            _local.loop = asyncio.get_event_loop()
            if _local.loop.is_closed():
                _local.loop = asyncio.new_event_loop()
        except RuntimeError:
            _local.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_local.loop)
    return _local.loop


def _run_async(coro) -> any:
    """
    线程安全的异步执行器

    在当前线程的 event loop 中运行协程

    Args:
        coro: 协程对象

    Returns:
        any: 协程的返回值
    """
    loop = _get_event_loop()
    return loop.run_until_complete(coro)


# 沙箱配置与容器管理（模块级单例）
_sandbox_config = SandboxConfig()
_container_manager = ContainerManager(_sandbox_config)
_container_manager.cleanup_stale()
_executor_cache: dict[str, SandboxExecutor] = {}  # session_id -> executor


def _get_sandbox_executor(session_id: str) -> SandboxExecutor:
    """获取或创建会话对应的沙箱执行器"""
    if session_id not in _executor_cache:
        _executor_cache[session_id] = SandboxExecutor(_container_manager, session_id)
    return _executor_cache[session_id]


def destroy_sandbox(session_id: str):
    """销毁会话对应的沙箱容器和执行器缓存"""
    _executor_cache.pop(session_id, None)
    _container_manager.destroy(session_id)


def safe_path(p: str) -> Path:
    """
    将字符串路径转换为 Path 对象

    Args:
        p: 路径字符串

    Returns:
        Path: Path 对象
    """
    return Path(p)


############# tool def #############

def run_bash(command: str) -> str:
    """
    执行 shell 命令

    安全性：
    - 检测并阻止危险命令（rm -rf /, sudo, shutdown 等）
    - 超时时间 120 秒
    - 最大返回 50000 字符

    Args:
        command: shell 命令字符串

    Returns:
        str: 命令执行结果（stdout + stderr）
               如果命令被阻止，返回错误信息
               如果超时，返回 "Error: Timeout (120s)"
    """
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int = None) -> str:
    """
    读取文件内容

    Args:
        path: 文件路径
        limit: 可选，限制读取的最大行数

    Returns:
        str: 文件内容
               如果设置了 limit 且文件行数超过 limit，会显示省略信息
               最多返回 50000 字符
               读取失败时返回错误信息
    """
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """
    将内容写入文件

    如果父目录不存在，会自动创建

    Args:
        path: 文件路径
        content: 要写入的内容

    Returns:
        str: 成功信息或错误信息
    """
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """
    在文件中替换文本

    使用精确匹配，只替换第一次出现的位置

    Args:
        path: 文件路径
        old_text: 要替换的旧文本
        new_text: 用于替换的新文本

    Returns:
        str: 成功信息或错误信息
               如果旧文本未找到，返回特定错误
    """
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def memory_search(user_message: str, session_id: str) -> str:
    """
    搜索历史记忆

    使用混合检索（BM25 + 向量），从 session 的历史记忆中检索与当前查询相关的内容

    Args:
        user_message: 用户当前的查询内容
        session_id: 会话唯一标识符

    Returns:
        str: 格式化的检索结果
               如果没有检索到相关内容，返回 "(未检索到相关历史记忆信息)"
               检索到结果时，返回格式化的历史信息列表
    """
    memories_section = "(未检索到相关历史记忆信息)"

    # 只有在有用户输入且 session 有历史记录时才检索
    if user_message and len(user_message.strip()) > 2:
        try:
            retriever = get_retriever(session_id)
            if retriever:
                results = retriever.search(
                    user_message,
                    top_k=int(os.getenv("MEMORY_TOP_K", "5")),
                    bm25_weight=0.5,
                    vector_weight=0.5
                )
                if results:
                    memories_section = retriever.format_for_prompt(results, max_length=800)
                    # 添加详细的日志信息
                    result_line = ""
                    for i, result in enumerate(results):
                        result_line += f'\n{i+1}. 内容: {result.content[:200].replace("\n", " ")}\n{result.bm25_score:.3f} | {result.vector_score:.3f} | {result.final_score:.3f}'
                    LOG.info(f"为用户查询检索到 {len(results)} 条相关记忆: {result_line}")
        except Exception as e:
            LOG.warning(f"检索历史记忆失败: {e}")
    return memories_section


def read_skill_resource(skill_name: str, resource_path: str) -> str:
    """
    读取 SKILL 目录下的资源文件

    Args:
        skill_name: SKILL 名称
        resource_path: 相对于 SKILL 目录的资源路径

    Returns:
        str: 资源文件内容
               如果 SKILL 不存在或资源不存在，返回错误信息
    """
    skills = SKILL_MANAGER.get_all_skils()
    if not skills or len(skills) == 0:
        return f"Skill({skill_name}) NOT found"

    skill = SKILL_MANAGER.get_skill(skill_name)
    if not skill:
        return f"Skill({skill_name}) NOT found"

    base_dir = skill.get("path", "")
    full_path = os.path.join(base_dir, resource_path)

    if not os.path.exists(full_path):
        return f"Resource not found in skill, full_path={full_path}"

    with open(full_path, 'r', encoding='utf-8') as f:
        return f.read()

############# tool def end #############


def create_tool_handler(
    tool_name: str,
    required_params: list[str],
    handler_func,
    optional_params: list[str] = None
):
    """
    创建带有错误处理的工具处理器包装器

    为工具调用添加统一的错误处理，包括：
    1. 必需参数验证
    2. 可选参数类型检查
    3. 异常捕获和结构化错误信息返回

    Args:
        tool_name: 工具名称，用于错误信息
        required_params: 必需参数列表
        handler_func: 实际的处理函数
        optional_params: 可选参数列表（用于日志记录）

    Returns:
        callable: 包装后的处理函数

    错误信息格式:
        [工具调用错误] <工具名>: <错误类型> - <详细说明>

        错误类型包括：
        - 缺少参数: 必需参数未提供
        - 参数类型错误: 参数类型不正确
        - 执行失败: 工具执行过程中发生异常
    """

    def wrapped(**kw):
        # 1. 验证必需参数
        missing_params = []
        invalid_params = []

        for param in required_params:
            if param not in kw:
                missing_params.append(param)
            elif kw[param] is None:
                missing_params.append(f"{param}(值为None)")
            # elif param in ["path", "url", "query", "command", "name", "skill_name", "resource_path", "old_text", "new_text", "content"]:
            #     # 这些参数必须是字符串类型
            #     if not isinstance(kw[param], str):
            #         invalid_params.append(f"{param}(期望字符串, 实际为{type(kw[param]).__name__})")
            #     elif kw[param].strip() == "":
            #         missing_params.append(f"{param}(空字符串)")

        if missing_params:
            error_msg = f"[工具调用错误] {tool_name}: 缺少必需参数 {missing_params}。请提供完整参数后重试。"
            LOG.warning(error_msg)
            return error_msg

        if invalid_params:
            error_msg = f"[工具调用错误] {tool_name}: 参数类型错误 {invalid_params}。请检查参数类型后重试。"
            LOG.warning(error_msg)
            return error_msg

        # 2. 执行工具并捕获异常
        try:
            result = handler_func(**kw)
            return result
        except FileNotFoundError as e:
            error_msg = f"[工具调用错误] {tool_name}: 文件或目录不存在 - {e}。请检查路径是否正确。"
            LOG.error(error_msg)
            return error_msg
        except PermissionError as e:
            error_msg = f"[工具调用错误] {tool_name}: 权限不足 - {e}。请检查文件权限。"
            LOG.error(error_msg)
            return error_msg
        except TimeoutError as e:
            error_msg = f"[工具调用错误] {tool_name}: 操作超时 - {e}。请尝试简化操作或增加超时时间。"
            LOG.error(error_msg)
            return error_msg
        except ValueError as e:
            error_msg = f"[工具调用错误] {tool_name}: 参数值无效 - {e}。请检查参数值是否合法。"
            LOG.error(error_msg)
            return error_msg
        except KeyError as e:
            error_msg = f"[工具调用错误] {tool_name}: 配置或数据缺失 - {e}。请检查相关配置。"
            LOG.error(error_msg)
            return error_msg
        except ImportError as e:
            error_msg = f"[工具调用错误] {tool_name}: 依赖模块缺失 - {e}。请安装所需依赖。"
            LOG.error(error_msg)
            return error_msg
        except ConnectionError as e:
            error_msg = f"[工具调用错误] {tool_name}: 网络连接失败 - {e}。请检查网络连接后重试。"
            LOG.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"[工具调用错误] {tool_name}: 执行失败 - {type(e).__name__}: {e}。请检查参数是否正确或稍后重试。"
            LOG.error(f"{error_msg}\n原始异常: ", exc_info=True)
            return error_msg

    return wrapped


# -- The dispatch map: {tool_name: handler} --
def _bash_handler(**kw):
    session_id = kw.get("session_id", "")
    if _sandbox_config.is_sandbox("bash"):
        return _get_sandbox_executor(session_id).run_bash(kw["command"])
    return run_bash(kw["command"])


def _read_handler(**kw):
    session_id = kw.get("session_id", "")
    if _sandbox_config.is_sandbox("read_file"):
        return _get_sandbox_executor(session_id).run_read(kw["path"], kw.get("limit"))
    return run_read(kw["path"], kw.get("limit"))


def _write_handler(**kw):
    session_id = kw.get("session_id", "")
    if _sandbox_config.is_sandbox("write_file"):
        return _get_sandbox_executor(session_id).run_write(kw["path"], kw["content"])
    return run_write(kw["path"], kw["content"])


def _edit_handler(**kw):
    session_id = kw.get("session_id", "")
    if _sandbox_config.is_sandbox("edit_file"):
        return _get_sandbox_executor(session_id).run_edit(kw["path"], kw["old_text"], kw["new_text"])
    return run_edit(kw["path"], kw["old_text"], kw["new_text"])


TOOL_HANDLERS = {
    "bash": create_tool_handler(
        "bash",
        ["command"],
        _bash_handler
    ),
    "read_file": create_tool_handler(
        "read_file",
        ["path"],
        _read_handler,
        ["limit"]
    ),
    "write_file": create_tool_handler(
        "write_file",
        ["path", "content"],
        _write_handler
    ),
    "edit_file": create_tool_handler(
        "edit_file",
        ["path", "old_text", "new_text"],
        _edit_handler
    ),
    "load_skill": create_tool_handler(
        "load_skill",
        ["name"],
        lambda **kw: SKILL_MANAGER.get_skill_content(kw["name"])
    ),
    "fallback_browser_fetch": create_tool_handler(
        "fallback_browser_fetch",
        ["url"],
        lambda **kw: _run_async(fetch_content(kw["url"]))
    ),
    "web_search": create_tool_handler(
        "web_search",
        ["query"],
        lambda **kw: _run_async(search(kw["query"]))
    ),
    "memory_search": create_tool_handler(
        "memory_search",
        ["user_message", "session_id"],
        lambda **kw: memory_search(kw["user_message"], kw["session_id"])
    ),
    "read_skill_resource": create_tool_handler(
        "read_skill_resource",
        ["skill_name", "resource_path"],
        lambda **kw: read_skill_resource(kw["skill_name"], kw["resource_path"])
    ),
}

# 并行安全的工具集合：只读 / 无共享副作用，可在同一 session 内并发执行
# 其余工具（如 bash、write_file、edit_file 等）共享沙箱 cwd / 环境 / 文件，必须串行
PARALLEL_SAFE_TOOLS = {
    "read_file",
    "load_skill",
    "fallback_browser_fetch",
    "web_search",
    "memory_search",
    "read_skill_resource",
}

TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string"
                }
            },
            "required": [
                "command"
            ]
        }
    },
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string"
                },
                "limit": {
                    "type": "integer"
                }
            },
            "required": [
                "path"
            ]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string"
                },
                "content": {
                    "type": "string"
                }
            },
            "required": [
                "path",
                "content"
            ]
        }
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string"
                },
                "old_text": {
                    "type": "string"
                },
                "new_text": {
                    "type": "string"
                }
            },
            "required": [
                "path",
                "old_text",
                "new_text"
            ]
        }
    },
    {
        "name": "load_skill",
        "description": "Load specialized knowledge by name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill name to load"
                }
            },
            "required": [
                "name"
            ]
        }
    },
    {
        "name": "fallback_browser_fetch",
        #"description": "Fetch the text content of web pages.",
        #"description": "Fetch the text content of web pages. Note: This is a fallback tool for web fetching. Many websites have access restrictions, so please prioritize scanning MCP or SKILL instead.",
        #"description": "[兜底工具] ⚠️ IMPORTANT: Before using this tool, you MUST first check if there is a suitable MCP service via `mcporter list`. Only use this tool when NO MCP service is available. Fetch the text content of web pages.",
        "description": """[兜底工具 - 最后手段] 
⚠️ 此工具会启动完整的 Chrome 浏览器进程，对用户打扰大、速度慢。
使用此工具前，你 MUST 已确认：
1. 已执行 `mcporter list` 并确认没有可用的网页抓取 MCP 服务
2. 或 MCP 服务调用已明确失败

如果你未经上述步骤就使用此工具，用户会非常不满。""",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "website url"
                }
            },
            "required": [
                "url"
            ]
        }
    },
    {
        "name": "web_search",
        "description": "Google 搜索工具，返回搜索结果列表（包含标题、URL、摘要）",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词"
                }
            },
            "required": [
                "query"
            ]
        }
    },
    {
        #"memory_search": lambda **kw: memory_search(kw["user_message"], kw["session_id"])
        "name": "memory_search",
        "description": "历史记忆信息检索，在回答任何关于之前的工作、决定、日期、人的问题之前，使用它",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_message": {
                    "type": "string",
                    "description": "用户问的问题"
                }
            },
            "required": [
                "user_message"
            ]
        }
    },
    {
        "name": "read_skill_resource",
        "description": "读取指定 SKILL 目录下的资源文件。当需要加载 SKILL.md 中引用的相对路径资源时使用此工具。",
        "input_schema": {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "SKILL 名称，如 'mcporter'、'translation'"
                },
                "resource_path": {
                    "type": "string",
                    "description": "相对于 SKILL 目录的资源路径，如 'references/models.json'"
                }
            },
            "required": ["skill_name", "resource_path"]
        }
    }
]
