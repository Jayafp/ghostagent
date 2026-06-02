import os
from app.log.logger import LOG

VALID_MODES = ("sandbox", "local")

SANDBOX_TOOL_KEYS = {
    "bash": "SANDBOX_BASH",
    "read_file": "SANDBOX_READ_FILE",
    "write_file": "SANDBOX_WRITE_FILE",
    "edit_file": "SANDBOX_EDIT_FILE",
}

DEFAULT_IMAGE = "python:3.11-slim"
DEFAULT_PERSIST_DIR = os.path.join(os.getcwd(), "sandbox_data")


class SandboxConfig:
    """沙箱配置管理，从 .env 加载各工具执行模式及全局沙箱参数"""

    def __init__(self):
        self.tool_modes: dict[str, str] = {}
        self.image: str = os.getenv("SANDBOX_IMAGE", DEFAULT_IMAGE)
        self.persist_dir: str = os.getenv("SANDBOX_PERSIST_DIR", DEFAULT_PERSIST_DIR)
        self._sandbox_available: bool = True
        self._load_tool_modes()
        self._validate()

    def _load_tool_modes(self):
        for tool_name, env_key in SANDBOX_TOOL_KEYS.items():
            value = os.getenv(env_key, "local").lower().strip()
            if value not in VALID_MODES:
                LOG.warning(f"沙箱配置 {env_key}={value} 无效，回退为 local。有效值: {VALID_MODES}")
                value = "local"
            self.tool_modes[tool_name] = value

    def _validate(self):
        if not os.path.isdir(self.persist_dir):
            try:
                os.makedirs(self.persist_dir, exist_ok=True)
            except OSError as e:
                LOG.error(f"沙箱持久化目录 {self.persist_dir} 无法创建: {e}，沙箱模式不可用")
                self._sandbox_available = False
                return

        if not os.access(self.persist_dir, os.W_OK):
            LOG.error(f"沙箱持久化目录 {self.persist_dir} 不可写，沙箱模式不可用")
            self._sandbox_available = False
            return

        sandbox_tools = [t for t, m in self.tool_modes.items() if m == "sandbox"]
        if sandbox_tools and not self._sandbox_available:
            LOG.warning(f"以下工具配置为 sandbox 但沙箱不可用，将回退为 local: {sandbox_tools}")
            for t in sandbox_tools:
                self.tool_modes[t] = "local"

    def mark_sandbox_unavailable(self):
        """Docker 不可用时调用，将所有 sandbox 工具降级为 local"""
        self._sandbox_available = False
        sandbox_tools = [t for t, m in self.tool_modes.items() if m == "sandbox"]
        if sandbox_tools:
            LOG.warning(f"Docker 不可用，以下工具回退为 local: {sandbox_tools}")
            for t in sandbox_tools:
                self.tool_modes[t] = "local"

    def is_sandbox(self, tool_name: str) -> bool:
        return self.tool_modes.get(tool_name, "local") == "sandbox"

    @property
    def sandbox_available(self) -> bool:
        return self._sandbox_available
