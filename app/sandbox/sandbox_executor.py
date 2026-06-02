import os
import tarfile
import io
from app.log.logger import LOG
from app.sandbox.container_manager import ContainerManager

WORKSPACE = "/workspace"


class SandboxExecutor:
    """在 Docker 沙箱容器中执行工具操作"""

    def __init__(self, container_manager: ContainerManager, session_id: str):
        self._mgr = container_manager
        self._session_id = session_id
        self._cwd = WORKSPACE  # 跟踪容器内当前工作目录

    def _get_container(self):
        return self._mgr.get_or_create(self._session_id)

    def _map_path(self, path: str) -> str:
        """将用户路径映射为容器内路径"""
        if os.path.isabs(path):
            if path.startswith(WORKSPACE):
                return path
            return None  # 越界路径
        return os.path.join(self._cwd, path)

    def _exec(self, cmd: str, workdir: str = None) -> tuple[int, str]:
        """在容器中执行命令，返回 (exit_code, output)"""
        container = self._get_container()
        if container is None:
            return -1, "Error: Sandbox container not available"

        wd = workdir or self._cwd
        try:
            exit_code, output = container.exec_run(
                cmd,
                workdir=wd,
                demux=False,
            )
            text = output.decode("utf-8", errors="replace") if output else ""
            return exit_code, text
        except Exception as e:
            LOG.error(f"沙箱执行命令失败: {e}")
            return -1, f"Error: {e}"

    def run_bash(self, command: str) -> str:
        """在沙箱中执行 bash 命令"""
        # 使用 bash -c 执行，并跟踪 cd 命令更新工作目录
        exit_code, output = self._exec(f"/bin/bash -c {quote(command)}")

        # 检测 cd 命令并更新 cwd
        self._update_cwd_after_bash(command)

        out = output.strip()
        if not out:
            return "(no output)"
        return out[:50000]

    def _update_cwd_after_bash(self, command: str):
        """解析 bash 命令中的 cd，更新跟踪的工作目录"""
        # 查询容器实际工作目录
        exit_code, cwd_out = self._exec("pwd")
        if exit_code == 0:
            new_cwd = cwd_out.strip()
            if new_cwd:
                self._cwd = new_cwd

    def run_read(self, path: str, limit: int = None) -> str:
        """在沙箱中读取文件"""
        container_path = self._map_path(path)
        if container_path is None:
            return f"Error: Access denied - path outside workspace: {path}"

        # 使用 head + cat 组合，支持 limit
        if limit:
            cmd = f"head -n {limit} {quote(container_path)}"
        else:
            cmd = f"cat {quote(container_path)}"

        exit_code, output = self._exec(cmd)
        if exit_code != 0:
            return f"Error: {output.strip()}"
        return output[:50000]

    def run_write(self, path: str, content: str) -> str:
        """在沙箱中写入文件"""
        container_path = self._map_path(path)
        if container_path is None:
            return f"Error: Access denied - path outside workspace: {path}"

        container = self._get_container()
        if container is None:
            return "Error: Sandbox container not available"

        # 自动创建父目录
        parent = os.path.dirname(container_path)
        self._exec(f"mkdir -p {quote(parent)}")

        # 通过 tar 写入文件内容
        try:
            self._put_file(container, container_path, content)
            return f"Wrote {len(content)} bytes to {path}"
        except Exception as e:
            return f"Error: {e}"

    def run_edit(self, path: str, old_text: str, new_text: str) -> str:
        """在沙箱中编辑文件（精确文本替换）"""
        container_path = self._map_path(path)
        if container_path is None:
            return f"Error: Access denied - path outside workspace: {path}"

        # 先读取文件内容检查 old_text 是否存在
        exit_code, content = self._exec(f"cat {quote(container_path)}")
        if exit_code != 0:
            return f"Error: Cannot read {path}"
        if old_text not in content:
            return f"Error: Text not found in {path}"

        # 使用 python 执行替换
        script = (
            "import sys\n"
            "p=sys.argv[1]\n"
            "c=open(p).read()\n"
            "c=c.replace(sys.argv[2],sys.argv[3],1)\n"
            "open(p,'w').write(c)\n"
        )
        cmd = (
            f'python3 -c {quote(script)} '
            f'{quote(container_path)} {quote(old_text)} {quote(new_text)}'
        )
        exit_code, output = self._exec(cmd)
        if exit_code != 0:
            return f"Error: {output.strip()}"
        return f"Edited {path}"

    def _put_file(self, container, container_path: str, content: str):
        """通过 tar 将文件写入容器"""
        filename = os.path.basename(container_path)
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode='w') as tar:
            info = tarfile.TarInfo(name=filename)
            data = content.encode("utf-8")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        tar_stream.seek(0)

        parent = os.path.dirname(container_path) or "/"
        ok = container.put_archive(parent, tar_stream)
        if not ok:
            raise RuntimeError("put_archive returned False")


def quote(s: str) -> str:
    """shell 单引号转义"""
    return "'" + s.replace("'", "'\\''") + "'"
