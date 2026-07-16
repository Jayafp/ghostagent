import os
import atexit
import threading
from app.log.logger import LOG

try:
    import docker
    from docker.errors import DockerException, NotFound, APIError
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

CONTAINER_LABEL = "ghostagent-sandbox"
CONTAINER_PREFIX = "ghostagent-sb-"


class ContainerManager:
    """管理每个会话的 Docker 沙箱容器生命周期"""

    def __init__(self, config: "SandboxConfig"):
        self.config = config
        self._client = None
        self._containers: dict[str, any] = {}  # session_id -> container
        self._docker_available = False
        self._lock = threading.Lock()
        self._init_client()

    def _init_client(self):
        if not DOCKER_AVAILABLE:
            LOG.warning("docker Python SDK 未安装，沙箱不可用")
            self.config.mark_sandbox_unavailable()
            return

        try:
            self._client = docker.from_env()
            self._client.ping()
            self._docker_available = True
            LOG.info("Docker daemon 连接成功，沙箱可用")
        except DockerException as e:
            LOG.warning(f"Docker daemon 不可用: {e}，所有沙箱工具将回退为 local 模式")
            self._docker_available = False
            self.config.mark_sandbox_unavailable()
            self._client = None

    @property
    def available(self) -> bool:
        return self._docker_available

    def _container_name(self, session_id: str) -> str:
        return f"{CONTAINER_PREFIX}{session_id}"

    def _persist_path(self, session_id: str) -> str:
        return os.path.join(self.config.persist_dir, session_id)

    def get_or_create(self, session_id: str):
        """获取或创建会话对应的沙箱容器"""
        if not self._docker_available:
            return None

        with self._lock:
            if session_id in self._containers:
                container = self._containers[session_id]
                try:
                    container.reload()
                    if container.status == "running":
                        return container
                except (NotFound, APIError):
                    del self._containers[session_id]

            # 查找已有容器（进程重启后复用）
            name = self._container_name(session_id)
            try:
                container = self._client.containers.get(name)
                container.reload()
                if container.status == "running":
                    self._containers[session_id] = container
                    return container
                else:
                    container.remove(force=True)
            except NotFound:
                pass
            except APIError as e:
                LOG.warning(f"查找容器 {name} 失败: {e}")

            return self._create_container(session_id)

    def _create_container(self, session_id: str):
        """创建新的沙箱容器"""
        persist_path = self._persist_path(session_id)
        os.makedirs(persist_path, exist_ok=True)

        name = self._container_name(session_id)
        try:
            container = self._client.containers.run(
                image=self.config.image,
                name=name,
                labels={"ghostagent": CONTAINER_LABEL, "session_id": session_id},
                volumes={
                    persist_path: {"bind": "/workspace", "mode": "rw"}
                },
                working_dir="/workspace",
                detach=True,
                tty=True,
                stdin_open=True,
            )
            self._containers[session_id] = container
            LOG.info(f"沙箱容器已创建: {name}，持久化目录: {persist_path}")
            return container
        except APIError as e:
            LOG.error(f"创建沙箱容器失败: {e}")
            return None

    def destroy(self, session_id: str):
        """销毁会话对应的沙箱容器。

        best-effort：Docker 不可用（``_client is None``，常见于无 Docker 环境，
        且多数子会话从未创建沙箱）时直接返回，避免 ``self._client.containers.get``
        抛 ``AttributeError`` 覆盖调用方（如子 Agent finally）的成功结果。
        """
        if not self._docker_available or self._client is None:
            return
        with self._lock:
            container = self._containers.pop(session_id, None)
            if container is None:
                name = self._container_name(session_id)
                try:
                    container = self._client.containers.get(name)
                except (NotFound, APIError):
                    return

            try:
                container.stop(timeout=5)
                container.remove(force=True)
                LOG.info(f"沙箱容器已销毁: {container.name}")
            except (NotFound, APIError) as e:
                LOG.warning(f"销毁沙箱容器失败: {e}")

    def cleanup_stale(self):
        """启动时清理残留的 ghostagent 沙箱容器"""
        if not self._docker_available:
            return

        with self._lock:
            try:
                containers = self._client.containers.list(
                    all=True,
                    filters={"label": f"ghostagent={CONTAINER_LABEL}"}
                )
                for c in containers:
                    try:
                        c.remove(force=True)
                        LOG.info(f"清理残留容器: {c.name}")
                    except APIError as e:
                        LOG.warning(f"清理容器 {c.name} 失败: {e}")
            except APIError as e:
                LOG.warning(f"查询残留容器失败: {e}")

    def destroy_all(self):
        """销毁所有本 manager 管理的容器"""
        with self._lock:
            for session_id in list(self._containers.keys()):
                container = self._containers.pop(session_id, None)
                if container is None:
                    continue
                try:
                    container.stop(timeout=5)
                    container.remove(force=True)
                    LOG.info(f"沙箱容器已销毁: {container.name}")
                except (NotFound, APIError) as e:
                    LOG.warning(f"销毁沙箱容器失败: {e}")
