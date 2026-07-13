import os
import subprocess
import asyncio
import aiohttp
import uuid
from typing import Optional

CHROME_PATH = os.getenv('chrome_path')


def get_chrome_version() -> int:
    """
    获取 Chrome 浏览器的主版本号

    通过运行 chrome --version 命令获取版本信息

    Returns:
        int: Chrome 主版本号（如 146）
        1: 如果获取失败（默认值）

    Example:
        >>> version = get_chrome_version()
        >>> print(version)
        146
    """
    try:
        result = subprocess.run([CHROME_PATH, "--version"], capture_output=True, text=True)
        version = result.stdout.strip()  # e.g., "Google Chrome 146.0.7680.165"
        version = version.split()[-1]  # "146.0.7680.165"
        version = version.split('.')[0]  # "146"
        return int(version)
    except Exception as e:
        print(e)
        return 1


# 全局 Chrome 版本号（模块加载时计算一次）
chrome_version = get_chrome_version()


def open_chrome(debug_port: int, headless: bool = True) -> subprocess.Popen:
    """
    启动 Chrome 浏览器并开启远程调试端口

    功能：
    1. 杀掉占用调试端口的旧进程
    2. 启动 Chrome 并启用 CDP (Chrome DevTools Protocol)
    3. 配置防检测和性能优化参数

    Args:
        debug_port: Chrome 调试端口（用于 CDP 连接）
        headless: 是否以无头模式启动（默认 True）
                - True: 无界面模式
                - False: 有界面模式（用于某些需要 GPU/渲染的页面）

    Returns:
        subprocess.Popen: Chrome 进程对象

    Note:
        - 使用固定 profile 目录 /tmp/chrome-debug-profile
        - 配置了多项防检测参数以绕过网站自动化检测
        - Chrome 122+ 版本使用 --headless=new 参数
    """
    # 每次使用唯一的 profile 目录，避免冲突
    temp_profile = f"/tmp/chrome-debug-profile-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    #temp_profile = f"/tmp/chrome-debug-profile"

    # 先杀掉占用端口的旧进程
    subprocess.run(
        f"lsof -ti :{debug_port} | xargs kill -9 2>/dev/null || true",
        shell=True,
        capture_output=True
    )

    if headless:
        headless_param = '--headless=new' if chrome_version >= 122 else '--headless'
    else:
        headless_param = ''

    chrome_process = subprocess.Popen([
        CHROME_PATH,
        f"--remote-debugging-port={debug_port}",
        f"--user-data-dir={temp_profile}",
        "--no-first-run",
        f"{headless_param}",
        "--disable-blink-features=AutomationControlled",  # 隐藏自动化标志
        "--disable-infobars",
        "--disable-dev-shm-usage",
        "--disable-browser-side-navigation",
        "--disable-gpu",
        "--disable-extensions",
        "--disable-background-networking",
        "--disable-sync",
        "--metrics-recording-only",
        "--disable-default-apps",
        "--mute-audio",
        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.7680.165 Safari/537.36",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return chrome_process


async def wait_for_chrome(port: int, timeout: int = 10) -> str:
    """
    等待 Chrome 调试端口就绪，返回 WebSocket URL

    轮询检查 Chrome 的 HTTP 调试端点，直到返回成功响应

    Args:
        port: Chrome 调试端口
        timeout: 最大等待时间（秒），默认 10 秒

    Returns:
        str: WebSocket 调试 URL (webSocketDebuggerUrl)
            格式：ws://localhost:{port}/devtools/browser/{id}

    Raises:
        TimeoutError: 如果在 timeout 秒内 Chrome 未就绪

    Implementation:
        1. 轮询 http://localhost:{port}/json/version
        2. 间隔 0.3 秒检查一次
        3. 返回 webSocketDebuggerUrl 字段
    """
    start = asyncio.get_event_loop().time()

    while asyncio.get_event_loop().time() - start < timeout:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://localhost:{port}/json/version",
                    timeout=aiohttp.ClientTimeout(total=1)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["webSocketDebuggerUrl"]
        except:
            pass
        await asyncio.sleep(0.3)

    raise TimeoutError(f"Chrome 未在 {timeout} 秒内启动")