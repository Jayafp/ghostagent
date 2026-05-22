import re
from typing import Set

# 需要屏蔽的资源类型（这些资源对文本提取无帮助）
BLOCKED_TYPES: Set[str] = {"image", "font", "media", "stylesheet"}

# 需要屏蔽的域名正则（主要是国外广告/分析服务，国内访问慢或不需要）
BLOCKED_PATTERN = re.compile(
    r"(google\.com|googletagmanager|google-analytics|googlesyndication|"
    r"facebook\.com|facebook\.net|twitter\.com|youtube\.com|doubleclick)",
    re.IGNORECASE
)


async def handle_route(route) -> None:
    """
    Playwright 路由处理器 - 屏蔽不必要的资源请求

    功能：
    1. 屏蔽图片、字体、媒体、CSS 等非必要资源（加快加载速度）
    2. 屏蔽国外广告和分析服务（避免加载缓慢或失败）

    Args:
        route: Playwright Route 对象，代表一个网络请求

    Returns:
        None

    Processing:
        - 如果请求的资源类型在 BLOCKED_TYPES 中，中止请求
        - 如果请求的 URL 匹配 BLOCKED_PATTERN，中止请求
        - 否则继续正常请求

    Example:
        >>> await page.route("**/*", handle_route)
    """
    if route.request.resource_type in BLOCKED_TYPES:
        await route.abort()
    elif BLOCKED_PATTERN.search(route.request.url):
        await route.abort()
    else:
        await route.continue_()