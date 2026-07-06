import asyncio
import os
import subprocess
from typing import List, Dict, Union

import aiohttp
from playwright.async_api import async_playwright, Page
from app.log.logger import LOG

from app.browser.open_chrome import open_chrome, wait_for_chrome
from app.browser.playwright_router import handle_route

FETCH_TIMEOUT = int(os.getenv('chrome_fetch_timeout'))
CHROME_DEBUG_PORT = int(os.getenv('chrome_debug_port'))
GOOGLE_SEARCH_URL = "https://www.google.com/search?q="


async def _parse_google_search_results(page: Page) -> List[Dict[str, str]]:
    """
    解析 Google 搜索结果页面

    使用 Playwright 在浏览器中执行 JavaScript 提取搜索结果

    Args:
        page: Playwright Page 对象，已加载 Google 搜索结果页

    Returns:
        List[Dict[str, str]]: 搜索结果列表
            每个结果包含以下字段：
            - title: str, 搜索结果标题
            - url: str, 搜索结果链接
            - description: str, 搜索结果摘要

    Note:
        - 等待搜索结果容器 div#search 出现
        - 最多等待 15 秒
        - 过滤掉没有标题或 URL 的结果
    """
    # 等待搜索结果容器
    await page.wait_for_selector("div#search", timeout=15000)

    results = await page.eval_on_selector_all(
        "div#search > div > div > div > div",
        """
        elements => elements.map(el => {
            const titleEl = el.querySelector('h3');
            const linkEl = el.querySelector('a[href^="http"]');
            const descEl = el.querySelector('div[style*="-webkit-line-clamp"], div.VwiC3b');

            return {
                title: titleEl ? titleEl.innerText : '',
                url: linkEl ? linkEl.href : '',
                description: descEl ? descEl.innerText : ''
            };
        }).filter(r => r.title && r.url);
        """
    )
    return results


async def google_search(query: str, retry: bool = True) -> Union[List[Dict[str, str]], List[str]]:
    """
    执行 Google 搜索并返回结果列表

    功能：
    1. 启动 Chrome 浏览器
    2. 注入反检测脚本绕过 Google 风控
    3. 访问 Google 搜索页面
    4. 解析搜索结果（标题、URL、描述）
    5. 返回前 5 条结果

    Args:
        query: 搜索关键词
        retry: 是否允许重试（默认 True）
            - 重试时使用更保守的加载策略

    Returns:
        List[Dict[str, str]]: 成功时返回搜索结果列表
            每个字典包含：title, url, description
        List[str]: 失败时返回 ["搜索返回异常，请稍后重试"]

    Raises:
        不抛出异常，所有错误都被捕获

    Implementation Notes:
        - 必须使用无头模式 False（headless=False），因为 Google 对 headless 检测严格
        - 注入反检测脚本隐藏 webdriver 属性
        - 设置 1920x1080 viewport 模拟真实用户
        - 失败时自动重试一次，使用更保守的加载策略

    Example:
        >>> results = await google_search("Python tutorial")
        >>> print(results[0]["title"])
        'Python Tutorial - W3Schools'
    """
    process = None
    try:
        # 必须使用有界面模式，Google 对无头模式检测严格
        headless = False
        process = open_chrome(CHROME_DEBUG_PORT, headless)

        # 等待 chrome 调试端口就绪
        ws_url = await wait_for_chrome(CHROME_DEBUG_PORT)

        async with async_playwright() as p:
            # 连接到已运行的 Chrome
            browser = await p.chromium.connect_over_cdp(ws_url)

            context = browser.contexts[0]
            page = context.pages[0] if context.pages else await context.new_page()

            # 注入反检测脚本，隐藏自动化标志
            await page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            """)

            # 设置合理的 viewport 模拟真实用户
            await page.set_viewport_size({"width": 1920, "height": 1080})

            wait_until = 'networkidle' if retry else 'load'
            LOG.info(f'google search, query={query}')
            await page.goto(
                f'{GOOGLE_SEARCH_URL}{query}',
                wait_until=wait_until,
                timeout=FETCH_TIMEOUT
            )

            # 解析搜索结果
            results = await _parse_google_search_results(page)
            await browser.close()

        LOG.info(f'google search, query={query}, result_count={len(results)}')
        return results[:10]

    except Exception as e:
        LOG.error(f'google search, query={query}, retry={retry}, error={e}')
        if retry:
            return await google_search(query, False)
        return ["搜索返回异常，请稍后重试"]

    finally:
        if process:
            process.terminate()


if __name__ == '__main__':
    text = asyncio.run(google_search('Claude Code 源码泄露 2026年3月31日'))
    print(text)