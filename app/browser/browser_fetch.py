import asyncio
import os
import subprocess
from typing import Optional, Union, List

import aiohttp
from playwright.async_api import async_playwright
from app.log.logger import LOG

from app.browser.open_chrome import open_chrome
from app.browser.open_chrome import wait_for_chrome
from app.browser.playwright_router import handle_route

FETCH_TIMEOUT = int(os.getenv('chrome_fetch_timeout'))
CHROME_DEBUG_PORT = int(os.getenv('chrome_debug_port'))


async def fetch_content(url: str, retry: bool = True) -> Union[str, List[str]]:
    """
    使用 Playwright + CDP 获取指定 URL 的网页内容

    功能：
    1. 启动 Chrome 浏览器（如未运行）
    2. 通过 Chrome DevTools Protocol (CDP) 连接
    3. 屏蔽不必要的资源（js/css/图片）
    4. 获取网页正文内容

    Args:
        url: 目标网页 URL
        retry: 是否允许重试（默认 True）
            - True: 首次使用 networkidle 等待策略，失败时重试
            - False: 使用 load 等待策略，有界面模式（用于某些特殊页面）

    Returns:
        str: 成功时返回网页 body 的纯文本内容
        List[str]: 失败时返回错误信息列表 ["获取网页内容异常，请稍后重试"]

    Raises:
        不抛出异常，所有错误都被捕获并返回错误信息

    Processing Flow:
        1. 启动 Chrome 并等待调试端口就绪
        2. 通过 CDP 连接到 Chrome
        3. 设置路由规则（屏蔽图片、CSS等）
        4. 导航到目标 URL
        5. 提取 body 的 inner_text
        6. 关闭浏览器并返回内容
        7. 异常时使用更保守的策略重试一次

    Note:
        有失败重试机制：
        - 首次使用 headless 模式和 networkidle
        - 失败后使用有界面模式和 load 事件
    """
    process = None
    try:
        # 打开 chrome 进程
        headless = True if retry else False
        process = open_chrome(CHROME_DEBUG_PORT, headless)

        # 等待 chrome 调试端口就绪
        ws_url = await wait_for_chrome(CHROME_DEBUG_PORT)

        async with async_playwright() as p:
            # 连接到已运行的 Chrome
            browser = await p.chromium.connect_over_cdp(ws_url)

            context = browser.contexts[0]
            page = context.pages[0] if context.pages else await context.new_page()

            # 设置路由，屏蔽不必要的资源
            #await page.route("**/*", handle_route)

            # 根据重试状态选择加载策略
            wait_until = 'networkidle' if retry else 'load'
            LOG.info(f'goto url: {url}，wait_until={wait_until}')
            await page.goto(url, wait_until=wait_until, timeout=FETCH_TIMEOUT)

            # 获取网页文本内容
            text = await page.inner_text("body")
            await browser.close()

        return text

    except Exception as e:
        LOG.error(f'fetch content error: {url}, retry={retry}, error={e}')
        if retry:
            # 重试一次，使用更保守的策略
            return await fetch_content(url, False)
        return ["获取网页内容异常，请稍后重试"]

    finally:
        if process:
            process.terminate()


if __name__ == '__main__':
    #http://www.baidu.com/link?url=scWdPLY00cOg-bNkAkqJf7Y-UZ7VPi7TyGBxDMlTiRJMtOzGjOAeWVaTqAIME-AX
    #https://www.bing.com/search?q=%E6%94%AF%E4%BB%98%E5%AE%9DN5D%E8%AE%BE%E5%A4%87&PC=U316&FORM=CHROMN
    #https://sankalp.bearblog.dev/how-prompt-caching-works/
    #load不行, https://opendocs.alipay.com/b/0irv8o
    #networkidle不行, 因为有css加载不出来, https://cloud.tencent.com/developer/article/2551041
    text = asyncio.run(fetch_content('https://github.com/Fission-AI/OpenSpec/issues/738', False))
    print(text)
