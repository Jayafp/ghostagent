import asyncio
import os
import requests
from typing import List, Dict, Union

from playwright.async_api import async_playwright, Page
from app.log.logger import LOG

from app.browser.open_chrome import open_chrome, wait_for_chrome
from app.browser.playwright_router import handle_route

FETCH_TIMEOUT = int(os.getenv('chrome_fetch_timeout'))
CHROME_DEBUG_PORT = int(os.getenv('chrome_debug_port'))
BAIDU_SEARCH_URL = "https://www.baidu.com/s?wd="


def decode_baidu_urls(results: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    批量解码百度搜索结果中的重定向链接

    Args:
        results: 搜索结果列表，每个结果包含 url 字段

    Returns:
        List[Dict[str, str]]: 解码后的结果列表
    """
    for result in results:
        result['url'] = decode_baidu_url(result['url'])
    return results


def decode_baidu_url(encoded_url: str) -> str:
    """
    解码百度的重定向链接

    百度搜索结果中的 URL 通常是跳转链接（baidu.com/link）
    通过发送 HEAD 请求获取真实 URL（Location header）

    Args:
        encoded_url: 百度的跳转链接（如 https://www.baidu.com/link?url=...）

    Returns:
        str: 解码后的真实 URL
             如果解码失败，返回原始 URL

    Implementation:
        1. 发送 HEAD 请求，禁止自动重定向
        2. 检查 302 响应中的 Location header
        3. 超时 3 秒，避免阻塞
    """
    try:
        resp = requests.head(
            encoded_url,
            allow_redirects=False,
            timeout=3
        )
        if resp.status_code == 302:
            return resp.headers.get('Location')
    except:
        pass
    return encoded_url


async def _parse_baidu_search_results(page: Page) -> List[Dict[str, str]]:
    """
    解析百度搜索结果页面

    使用 Playwright 在浏览器中执行 JavaScript 提取搜索结果

    Args:
        page: Playwright Page 对象，已加载百度搜索结果页

    Returns:
        List[Dict[str, str]]: 搜索结果列表
            每个结果包含以下字段：
            - title: str, 搜索结果标题
            - url: str, 搜索结果链接（可能包含百度跳转链接）
            - description: str, 搜索结果摘要

    Note:
        - 等待搜索结果容器 #content_left 出现
        - 最多等待 10 秒
        - 尝试从 data-tools 属性提取真实 URL
        - 过滤掉没有标题或 URL 的结果
    """
    # 等待搜索结果容器
    await page.wait_for_selector("#content_left", timeout=10000)

    results = await page.eval_on_selector_all(
        ".result, .c-container",
        """
        elements => elements.map(el => {
            // 标题通常在 h3 或 .t 中
            const titleEl = el.querySelector('h3, .t');
            // 链接 - 百度的真实链接需要获取 data-tools 或通过 href 属性
            const linkEl = el.querySelector('a[href]');
            // 描述在 .c-abstract 或 .c-span9 或 .c-row 中
            const descEl = el.querySelector('.c-abstract, .c-span9, .content-right_8Zs40');

            let url = '';
            if (linkEl) {
                url = linkEl.href;
                // 如果是百度跳转链接，尝试提取真实 URL
                if (url.includes('baidu.com/link')) {
                    const dataTools = linkEl.getAttribute('data-tools') || el.getAttribute('data-tools');
                    if (dataTools) {
                        try {
                            const toolsData = JSON.parse(dataTools);
                            if (toolsData && toolsData.title && toolsData.title.match(/\\[(https?:\\/\\/[^\\]]+)\\]/)) {
                                url = toolsData.title.match(/\\[(https?:\\/\\/[^\\]]+)\\]/)[1];
                            } else if (toolsData && toolsData.url) {
                                url = toolsData.url;
                            }
                        } catch (e) {}
                    }
                }
            }

            return {
                title: titleEl ? titleEl.innerText.trim() : '',
                url: url,
                description: descEl ? descEl.innerText.trim() : ''
            };
        }).filter(r => r.title && r.url);
        """
    )
    return results


async def baidu_search(query: str, retry: bool = True) -> Union[List[Dict[str, str]], List[str]]:
    """
    执行百度搜索并返回结果列表

    功能：
    1. 启动 Chrome 浏览器
    2. 访问百度并搜索关键词
    3. 解析搜索结果（标题、URL、描述）
    4. 解码百度跳转链接
    5. 过滤并返回前 5 条有效结果

    Args:
        query: 搜索关键词
        retry: 是否允许重试（默认 True）
            - 首次使用无头模式
            - 重试时使用有界面模式

    Returns:
        List[Dict[str, str]]: 成功时返回搜索结果列表
            每个字典包含：title, url, description
        List[str]: 失败时返回 ["搜索返回异常，请稍后重试"]

    Raises:
        不抛出异常，所有错误都被捕获

    Processing Flow:
        1. 启动 Chrome（首次 headless，重试时非 headless）
        2. 等待 CDP 端口就绪
        3. 导航到百度搜索页面
        4. 解析搜索结果
        5. 过滤百度内部链接（baidu.com/s）
        6. 解码跳转链接
        7. 返回前 5 条结果

    Note:
        - 百度的链接是跳转链接，需要额外解码
        - decode_baidu_urls() 会发送 HEAD 请求获取真实 URL
        - 过滤掉 www.baidu.com/s 的内部搜索结果链接

    Example:
        >>> results = await baidu_search("Python 教程")
        >>> print(results[0]["title"])
        'Python教程 - 廖雪峰的官方网站'
    """
    process = None
    try:
        # 首次尝试使用无头模式，重试时使用有界面模式
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
            await page.route("**/*", handle_route)

            #wait_until = 'networkidle' if retry else 'load'
            wait_until = 'load' if retry else 'networkidle'
            LOG.info(f'baidu search, query={query}')
            await page.goto(
                f'{BAIDU_SEARCH_URL}{query}',
                wait_until=wait_until,
                timeout=FETCH_TIMEOUT
            )

            # 解析网页内容，拿到 title, url, description
            results = await _parse_baidu_search_results(page)
            await browser.close()

        LOG.info(f'baidu search, query={query}, result_count={len(results)}')

        # 过滤掉百度内部搜索结果链接（如 www.baidu.com/s?wd=...）
        results = [r for r in results if 'www.baidu.com/s' not in r['url']]

        # 取前 5 条结果
        results = results[:5]

        # 解码百度跳转链接
        results = decode_baidu_urls(results)

        return results

    except Exception as e:
        LOG.error(f'baidu search error, query={query}, error={e}')
        if retry:
            return await baidu_search(query, False)
        return ["搜索返回异常，请稍后重试"]

    finally:
        if process:
            process.terminate()


if __name__ == '__main__':
    text = asyncio.run(baidu_search('支付宝 神券项目'))
    print(text)

    # 测试 decode_baidu_url 函数
    urls = [
        'https://www.baidu.com/s?tn=news&rtt=1&bsst=1&wd=%E6%94%AF%E4%BB%98%E5%AE%9D+%E7%A5%9E%E5%88%B8%E9%A1%B9%E7%9B%AE&cl=2',
        'https://aiqicha.baidu.com/feedback/official?from=baidu&type=gw',
    ]
    for url in urls:
        print(url)
        print('\t' + decode_baidu_url(url))
