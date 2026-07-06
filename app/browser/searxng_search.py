import asyncio
import os
import subprocess
from typing import List, Dict, Union

import aiohttp
from app.log.logger import LOG

SEARXNG_URL = os.getenv('searxng_url', 'http://localhost:8182')
SEARXNG_TIMEOUT = int(os.getenv('searxng_timeout', '10'))
# 引擎列表（逗号分隔），SearxNG 会聚合多源结果，单引擎被封不影响整体
SEARXNG_ENGINES = os.getenv('searxng_engines', 'google cse,baidu')

# docker-compose 文件路径（app/browser/searxng_search.py → 项目根/docker/searxng/）
_COMPOSE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'docker', 'searxng', 'docker-compose.yml',
)


def _is_searxng_running() -> bool:
    """检查 searxng 容器是否在运行"""
    try:
        result = subprocess.run(
            ['docker', 'inspect', '-f', '{{.State.Running}}', 'searxng'],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() == 'true'
    except Exception:
        return False


def ensure_searxng_running() -> None:
    """
    确保 SearxNG 容器已启动

    未运行则通过 docker compose up -d 启动；已运行则不操作（不重启，避免
    影响正在进行的搜索）。如需应用 settings.yml 改动，用项目根的 searxng.sh
    脚本（会重启已运行的容器）。
    """
    if _is_searxng_running():
        LOG.info('searxng container already running')
        return
    try:
        LOG.info('starting searxng container...')
        subprocess.run(
            ['docker', 'compose', '-f', _COMPOSE_FILE, 'up', '-d'],
            capture_output=True, text=True, timeout=60, check=True,
        )
        LOG.info('searxng container started')
    except Exception as e:
        LOG.error(f'failed to start searxng container: {e}')


async def searxng_search(query: str) -> Union[List[Dict[str, str]], List[str]]:
    """
    通过 SearxNG 元搜索引擎执行搜索

    SearxNG 是自托管的元搜索引擎，聚合 google/bing/duckduckgo/baidu 等多个引擎结果。
    相比 CDP 浏览器方案：轻量 HTTP 调用、多源容错（单引擎被封不影响整体）、
    自带引擎级超时，避免某个引擎卡住导致整体超时。

    依赖 docker/searxng 部署的 SearxNG 服务，通过 searxng_url 配置地址。
    代理在 SearxNG 的 settings.yml 中配置（outgoing.proxies）。

    Args:
        query: 搜索关键词

    Returns:
        List[Dict[str, str]]: 成功时返回搜索结果列表（前 5 条）
            每个字典包含：title, url, description
        List[str]: 失败时返回 ["搜索返回异常，请稍后重试"]

    Example:
        >>> results = await searxng_search("Python 教程")
        >>> print(results[0]["title"])
    """
    params = {
        'q': query,
        'format': 'json',
        'engines': SEARXNG_ENGINES,
        'safesearch': 0,
    }
    try:
        timeout = aiohttp.ClientTimeout(total=SEARXNG_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f'{SEARXNG_URL}/search', params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()

        results = [
            {
                'title': r.get('title', ''),
                'url': r.get('url', ''),
                'description': r.get('content', ''),
            }
            for r in data.get('results', [])
            if r.get('title') and r.get('url')
        ][:5]

        LOG.info(f'searxng search, query={query}, result_count={len(results)}')
        return results
    except Exception as e:
        LOG.error(f'searxng search error, query={query}, error={e}')
        return ["搜索返回异常，请稍后重试"]


if __name__ == '__main__':
    print(asyncio.run(searxng_search('阿里巴巴 BABA 股价下跌 2026')))
