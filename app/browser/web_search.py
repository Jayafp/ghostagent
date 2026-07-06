import os

from app.browser.baidu_search import baidu_search
from app.browser.google_search import google_search
from app.browser.searxng_search import searxng_search

web_search_engine = os.getenv('web_search_engine', 'google')

async def search(query: str):
    engine = web_search_engine.lower()
    if engine == 'baidu':
        return await baidu_search(query)
    elif engine == 'searxng':
        return await searxng_search(query)
    else:
        return await google_search(query)
