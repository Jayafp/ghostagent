import os

from app.browser.baidu_search import baidu_search
from app.browser.google_search import google_search

web_search_engine = os.getenv('web_search_engine', 'google')

async def search(query: str):
    if web_search_engine.lower() == 'baidu':
        return await baidu_search(query)
    else:
        return await google_search(query)