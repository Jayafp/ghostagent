import os
from dotenv import load_dotenv
load_dotenv()

from web.webui import run

# 使用 searxng 搜索引擎时，确保容器已启动（未运行才启动，不重启已运行的）
if os.getenv('web_search_engine', '').lower() == 'searxng':
    from app.browser.searxng_search import ensure_searxng_running
    ensure_searxng_running()

try :
    run()
except Exception as e:
    print(e)
