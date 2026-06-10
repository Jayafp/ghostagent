import os
from dotenv import load_dotenv

load_dotenv(override=True)

from app.log.logger import LOG
api_key = os.getenv("ANTHROPIC_API_KEY")
if api_key:
    api_key = api_key[:5] + "***" + api_key[-5:]
LOG.info(f'init app module success, api_url={os.getenv("ANTHROPIC_BASE_URL")}, api_key={api_key}, model_id={os.getenv("MODEL_ID")}')