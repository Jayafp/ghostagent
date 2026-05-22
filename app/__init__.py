from dotenv import load_dotenv

load_dotenv(override=True)

from app.log.logger import LOG
LOG.info('init app module success...')