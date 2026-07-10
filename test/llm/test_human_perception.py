import time

from app.llm.human_perception import async_generate_perception_ifneed


def test_async_generate_perception_ifneed(session_id: str):
    async_generate_perception_ifneed(session_id)


if __name__ == "__main__":
    session_id = "test"
    test_async_generate_perception_ifneed(session_id)
    time.sleep(999)
