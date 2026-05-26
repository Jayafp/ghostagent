import json

from web.history_message import format_history_from_memory


def test_format_history_from_memory():
    session_id = 'test'

    print(json.dumps(format_history_from_memory(session_id), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    test_format_history_from_memory()