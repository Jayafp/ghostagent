from app.llm.memory_manager import load_recent_messages
from app.llm.utils import is_pure_user_message, is_answer_content
from app.log.logger import LOG

priority = {'thinking': 0, 'text': 1}


def append_history_message(history, user_msg, assistant_msg):
    if assistant_msg[:8] == '```\n\n```':
        assistant_msg = assistant_msg[8:]
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": assistant_msg})


def format_history_from_memory(session_id: str, rounds: int = 10) -> list:
    """
    从 memory 加载历史对话并格式化为 Gradio Chatbot 的 messages 格式

    格式: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    过程内容(tool_use/tool_result/thinking)放入 code 代码块
    """
    try :
        messages = load_recent_messages(session_id, rounds=rounds)
        if not messages:
            return []

        history = []
        cnt_status = 0  # 0:找role=user的记录, 1:找role=assistant的记录
        user_msg = None
        assistant_msg = "```\n"
        cnt_is_answer = False
        for msg in messages:
            content = msg.get("content", "")
            if cnt_status == 0:
                if is_pure_user_message(msg):
                    user_msg = content
                    cnt_status = 1
                else:
                    pass
            else:
                if is_pure_user_message(msg):
                    assistant_msg += '\n```'
                    append_history_message(history, user_msg, assistant_msg)

                    user_msg = content
                    cnt_status = 1

                    # reinit
                    assistant_msg = '```\n'
                    cnt_is_answer = False
                else:
                    if isinstance(content, list):
                        cnt_is_answer = is_answer_content(content)
                        content = sorted(content, key=lambda x: priority.get(x.get('type', ''), 2))

                        # 纯回答消息：先关闭前面工具调用累积的代码块，避免最终回答被困在 ``` 内
                        if cnt_is_answer:
                            assistant_msg += '\n```\n'

                        for block in content:
                            if not isinstance(block, dict):
                                # 忽略, 正常不会有这样的数据
                                continue
                            block_type = block.get('type', '')
                            if 'thinking' == block_type:
                                # 不回显 thinking 记录
                                pass
                            elif 'text' == block_type:
                                assistant_msg += block.get('text', '')
                                assistant_msg += '\n'
                            elif 'tool_use' == block_type:
                                assistant_msg += '🔧 调用工具: '
                                assistant_msg += str(block.get('input'))
                                assistant_msg += '\n'
                            elif 'tool_result' == block_type:
                                assistant_msg += '👉🏻 工具结果:'
                                tool_result = block.get('content', '')
                                if tool_result and len(tool_result) > 200:
                                    assistant_msg += tool_result[:200].replace("\n", " ") + "..."
                                else:
                                    assistant_msg += tool_result.replace("\n", " ")
                                assistant_msg += '\n'

                    if cnt_is_answer:
                        append_history_message(history, user_msg, assistant_msg)

                        # reinit
                        cnt_status = 0
                        user_msg = None
                        assistant_msg = '```\n'
                        cnt_is_answer = False

        if not cnt_is_answer and user_msg:
            append_history_message(history, user_msg, assistant_msg)

        return history
    except Exception as e:
        LOG.exception(f'加载历史信息失败, {e}')
        return []


if __name__ == '__main__':
    session_id = 'main'
    print(format_history_from_memory(session_id))