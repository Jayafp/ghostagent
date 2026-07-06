"""助手回复流式段落的渲染。

把 agent 事件流攒成的有序段落列表渲染成最终展示文本：
工具过程(PROCESS)用代码块包裹、正文(TEXT/ANSWER)用 Markdown，二者可交错出现。
任务进度面板插在最后一段正文之前，没有正文时追加末尾。

抽成独立无依赖模块，便于单测（避免触发 webui 的重导入副作用）。
"""


def render_segments(segments, task_panel=""):
    """把有序段落列表渲染为助手回复文本。

    Args:
        segments: [{"kind": "process"|"text", "text": str}]，按事件到达顺序排列。
                  调用方负责把相邻同类型事件合并进同一段落。
        task_panel: 任务进度面板文本，空字符串表示不展示。

    Returns:
        str: 拼接后的展示文本。工具过程段落渲染为 ``` 代码块，正文段落原样输出
             （由前端按 Markdown 渲染）。任务面板插在最后一段"非空正文"之前
             （保持"过程之下、回复之上"语义），没有正文段落时追加到末尾。
    """
    # 定位最后一段"非空正文"，任务面板插在它之前
    last_text_idx = -1
    for i, seg in enumerate(segments):
        if seg["kind"] == "text" and seg["text"].strip():
            last_text_idx = i

    rendered = []
    for i, seg in enumerate(segments):
        text = seg["text"]
        if not text.strip():
            continue
        if i == last_text_idx and task_panel:
            rendered.append(task_panel)
        rendered.append(f"```\n{text.strip()}\n```" if seg["kind"] == "process" else text)
    if task_panel and last_text_idx == -1:
        rendered.append(task_panel)

    return "\n".join(rendered)
