import asyncio
import os
import uuid
from datetime import datetime
from pathlib import Path
import gradio as gr
from PIL import Image as PILImage
from app.llm.react_agent import agent_loop, StreamEvent
from web.history_message import format_history_from_memory
from app.log.logger import LOG
from app.llm.image_info import get_image_info


# 当前上传的图片路径（一次性使用，仅当前对话有效）
_current_image_path = {}


def generate_image_filename() -> str:
    """生成图片文件名: yyyyMMdd-uuid[:4]"""
    date_str = datetime.now().strftime("%Y%m%d")
    short_uuid = uuid.uuid4().hex[:4]
    return f"{date_str}-{short_uuid}"


def get_temp_dir(session_id: str) -> Path:
    """获取临时目录路径 temp/{session_id}/"""
    temp_dir = Path("temp") / session_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def clear_session_images(session_id: str):
    """清除session的所有临时图片"""
    if not session_id:
        return
    temp_dir = get_temp_dir(session_id)
    for img_file in temp_dir.glob("*"):
        try:
            img_file.unlink()
            LOG.info(f"已删除临时图片: {img_file}")
        except Exception as e:
            LOG.warning(f"删除临时图片失败: {img_file}, {e}")


def save_image_to_temp(image_path: str, session_id: str) -> str:
    """保存上传的图片到 temp/{session_id}/，文件名 yyyyMMdd-uuid[:4].ext"""
    if not image_path or not session_id:
        return ""

    try:
        # 清除之前的临时图片
        clear_session_images(session_id)

        # 获取原文件扩展名
        src_path = Path(image_path)
        ext = src_path.suffix.lower()
        if ext not in ['.png', '.jpg', '.jpeg', '.webp', '.gif']:
            ext = '.jpg'

        # 生成新文件名: yyyyMMdd-uuid[:4].ext
        filename = generate_image_filename()
        temp_dir = get_temp_dir(session_id)
        dest_path = temp_dir / f"{filename}{ext}"

        # 使用 PIL 保存图片
        with PILImage.open(image_path) as img:
            if img.mode in ('RGBA', 'LA', 'P'):
                if ext in ['.jpg', '.jpeg']:
                    background = PILImage.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                else:
                    img = img.convert('RGB')
            img.save(dest_path)

        LOG.info(f"图片已保存: {dest_path}")
        return str(dest_path)
    except Exception as e:
        LOG.exception(f"保存图片失败: {e}")
        return ""


# JS: 检测session，没有则重定向
SESSION_REDIRECT_JS = """
<script>
(function() {
    const url = new URL(window.location.href);
    if (!url.searchParams.has('session')) {
        const sid = Date.now().toString(36) + Math.random().toString(36).substr(2, 6);
        url.searchParams.set('session', sid);
        window.location.replace(url.toString());
    }
})();
</script>
"""


# 紧凑样式 CSS
COMPACT_CSS = """
/* 整体布局 - 确保聊天区域填满剩余空间 */
html, body { overflow-x: hidden; }

input, textarea {
    box-sizing: border-box !important;
}

.gradio-container {
    height: 100vh !important;
    display: flex !important;
    flex-direction: column !important;
    max-width: 100vw !important;
}

.gradio-container .gr-row {
    overflow-x: hidden !important;
    max-width: 100% !important;
}

.gradio-container .main {
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
    min-height: 0 !important;
}

.gradio-container .wrap {
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
    min-height: 0 !important;
}

/* 聊天区域滚动优化 - 去掉 smooth 避免延迟 */
.gradio-chatbot {
    overflow-y: auto !important;
    will-change: scroll-position !important;
    -webkit-overflow-scrolling: touch !important;
}

/* Gradio Chatbot 内部滚动容器 */
.gradio-chatbot .overflow-y-auto,
.gradio-chatbot [class*="scroll"] {
    scroll-behavior: auto !important;
}

/* 减少重绘 - 消息容器 */
.message-wrap {
    contain: layout style paint !important;
}

/* Loading 动画 */
.loading-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #666;
    padding: 4px 0;
}

/* 输入框样式 - 限制最大高度 */
.auto-resize-input textarea {
    max-height: 80px !important;
    overflow-y: auto !important;
    resize: none !important;
    font-size: 16px !important;
}

.loading-text {
    font-size: 14px;
}

.loading-dots {
    display: flex;
    gap: 4px;
}

.loading-dots span {
    width: 8px;
    height: 8px;
    background-color: #999;
    border-radius: 50%;
    animation: bounce 1.4s infinite ease-in-out both;
}

.loading-dots span:nth-child(1) {
    animation-delay: -0.32s;
}

.loading-dots span:nth-child(2) {
    animation-delay: -0.16s;
}

.loading-dots span:nth-child(3) {
    animation-delay: 0s;
}

@keyframes bounce {
    0%, 80%, 100% {
        transform: scale(0);
        opacity: 0.5;
    }
    40% {
        transform: scale(1);
        opacity: 1;
    }
}

/* 聊天消息区域更紧凑 */
.gradio-container .prose {
    line-height: 1.4 !important;
    margin-top: 0.3em !important;
    margin-bottom: 0.3em !important;
}

/* 减少段落间距 */
.gradio-container .prose p {
    margin-top: 0.3em !important;
    margin-bottom: 0.3em !important;
}

/* 减少列表间距 */
.gradio-container .prose ul, .gradio-container .prose ol {
    margin-top: 0.3em !important;
    margin-bottom: 0.3em !important;
    padding-left: 1.2em !important;
}

.gradio-container .prose li {
    margin-top: 0.1em !important;
    margin-bottom: 0.1em !important;
}

/* 代码块更紧凑 */
.gradio-container .prose pre {
    padding: 0.5em !important;
    margin-top: 0.3em !important;
    margin-bottom: 0.3em !important;
    font-size: 0.85em !important;
}

.gradio-container .prose code {
    padding: 0.1em 0.3em !important;
    font-size: 0.85em !important;
}

/* 消息气泡间距 */
.gradio-container .message-wrap {
    gap: 0.5em !important;
}

.gradio-container .prose hr {
    margin-top: 1.0em !important;
    margin-bottom: 1.0em !important;
}

.gradio-container .prose h1 {
    font-size: 1.4em !important;
}
.gradio-container .prose h2 {
    font-size: 1.2em !important;
}
.gradio-container .prose h3 {
    font-size: 1.05em !important;
}
.gradio-container .prose h4 {
    font-size: 1.0em !important;
}
.gradio-container .prose h5 {
    font-size: 1.0em !important;
}
.gradio-container .prose h6 {
    font-size: 0.9em !important;
}

/* 图片上传区域固定为 80x80 */
.image-upload-box,
.image-upload-box .image-container {
    width: 80px !important;
    height: 80px !important;
    min-width: 80px !important;
    min-height: 80px !important;
    max-width: 80px !important;
    max-height: 80px !important;
}

.image-upload-box img {
    width: 80px !important;
    height: 80px !important;
    object-fit: cover !important;
}

/* 调整上传按钮的字体大小 */
.image-upload-box .upload-text {
    font-size: 12px !important;
}
"""

async def chat(message, history, session_id):
    """处理对话 - 异步流式版本，实时输出（带节流优化）

    返回完整的更新后历史列表，确保历史消息不会丢失
    注意：history 应该已经包含用户消息（由调用方添加）
    """
    if not session_id:
        yield [{"role": "assistant", "content": "会话异常，请刷新页面"}]
        return

    try:
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def run_agent():
            try:
                for event in agent_loop(message.strip(), session_id):
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            except Exception as e:
                LOG.exception('agent_loop error')
                loop.call_soon_threadsafe(queue.put_nowait, {
                    "type": StreamEvent.PROCESS,
                    "content": f"\n❌ 处理错误: {str(e)}"
                })
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(None, run_agent)

        process_text = ""  # 确定的过程内容
        pending_text = ""  # 待定内容（可能是答案）
        last_output = ""  # 上次输出的内容，用于检测变化
        last_yield_time = 0  # 上次 yield 的时间戳
        min_yield_interval = 0.05  # 最小更新间隔（秒），30ms 节流

        def build_output():
            """构建当前输出内容"""
            output = ""
            if process_text.strip():
                output += f"```\n{process_text.strip()}\n```"
            if pending_text:
                if process_text.strip():
                    output += "\n"
                output += pending_text
            return output

        while True:
            event = await queue.get()
            if event is None:
                break

            if isinstance(event, dict):
                event_type = event.get("type", "")
                content = event.get("content", "")

                if event_type == StreamEvent.PROCESS:
                    # 确定的过程内容，加入 process_text
                    if pending_text:
                        process_text += pending_text
                        pending_text = ""
                    process_text += content
                elif event_type == "text":
                    # 待定文本
                    pending_text += content
                elif event_type == StreamEvent.ANSWER:
                    # 直接答案（命令结果等）
                    pending_text = content
            else:
                pending_text += str(event)

            # 节流控制：检查是否需要更新 UI
            current_time = asyncio.get_event_loop().time()
            current_output = build_output()

            # 仅在内容变化且满足时间间隔时才更新
            if current_output != last_output:
                time_since_last_yield = current_time - last_yield_time

                # 如果距离上次更新超过阈值，或者内容长度变化较大，则触发更新
                content_growth = len(current_output) - len(last_output)
                should_yield = (
                    time_since_last_yield >= min_yield_interval or
                    content_growth >= 50  # 累积 50 个字符也触发
                )

                if should_yield:
                    last_output = current_output
                    last_yield_time = current_time
                    updated_history = history.copy()
                    updated_history.append({"role": "assistant", "content": current_output})
                    yield updated_history

        # 确保最终内容被 yield（可能有节流未发出的最新内容）
        final_output = build_output()
        if final_output != last_output or final_output:
            updated_history = history.copy()
            updated_history.append({"role": "assistant", "content": final_output})
            yield updated_history

    except Exception as e:
        LOG.exception('chat error')
        history.append({"role": "assistant", "content": f"处理请求时发生错误: {str(e)}"})
        yield history


def run():
    # 创建界面
    with gr.Blocks(fill_height=True, title='👻 鬼才 - 你的个人猪理') as webui:
        gr.Markdown("# 👻 鬼才")

        # 隐藏组件存储session
        session_box = gr.Textbox(visible=False, elem_id="session_id")

        # 使用更低级的组件来完全控制历史消息
        with gr.Column(scale=1, min_width=100):
            chatbot = gr.Chatbot(
                label="对话",
                height="calc(100vh - 235px)",
                min_height="400px",
                show_label=False,
                autoscroll=True,
            )

            # 输入区域（包含图片上传、文本输入、发送按钮）
            with gr.Row(variant="compact"):
                # 图片上传组件（上传后显示图片预览）
                image_upload = gr.Image(
                    sources=["upload"],
                    type="filepath",
                    height=80,
                    label=None,
                    show_label=False,
                    interactive=True,
                    elem_classes=["image-upload-box"],
                )

                msg_input = gr.Textbox(
                    label="发送消息",
                    placeholder="请输入消息...",
                    show_label=False,
                    lines=1,
                    max_lines=4,
                    scale=10,
                    elem_classes=["auto-resize-input"],
                )

                with gr.Column(scale=1, min_width=120):
                    with gr.Row():
                        submit_btn = gr.Button("发送", variant="primary", scale=1)
                        clear_btn = gr.Button("清除", variant="secondary", scale=1)

        # 构建客户端函数：发送消息
        async def submit_message(message, history, session_id, image_path):
            history = history or []

            # 使用保存到 temp/ 目录的图片路径
            saved_image_path = _current_image_path.get(session_id, "")
            has_image = bool(saved_image_path)

            if not message.strip() and not has_image:
                yield history, "", None
                return

            # 如果有上传的图片，先获取图片描述
            final_message = message.strip() if message else ""

            if saved_image_path:
                # 使用 loading 状态显示图片处理中
                display_text = f"📷 [正在分析图片...]\n{final_message}" if final_message else "📷 [正在分析图片...]"
                history.append({"role": "user", "content": display_text})
                yield history.copy(), "", gr.update()

                try:
                    LOG.info(f"正在获取图片 {saved_image_path} 的描述...")
                    image_description = get_image_info(saved_image_path)

                    if image_description:
                        if final_message:
                            final_message = f"用户上传了图片, 图片描述如下:\n```\n{image_description}\n```\n\n用户的问题是:\n```\n{final_message}\n```\n"
                        else:
                            final_message = f"用户上传了图片, 图片描述如下:\n```\n{image_description}\n```"
                        LOG.info(f"图片描述获取成功: {image_description[:100]}...")
                    else:
                        # 描述获取失败，但继续处理文本
                        if final_message:
                            final_message = f"用户上传了图片, 但无法获取图片描述。\n\n用户的问题是:\n{final_message}"
                        else:
                            final_message = "用户上传了图片, 但无法获取图片描述。"
                        LOG.warning("图片描述获取失败")

                    # 更新最后一条用户消息
                    history[-1] = {"role": "user", "content": message.strip() if message else "📷 [图片]"}

                except Exception as e:
                    LOG.exception(f"图片处理出错: {e}")
                    history[-1] = {"role": "user", "content": message.strip() if message else "📷 [图片]"}
                    if final_message:
                        final_message = f"用户上传了图片, 但图片处理出错: {str(e)[:50]}。\n\n用户的问题是:\n{final_message}"
                    else:
                        final_message = f"用户上传了图片, 但图片处理出错: {str(e)[:50]}。"

            # 先立即显示用户输入的消息（如果没有图片，或作为更新）
            if not history or history[-1]["role"] != "user":
                history.append({"role": "user", "content": message.strip() if message else "📷 [图片]"})

            # 添加临时的 loading 消息（带动画）
            history.append({"role": "assistant", "content": '<div class="loading-indicator"><span class="loading-dots"><span></span><span></span><span></span></span><span class="loading-text">思考中</span></div>'})
            yield history.copy(), "", gr.update()

            # 流式输出助手回复（移除 loading 消息）
            async for updated_history in chat(final_message, history[:-1], session_id):
                yield updated_history, "", gr.update()

            # 成功处理后清除图片组件和临时文件
            if has_image:
                clear_session_images(session_id)
                if session_id in _current_image_path:
                    del _current_image_path[session_id]
                yield gr.update(), "", None

        # 点击发送按钮或按回车触发
        submit_event = msg_input.submit(
            fn=submit_message,
            inputs=[msg_input, chatbot, session_box, image_upload],
            outputs=[chatbot, msg_input, image_upload],
        )

        submit_btn.click(
            fn=submit_message,
            inputs=[msg_input, chatbot, session_box, image_upload],
            outputs=[chatbot, msg_input, image_upload],
        )

        # 清除对话（同时清除临时图片）
        def on_clear_conversation(session_id):
            clear_session_images(session_id)
            if session_id in _current_image_path:
                del _current_image_path[session_id]
            return [], "", None

        clear_btn.click(
            fn=on_clear_conversation,
            inputs=[session_box],
            outputs=[chatbot, msg_input, image_upload],
        )

        # 使用 Textbox 作为中介，JS设置值然后触发 change 事件
        js_session_input = gr.Textbox(visible=False, label="js_session")

        def on_session_change(session_id_from_js):
            """当JS设置了session_id后触发"""
            LOG.info(f"on_session_change 被调用，session_id='{session_id_from_js}'")
            if not session_id_from_js:
                return session_id_from_js, []
            history = format_history_from_memory(session_id_from_js, rounds=20)
            LOG.info(f"加载了 {len(history) // 2} 对历史消息")
            return session_id_from_js, history

        # JS 设置 session_id 到隐藏输入框
        webui.load(
            fn=None,
            inputs=None,
            outputs=js_session_input,
            js="() => { const sid = new URL(window.location.href).searchParams.get('session') || ''; console.log('JS获取session:', sid); return sid; }"
        )

        # 当输入框值变化时，加载历史消息
        js_session_input.change(
            fn=on_session_change,
            inputs=js_session_input,
            outputs=[session_box, chatbot],
        )

        # 图片上传事件（用户点击X删除图片时也会触发，此时 value=None）
        def on_image_change(image_path, session_id):
            """当图片上传或删除时触发 - 保存到 temp/{session_id}/"""
            if not image_path:
                # 图片被删除，清除临时文件
                clear_session_images(session_id)
                if session_id in _current_image_path:
                    del _current_image_path[session_id]
                return

            if not session_id:
                return

            # 保存图片到 temp/{session_id}/
            saved_path = save_image_to_temp(image_path, session_id)
            if saved_path:
                _current_image_path[session_id] = saved_path
                LOG.info(f"图片已保存: {saved_path}")

        image_upload.change(
            fn=on_image_change,
            inputs=[image_upload, session_box],
            outputs=None,
        )

    # 启动配置
    webui.queue(
        max_size=40,                    # 队列最大等待数
        default_concurrency_limit=20    # 最大并发数
    )

    webui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        head=SESSION_REDIRECT_JS,
        css=COMPACT_CSS,
    )