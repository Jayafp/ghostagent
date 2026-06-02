
个人练手用的AI助理，包含：上下文管理、Agent记忆、Sandbox、Skill能力、工具以及MCP等。

## 环境准备
- pip install -r requirements.txt
- 修改.env，配置api_key和相关路径
- 【可选】安装并启动docker，作为Bash、读写文件的沙箱容器

## 启动项目
```Bash
python3 main.py
```
访问 http://127.0.0.1:7860/