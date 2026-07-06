
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

## searxng搜索说明

默认配置了vpn代理，端口是 **10902**。

### 启动 searxng docker 容器
cd <project>/docker/searxng

docker compose up -d

### 重启searxng容器
cd <project>/docker/searxng

docker compose -f docker-compose.yml restart 2>&1

### 测试searxng搜索结果
curl 'http://localhost:8182/search?q=test&format=json'

