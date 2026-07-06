#!/usr/bin/env bash
# 管理 SearxNG 容器：自动检测可用代理端口 → 更新 settings.yml → 启动/重启
# 代理端口检测: 扫描 10900-10909, 找到能访问境外的端口写入 settings.yml
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker/searxng/docker-compose.yml"
SETTINGS_FILE="$SCRIPT_DIR/docker/searxng/settings.yml"
PROXY_TEST_URL="https://www.google.com"

# 1. 自动检测可用代理端口（扫 10900-10909）
echo "检测可用代理端口 (10900-10909)..."
PROXY_PORT=""
for port in 10900 10901 10902 10903 10904 10905 10906 10907 10908 10909; do
    if curl -fL -x "http://127.0.0.1:$port" -sS -o /dev/null -m 4 "$PROXY_TEST_URL" 2>/dev/null; then
        PROXY_PORT=$port
        echo "  ✓ 端口 $port 可用"
        break
    fi
done

if [ -z "$PROXY_PORT" ]; then
    echo "  ⚠️  未找到可用代理端口（10900-10909 均不通），请确认 VPN/代理已开启"
    echo "     settings.yml 保持原配置继续启动"
else
    # 2. 更新 settings.yml 的代理端口（host.docker.internal:端口）
    CURRENT_PORT=$(grep -oE "host\.docker\.internal:[0-9]+" "$SETTINGS_FILE" | head -1 | sed 's/.*://')
    if [ "$CURRENT_PORT" = "$PROXY_PORT" ]; then
        echo "  settings.yml 代理端口已是 $PROXY_PORT，无需更新"
    else
        echo "  更新 settings.yml 代理端口: ${CURRENT_PORT:-未知} → $PROXY_PORT"
        sed -i '' -E "s|host\.docker\.internal:[0-9]+|host.docker.internal:$PROXY_PORT|" "$SETTINGS_FILE"
    fi
fi

echo ""

# 3. 启动或重启容器
if [ "$(docker inspect -f '{{.State.Running}}' searxng 2>/dev/null)" = "true" ]; then
    echo "searxng 已在运行，重启中..."
    docker compose -f "$COMPOSE_FILE" restart
    echo "searxng 已重启"
else
    echo "searxng 未运行，启动中..."
    docker compose -f "$COMPOSE_FILE" up -d
    echo "searxng 已启动"
fi
