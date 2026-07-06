#!/usr/bin/env bash
# 管理本地 SearxNG 容器：未运行则启动，已运行则重启（应用 settings.yml 改动）
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker/searxng/docker-compose.yml"

if [ "$(docker inspect -f '{{.State.Running}}' searxng 2>/dev/null)" = "true" ]; then
    echo "searxng 已在运行，重启中..."
    docker compose -f "$COMPOSE_FILE" restart
    echo "searxng 已重启"
else
    echo "searxng 未运行，启动中..."
    docker compose -f "$COMPOSE_FILE" up -d
    echo "searxng 已启动"
fi
