# SearxNG 搜索服务

自托管元搜索引擎，聚合 google / bing / duckduckgo / baidu 多源结果，供 `app/browser/searxng_search.py` 调用。

## 启动

```bash
cd docker/searxng
docker compose up -d
```

启动后验证 JSON API 可用：

```bash
curl 'http://localhost:8080/search?q=test&format=json'
```

返回 JSON 即正常。

## 代理

`settings.yml` 中 `outgoing.proxies` 指向宿主机代理 `http://host.docker.internal:10901`。

- 容器内无法通过 `127.0.0.1` 访问宿主机，必须用 `host.docker.internal`
- 代理端口变化时，修改 `settings.yml` 后执行 `docker compose restart`

## 切换搜索引擎

在项目根 `.env` 中：

```
web_search_engine = "searxng"
```

## 排查

- 查看日志：`docker compose logs -f searxng`
- 某引擎无结果：多为该引擎被源站限流，SearxNG 会自动降级，其他引擎仍返回结果
- 连接代理失败（502/超时）：确认宿主机代理 10901 已开启
- 404 / 无 JSON 返回：确认 `settings.yml` 中 `search.formats` 包含 `json`
