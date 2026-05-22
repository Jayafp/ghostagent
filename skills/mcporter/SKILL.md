---
name: mcporter
description: Use the mcporter CLI to list, configure, auth, and call MCP servers/tools directly (HTTP or stdio), including ad-hoc servers, config edits, and CLI/type generation.
homepage: http://mcporter.dev
metadata:
  {
    "openclaw":
      {
        "emoji": "📦",
        "requires": { "bins": ["mcporter"] },
        "install":
          [
            {
              "id": "node",
              "kind": "node",
              "package": "mcporter",
              "bins": ["mcporter"],
              "label": "Install mcporter (node)",
            },
          ],
      },
  }
---

# mcporter

Use `mcporter` to work with MCP servers directly.

## ⚠️ 调用前必须执行

**每次调用 MCP 工具前，必须先执行：**
```
mcporter list <server> --schema
```
查看可用的工具名称和参数，然后用返回的 **精确名称** 调用。

**禁止**：凭记忆猜测工具名或参数名。

## 命令格式规范

### 调用工具（必须遵守此格式）
```
mcporter call <server>.<tool> key=value
```

**格式规则：**
- `<server>` 和 `<tool>` 之间用 `.` 连接（一个点，不是空格、`::`、`/`）
- 参数格式：`key=value`（等号，无空格）
- 字符串值：`key="value with spaces"` 或 `key=value`
- 多参数：空格分隔 `key1=value1 key2=value2`

### 常见错误 ❌ → 正确 ✅

| 错误写法 | 正确写法 |
|---------|---------|
| `mcporter call skylark resolve_url` | `mcporter call skylark.skylark_resolve_url` |
| `mcporter call skylark::resolve_url` | `mcporter call skylark.skylark_resolve_url` |
| `mcporter call skylark --tool resolve_url` | `mcporter call skylark.skylark_resolve_url` |
| `mcporter call skylark.skylark_resolve_url url = value` | `mcporter call skylark.skylark_resolve_url url=value` |

## Quick start

- `mcporter list`
- `mcporter list <server> --schema`
- `mcporter call <server>.<tool> key=value`

## Examples

- `mcporter list`
- `mcporter list skylark --schema`
- `mcporter call skylark.skylark_resolve_url url="https://yuque.antfin.com/jiaye.wwh/iot/ed07dgv947nx2lh9"`

Call tools

- Selector: `mcporter call linear.list_issues team=ENG limit:5`
- Function syntax: `mcporter call "linear.create_issue(title: \"Bug\")"`
- Full URL: `mcporter call https://api.example.com/mcp.fetch url:https://example.com`
- Stdio: `mcporter call --stdio "bun run ./server.ts" scrape url=https://example.com`
- JSON payload: `mcporter call <server.tool> --args '{"limit":5}'`

Auth + config

- OAuth: `mcporter auth <server | url> [--reset]`
- Config: `mcporter config list|get|add|remove|import|login|logout`

Daemon

- `mcporter daemon start|status|stop|restart`

Codegen

- CLI: `mcporter generate-cli --server <name>` or `--command <url>`
- Inspect: `mcporter inspect-cli <path> [--json]`
- TS: `mcporter emit-ts <server> --mode client|types`

Notes

- Config default: `./config/mcporter.json` (override with `--config`).
- Prefer `--output json` for machine-readable results.