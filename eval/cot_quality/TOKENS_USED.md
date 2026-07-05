# 令牌使用记录（CoT 质量审计）

平台：lingleap（new-api 网关），base_url = `https://api.lingleap.com/v1`

| 令牌名 | key | 用途 / judge | 实际可用模型 | 备注 |
|---|---|---|---|---|
| hzy-gpt | `sk-kOfYeodZ1sIaxDRPMiPF5GTLwHVsNjDHOVsGIbA42n7spLdf` | GPT judge | `gpt-5.4` / `gpt-5.4-xhigh` / `gpt-5.5`（200 可用）| 该令牌处于「GPT-官转」分组：gpt-4o 系报 Shell 渠道过期、gpt-5/5.1/5.2 报"无权访问 Azure 分组"，**仅 5.4/5.5 系可用**。Claude/Gemini 不在此组。 |

## 各阶段实际使用
- **2026-06-25 GPT 阶段**：用 `hzy-gpt` 令牌（当时在 GPT-官转 组）+ 模型 `gpt-5.5` 跑 400 条 GPT judge。结果：overall 75.9。
- **2026-06-25 Gemini 阶段**：同一把 `hzy-gpt` 令牌（用户已切到 Gemini 分组）+ 模型 `gemini-2.5-pro` 跑 400 条 Gemini judge。
  - 注：该令牌按需切换分组——同一 key 在不同时间属于不同分组（GPT-官转 → Gemini-x）。

## 待补（Claude / Gemini）
需另建令牌（分组按厂商隔离，一把覆盖不了三家）：
- Claude judge：令牌需在 `Claude-AWS` 组，模型 `claude-sonnet-4-5-20250929`
- Gemini judge：令牌需在 `Gemini-直连` 组，模型 `gemini-2.5-pro`
拿到后填 `NEWAPI_KEY_CLAUDE` / `NEWAPI_KEY_GEMINI`，断点续跑即可（已有 GPT 结果不重算）。
