---
name: Live chart engine scope
description: The intended scope for live market charts and market/timeframe controls in the AI Trading Debate app.
---

Keep the live candlestick chart and AI forecast engine. Remove only the manual market/currency and timeframe selectors. Infer supported market and interval from the trader's Extra Context or existing chart analysis, and pass the full context—including quoted information and direct questions—to every live model round.

**Why:** The user clarified that removing market/timeframe selection was not a request to remove live charts; deleting the engine was an overbroad interpretation.

**How to apply:** Preserve live candles, indicators, forecast overlays, and existing market-data/optional MT5 flows when changing this app. Do not add manual market or timeframe selectors unless the user explicitly asks for them. Treat trader-provided context as authoritative.
