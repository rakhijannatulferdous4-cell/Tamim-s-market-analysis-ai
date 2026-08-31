---
name: Streamlit long-run resilience
description: Why this app caches completed debates and tunes the Streamlit websocket for slow provider calls.
---

For long AI workflows, preserve the completed debate in Streamlit session state and render that cached result on reconnect/rerun instead of starting the analysis again. Use a single submit form for chart inputs, keep the websocket heartbeat enabled, and bound slow provider requests so one model cannot hold the page indefinitely.

**Why:** The proxied Streamlit preview can reconnect while a slow provider call is still running; without cached output, the UI appears to refresh and the finished result is lost.

**How to apply:** Keep provider calls concurrent where safe, render fast results as they complete, use provider-specific deadlines, and retain the last completed result for harmless reruns.