---
name: Model roster runtime
description: Provider catalogs can include non-chat entries and repeated vision calls can trigger rate limits.
---

Only activate provider catalog entries that can answer chat completions; exclude audio, moderation, guard, embedding, and generation-only entries. Reuse the primary vision result when the same active vision model is later asked for its vote, and use bounded retry for transient provider rate limits. Count that reused vision vote exactly once in the final tally.

**Why:** Live provider `/models` catalogs contain callable-looking entries that reject chat requests, while sending the same vision image twice can turn a healthy model into a 429 failure during one debate. Counting the vision result twice can also override a clear majority with an inconsistent LLM synthesis.

**How to apply:** Keep the active roster capability-aware, distinguish direct-image models from text-only models, and make every displayed active model correspond to one answer-producing path.