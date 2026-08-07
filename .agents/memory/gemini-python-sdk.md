---
name: Gemini Python SDK
description: Compatibility guidance for Gemini features in this Python app.
---

Use the maintained `google-genai` SDK for Gemini requests and image inputs rather than the deprecated `google.generativeai` package.

**Why:** The older package is no longer receiving updates or bug fixes, while the maintained SDK is already available in the project runtime.

**How to apply:** For future Gemini text or vision work, initialize `google.genai.Client` with the configured secret and use the current `types` request objects.