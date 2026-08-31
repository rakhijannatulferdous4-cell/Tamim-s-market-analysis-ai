---
name: Published artifact routing
description: How this workspace selects services for published deployments
---

Publishing uses the registered artifact inventory and each registered artifact's production service definitions. A legacy or unregistered standalone Streamlit artifact can run in development while being omitted from the published process set.

**Why:** The published process list previously contained only the registered API artifact, so the live root served the wrong service even though the Streamlit workflow was healthy.

**How to apply:** Before diagnosing application code, compare `listArtifacts()` with the artifact TOMLs and confirm the published root service has a validated `[services.production.run]` and startup health path.