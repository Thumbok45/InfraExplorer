# The Lab — Notes

Running notes for the local multi-agent AI project. Not related to Provenance Phase 2A.

---

## 2026-10-03 — Node failure handling

Design requirement: handle a node going down mid-task.

- If a worker node (3080 Ti or 4060) crashes mid-task, the orchestrator (5090) must retry or reroute the task, not hang.
- This is the part that turns a demo into something used every day.
- Hub-and-spoke topology: 5090 orchestrates, 3080 Ti handles medium tasks, 4060 handles light tasks (summarization, classification, quick lookups).
- Each machine runs LM Studio server mode, exposing an OpenAI-compatible endpoint on its LAN IP.
- All machines wired Ethernet.
- Hardware: 5090 (32 GB VRAM), 3080 Ti (12 GB VRAM, 32 GB DDR4 3200 system RAM), 4060 (8 GB VRAM). Dell Latitude 7214 dropped from the project.
