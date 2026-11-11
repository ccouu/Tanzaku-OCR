# AGENTS.md — Master Multi-Agent Routing & Orchestration Topology

## 1. Project Overview & Architecture Policy
- **Project Goal:** Mobile Vision Transformer (ViT) for Handwritten Japanese Text Recognition and Translation.
- **Orchestration Pattern:** Asymmetric top-down delegation managed by **Sol6** (Codex App) via **Herdr** (Agent-Native Rust Terminal Multiplexer over Unix Domain Socket IPC).
- **Core Security Directives:** Zero-knowledge proxy / environment key loading. NEVER output raw API keys (`GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`) or `.env` files into workspace context or git tree. Inject keys strictly at shell boundaries or from `~/.config/credentials/`.
- **KV Cache Optimization:** Keep system prompt prefixes and static tool definitions uniform at Token 0. Do NOT prepend dynamic timestamps or session UUIDs. Sub-agents run on distinct model families; treat each pane's KV cache as isolated.
- **Autonomous Vibe Coding & Permission Policy:** Sub-agent sessions (`codex` and `agy`) run in autonomous vibe coding mode. File modifications, directory operations, tests, and non-destructive shell commands within workspace boundaries are auto-approved without prompting. Agents MUST ONLY halt and request user permission if a dangerous or destructive action is detected (e.g., deletions outside workspace, modifying protected system/git paths, unvetted external network access, credential exposure, or privilege escalation).
- **Targeted Model Enforcement:** When spawning sub-agents in Herdr panes, explicit model flags (`-m` / `--model`) MUST be passed after `--` to prevent sub-agents from falling back to host defaults or triggering model mismatch warnings.
- **Reasoning Effort Standardization:** All sub-agent processes (`codex` and `agy`) must be initialized with reasoning effort set to `"medium"` (`-c model_reasoning_effort="medium"` for Codex, `--effort medium` for AGY).

---

## 2. Five-Agent Role Specification & Model Mapping

| Agent Alias | Logical Model | Target CLI Flag | Core Responsibility | Herdr Pane | Primary Skill File | Complete Invocation Arguments |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sol6** | Codex App | *(Host CLI)* | Master Orchestrator & Task Router | Main CLI / Host | `AGENTS.md` | Host process (`-a on-request -s workspace-write -c model_reasoning_effort="medium"`) |
| **agy** | Gemini 6 Argon | `--model gemini-4-argon-medium` | Research, Architecture, ViT Topology | `herdr-agy` | `.codex/skills/vision-vit-mobile.md` | `--mode auto --model gemini-4-argon-medium --effort medium` |
| **Terra** | Codex 6.1 Terra | `-m gpt-6.1-terra` | Core PyTorch/CoreML Code Implementation | `herdr-terra` | `.codex/skills/vision-vit-mobile.md` | `-a on-request -s workspace-write -m gpt-6.1-sol -c model_reasoning_effort="medium"` |
| **Flash** | DeepSeek 4.1 Flash | `--profile=deepseek` | Fast Static Analysis, Security & Code Review | `herdr-flash` | `.codex/skills/code-review-flash.md` | `-a on-request -s workspace-write --profile=deepseek -c model_reasoning_effort="medium"` |
| **Sol** | Codex 6 Sol | `-m gpt-6.1-sol` | Parameter Estimation & Mobile Latency Profiling | `herdr-sol` | `.codex/skills/estimation-profile.md` | `-a on-request -s workspace-write -m gpt-6-luna -c model_reasoning_effort="medium"` |

> *Note: Model identifiers (`-m` / `--model`) can be adjusted based on active account entitlements (e.g. `gemini-4-argon-high`, `gpt-5.3-codex`, or local OSS endpoints).*

---

## 3. Autonomous Execution & Dangerous-Action Permission Directives

To maintain seamless vibe coding throughput while preserving strict system safety:

### A. Codex Sessions (`Terra`,  `Flash`, `Sol6`, `Sol`)
- **Invocation Command:** Always start Codex panes with explicit flags `-a on-request -s workspace-write -c model_reasoning_effort="medium"` and model/profile options (e.g. `-m <model_id>` or `--profile=<profile>`).
- **Permission Mechanics:**
  - `approval_policy = "on-request"`
  - `sandbox_mode = "workspace-write"`
  - Direct human-in-the-loop review (safer than `--approve-for-me` auto-reviewer).
- **Reasoning Effort:** Configured to `"medium"` via `-c model_reasoning_effort="medium"`.
- **Auto-Approved:** Safe file creations, edits, formatting, builds, profiling runs, and unit tests strictly constrained inside the workspace directory (`d:\profolio\japanese_translation`).
- **Approval Required (Dangerous Action):**
  - Executing file deletions or writes outside the workspace root.
  - Modifying system files, system registries, or sensitive user directories.
  - Unsanitized package installations or foreign binary execution.
  - Commands requesting elevated privileges (`sudo`, administrative PowerShell, modifying security settings).
  - Exposure or extraction of authentication credentials and environment secrets.

### B. Antigravity (`agy`) Sessions (`agy` / Gemini 6 Argon)
- **Invocation Command:** Always start `agy` panes with `--mode auto --model <model_id> --effort medium`.
- **Permission Mechanics:**
  - `auto` mode automatically approves all workspace file edits and safe, reversible terminal commands inside the sandbox.
  - Does NOT prompt for routine code writes, token generation, or local inspections.
- **Reasoning Effort:** Configured to `"medium"` via `--effort medium`.
- **Approval Required (Dangerous Action):**
  - Irreversible or destructive file system operations (`rm -rf`, recursive wipe).
  - Terminal commands requiring unverified outbound network connections or escaping the sandbox boundary.
  - Actions flagged by the Antigravity safety filter as potentially high risk.

---

## 4. Herdr Multiplexer IPC Protocols

### Sub-Agent Lifecycle Commands

#### 1. Pane Allocation
Create an isolated background sibling pane for the sub-agent preserving working directory:
```bash
herdr pane split --current --direction right --cwd "$PWD" --no-focus
```

#### 2. Spawning Sub-Agents (with Targeted Models, Reasoning Effort & Permission Flags)
Pass native agent arguments strictly after `--`:

- **For Terra (Codex 6.1 Terra):**
  ```bash
  herdr agent start herdr-terra --kind codex --pane <pane-id> -- -a on-request -s workspace-write -m gpt-6.1-sol -c model_reasoning_effort="medium"
  ```

- **For agy (Gemini 6 Argon):**
  ```bash
  herdr agent start herdr-agy --kind agy --pane <pane-id> -- --mode auto --model gemini-4-argon-medium --effort medium
  ```

- **For Flash (DeepSeek 4.1 Flash via Codex Profile):**
  ```bash
  herdr agent start herdr-flash --kind codex --pane <pane-id> -- -a on-request -s workspace-write --profile=deepseek -c model_reasoning_effort="medium"
  ```

- **For sol (Codex 6 Sol):**
  ```bash
  herdr agent start herdr-sol --kind codex --pane <pane-id> -- -a on-request -s workspace-write -m gpt-6-luna -c model_reasoning_effort="medium"
  ```

#### 3. Task Dispatch
Submit prompts to sub-agents with skill ingestion references and synchronous wait:
```bash
herdr agent prompt <agent-alias> "Ingest .codex/skills/<skill-file>.md and perform task: <instructions>" --wait --timeout 180000
```

#### 4. State Monitoring & Dangerous Action Handling
- **`idle` / `done`**: Task finished. Safe to ingest output logs.
- **`working`**: Inference / execution active. Do NOT interrupt.
- **`blocked`**: **Dangerous action detected.** The agent halted and presented an approval prompt.
  - **Inspection:** Inspect the prompt using:
    ```bash
    herdr agent read <agent-alias> --source visible
    ```
  - **Resolution:** Sol6 evaluates if the action is legitimate. If safe and intentional, send confirmation; if dangerous or unexpected, send denial / cancel:
    ```bash
    herdr agent send-keys <agent-alias> "y" enter
    # Or to reject/cancel:
    herdr agent send-keys <agent-alias> "n" enter
    ```

#### 5. Output Extraction
Ingest unwrapped logs stripping ANSI formatting codes:
```bash
herdr agent read <agent-alias> --source recent-unwrapped --lines 150
```
