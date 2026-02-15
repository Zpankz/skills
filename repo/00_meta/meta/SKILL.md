---
name: meta
description: "Emit, validate, and version-control deterministic governance artifacts for the local skill corpus, including DAG/ontology outputs, control scores, and bridge maps. Use this skill when rebuilding `.index` outputs or validating cross-skill topology."
license: MIT
compatibility: "Designed for the Context-Engineering corpus; portable across Claude Code, Codex, Gemini, and AgentSkills-compatible runners."
allowed-tools: "Bash Python Read Write Execute"
metadata:
  context: fork
  model: opus
  entry-point: .index/interfaces/emit_index.py
  agent: governance
  supported-platforms:
    - claude
    - codex
    - gemini
    - agentskills
    - openai
  tags:
    - governance
    - control-plane
    - index-emission
    - dag
    - ontology
    - optimization
  user-invocable: true
---
# Meta Skill

Use this skill to emit and validate the entire control-plane output for the skill corpus.

## Fast path

```bash
python .index/interfaces/emit_index.py
```

## Canonical flow

1. Validate required corpus artifacts and schema contracts in `.index`.
2. Resolve aliases, dependencies, SCC structure, and control ontology mappings.
3. Emit canonical governance artifacts including `.index/main_index.yaml`, `.index/main_index_payload.json`, and `.index/skill_graph.json`.
4. Validate hard/soft edge topology and compute MCTSR quality checks.
5. Persist deterministic quality and processing reports for auditability.

## Runtime dependencies

- Hard dependency: `system-skill`, `skill-orchestrator`.
- Soft compatibility references: `skill-protocol`, `skill-updater`.

## Platform compatibility metadata

- `agents/openai.yaml` for Codex/agent interfaces.
- `agents/claude.json` for Claude-style consumers.
- `agents/codex.json` for Codex runtime hooks.
- `agents/gemini.json` for Gemini CLI/assistant integration.
- `agents/agent-skills.json` for open-source AgentSkills-compatible loaders.

## Inputs and outputs

### Inputs

- `.index/main_index_payload.json`
- `.index/main_index.yaml`
- `.index/skill_graph.json`
- `.index/clustered_graph.json`
- `.index/control_ontology.json`
- `.index/control_ontology.schema.json`
- `.index/hyperedges.json`
- `.index/bridge_index.json`
- `.index/bridge_candidates.csv`
- `.index/quality_report.md`
- `.index/processing_report.md`

### Outputs

- `.index/main_index.yaml`
- `.index/main_index_payload.json`
- `.index/quality_report.md`
- `.index/processing_report.md`

## Notes

- This skill is designed for deterministic index emission and control-plane validation; it does not modify source skill definitions.
- Outputs are emitted in deterministic order to minimise drift across runs.
