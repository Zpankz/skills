# Skill Refactor Index

This directory stores generated artifacts for the dependency and bridge model after reorganization.

- `main_index.yaml`: canonical path/dependency index with aliases and edge sets
- `skill_graph.json`: hard/soft edge graph, SCC and condensed view
- `clustered_graph.json`: condensed SCC graph payload
- `main_index_payload.json`: machine-readable aggregate index payload
- `bridge_index.json`: bridge manifest mapping and per-skill link paths
- `control_ontology.schema.json`: JSON schema for ontology validation
- `mctsr` metrics embedded in `main_index_payload.json` (`metrics.mctsr`, `metrics.mctsr_passed`, `metrics.mctsr_components`)
- `hyperedges.json`: workflow hyperedges for multi-input/output transformations
- `bridge_candidates.csv`: candidate semantic soft links
- `control_ontology.json`: control-plane relational ontology (nodes, relations, topology, atomicity)
- `processing_report.md`: run summary
- `quality_report.md`: metrics and cycle analysis
- `interfaces/emit_index.py`: helper script for index emission and validation
