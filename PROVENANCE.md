# Provenance

This public archive was produced in response to the request to extract the first HEU experiments for portfolio and CV use.

## Source identity

- Source repository: `hydra-dynamix/HH-Envelope-Model` (private)
- Source revision: `dc838469820431f796319d81f7cac0e7d191501d`
- Source revision date: 2026-02-08
- Archived source directory: `_depreciated/` (the spelling used upstream)
- Extraction date: 2026-08-29

## Selection

The operator chose a focused archive rather than publication of the entire deprecated tree. Included are:

- the paper draft and its figures;
- all six original demonstration notebooks and their embedded outputs;
- HEU Experiments 01–04 from the archived tree;
- Experiment 05, its paper figures, and its result arrays from the source root;
- the runtime modules and focused transcriber diagnostics imported by those experiments;
- generated diagnostics and plots for Experiments 01–05.

Excluded are unrelated groove-detector files, the broad transcription program and audio dataset, curiosity and Goldilocks studies, later topology/network work, private project instructions, and private repository history.

## Public-archive changes

The scientific source, saved arrays, diagnostics, paper text, and notebook outputs were retained. Public-facing changes are limited to:

1. adding `README.md`, `LICENSE`, `EXTRACTION_SPEC.md`, this provenance record, packaging metadata, a focused CLI, and Python CI;
2. narrowing dependencies and the `hh_transcriber` package initializer to the included diagnostics, including a NumPy `<2.0` compatibility bound because the archived diagnostics use `numpy.trapz`;
3. changing Notebook 02's stale import from `hh_envelope.polyphony.coupling` to its current location, `hh_envelope.coupling`;
4. excluding caches and local work-provenance databases.

No historical result was regenerated or replaced during extraction. Regenerated paper figures are byte-compared during validation; any difference is reported rather than silently accepted.
