# HEU public archive extraction specification

## Request

Create and publish `Bakobiibizo/heu` as an MIT-licensed archive extracted from the first HEU experiments in the private `hydra-dynamix/HH-Envelope-Model` repository.

The operator selected the focused archive option on 2026-02-19: paper and figures, six demonstration notebooks, HEU experiments 01–05, generated results, and the code required to inspect and run those experiments. The unrelated groove-detector, broad transcriber, curiosity, Goldilocks, and later topology programs are out of scope.

## Source

- Source repository: `hydra-dynamix/HH-Envelope-Model` (private)
- Source revision: `dc838469820431f796319d81f7cac0e7d191501d`
- Archived material: `_depreciated/` (source spelling)
- Paper and Experiment 05 additions: current source tree at the same revision

## Included surfaces

- `docs/paper/`: paper draft, component sections, figure generator, and figures
- `notebooks/`: notebooks 00–05 and notebook utilities
- `hh_envelope/`: runtime modules required by Experiments 01–05
- `hh_transcriber/experiments/diagnostics/`: diagnostics imported by the archived experiment scripts
- `results/`: experiment 01–05 diagnostics, arrays, and plots
- Experiment CLI, packaging metadata, provenance, license, and public-facing README

Notebook outputs and generated result files are retained as historical evidence. A stale Notebook 02 import may be updated only to reflect the module move from `hh_envelope.polyphony.coupling` to `hh_envelope.coupling`; scientific outputs must not be rewritten.

## Acceptance criteria

1. The repository contains the requested focused archive and identifies its source revision.
2. It contains an MIT license and a clear README with paper, notebook, result, and reproduction links.
3. No credentials, private keys, environment files, or unrelated private-project metadata are committed.
4. Python sources compile and the available focused tests pass.
5. Paper figures regenerate from the committed experiment arrays.
6. A public GitHub repository exists at `https://github.com/Bakobiibizo/heu` and its `main` branch matches the validated local commit.
