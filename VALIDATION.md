# Extraction validation

Validated on 2026-08-29 against source revision `dc838469820431f796319d81f7cac0e7d191501d`.

## Results

| Check | Result | Evidence |
| --- | --- | --- |
| Focused archive inventory | Pass | 15 paper files, 6 valid notebook JSON files, 6 experiment entry scripts covering Experiments 01–05, and 79 result files |
| Python compilation | Pass | `python -m compileall -q hh_envelope hh_transcriber run.py docs/paper/generate_figures.py` |
| Focused tests | Pass | `uv run pytest -q`: **38 passed** |
| Package build | Pass | `uv build` produced an sdist and wheel |
| Hosted CI | Pass | GitHub Actions run [`33256436641`](https://github.com/Bakobiibizo/heu/actions/runs/33256436641) completed compile and test jobs successfully |
| Experiment 01 smoke run | Pass | Acceptance checks passed; PLV was 1.000 at 90, 120, and 140 BPM; all diagnostics were valid |
| Paper figure generation | Pass with reproducibility note | All four figures regenerated from `results/exp_04/figure_data.npz` and `results/exp_05/figure_data.npz` |
| Local documentation links | Pass | README and paper-local targets exist |
| Credential scan | Pass within checked archive | No suspicious filenames or common key/token/password patterns were found outside ignored local tooling and build directories |

## Hosted CI configuration finding

An intermediate maintenance commit used the latest `astral-sh/setup-uv` major as a moving tag (`@v10`). GitHub Actions could not resolve that tag, so run [`33256410301`](https://github.com/Bakobiibizo/heu/actions/runs/33256410301) failed during action setup before repository code ran. Pinning the published release tag `@v10.0.1` corrected the configuration; the subsequent hosted compile and test run passed. The failed run remains available as primary evidence in the Actions history.

## Dependency compatibility finding

The first Experiment 01 smoke run reached an accepted scientific result but then failed during diagnostics because NumPy 2.5 no longer provides `numpy.trapz`:

```text
AttributeError: module 'numpy' has no attribute 'trapz'. Did you mean: 'trace'?
```

The archive now declares `numpy>=1.26,<2.0`, matching the historical API used by the preserved diagnostics. With NumPy 1.26.4, the unchanged Experiment 01 source completed with exit code 0 and all diagnostics valid. The failure was not converted into a pass by changing an assertion or result.

## Figure reproducibility note

The figure generator completed successfully under Python 3.14.2, NumPy 1.26.4, and Matplotlib 3.11.1, but regenerated PNG encodings/dimensions were not byte-identical to the source artifacts. The source artifacts were restored rather than overwritten.

| Figure | Source SHA-256 | Regenerated SHA-256 |
| --- | --- | --- |
| `fig1.png` | `073b459aeb9514b714de1fb39ed85c7c6a36a7e375ea3fcea05d9f639769aa85` | `8a1c4092f4b312513a46b02124edb9bead72bbd61dd66c22110a0137a6978fc0` |
| `fig2.png` | `4521f88b1f2f5fc5911e759e72224fa53dd85da0c405411758eeb980f31e41d2` | `869eb83bab2298f7f22d6f38bf60011fdc13d10cbb236055cffe986bbfe200d2` |
| `fig3.png` | `cafe5e4121e871d3333a6d95d57f8852d9737396068afcdc6066635567ef2f0a` | `aa29c774ed98352944773684de7bd5343480b69b6de6a2db4fa4c2cdb3c2c958` |
| `fig4.png` | `4e5b53fc026368e368ea6614823306fef61995d4742ab265e5722920b3477e09` | `c6f7a00288647be7dce08eee5c1e4383d0f383b94323e90191a8d5442995620f` |

This validates that the committed arrays and generator are operational. It does not establish bit-for-bit reproducibility across plotting-library versions.
