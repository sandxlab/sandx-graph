# Contributing to sandx-graph

Thanks for your interest. This document covers how to set up a development environment, run tests, and submit a pull request.

---

## Development setup

```bash
git clone https://github.com/sandxlab/sandx-graph
cd sandx-graph
pip install -e ".[dev]"
```

For NetworkX export support:

```bash
pip install networkx
```

## Running tests

```bash
pytest tests/ -q
```

Tests requiring NetworkX are skipped automatically if it is not installed. To run the full suite including NetworkX tests:

```bash
pip install networkx && pytest tests/ -q
```

With coverage:

```bash
pytest tests/ --cov=sandx_graph --cov-report=term-missing -q
```

## Linting

```bash
ruff check src tests
```

We use `ruff` with `line-length = 100`. Fix lint errors before opening a PR — CI will reject anything that fails.

## Code style

- Type-annotate all public functions and methods.
- No comments explaining *what* code does. Only add a comment when the *why* is non-obvious (a hidden constraint, a workaround, a subtle invariant).
- Edge weights must be in `[0, 1]`. Functions that accept weights should validate this at system boundaries.
- Consensus scores must always be in `[0, 1]` with `1.0` meaning unanimous agreement.

## Before opening a PR

1. Tests pass: `pytest tests/ -q`
2. Lint passes: `ruff check src tests`
3. New behaviour has test coverage.
4. Consensus score invariants are preserved: isolated nodes return `1.0`, scores are bounded `[0, 1]`.

## Pull request process

- Branch off `main`. Name your branch `feat/short-description` or `fix/short-description`.
- Keep PRs focused. One logical change per PR.
- PR description should explain *why*, not just *what*.
- At least one approving review is required before merge.

## Reporting issues

Use the [GitHub issue tracker](https://github.com/sandxlab/sandx-graph/issues).

- **Bug reports:** include Python version, sandx-graph version (`pip show sandx-graph`), minimal reproducing example, and the full traceback.
- **Feature requests:** describe the use case, not just the feature. What problem does it solve?

## Design principles

sandx-graph sits downstream of sandx-er in the SandX pipeline. Its primary input is a resolved identity graph from the ER engine. Changes should preserve the composability of that pipeline. Graph algorithms that require external solvers or break the pure-Python constraint will require strong justification.

---

Apache 2.0 license. By contributing you agree your changes are released under the same license.
