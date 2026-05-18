# Migration guide

This document tracks breaking changes between major releases. For new releases
the source of truth is `CHANGELOG.md`; this file is the narrative version with
worked examples.

## Upgrading from 0.x to 1.0

**TL;DR:** Nothing breaks. `1.0` is a stability commitment, not a redesign.
Every import path that worked in `0.5.0` still works in `1.0.0`.

### What changed

1. **Version classifier.** The `Development Status` classifier in
   `pyproject.toml` moved from `4 - Beta` to `5 - Production/Stable`. This is
   a signal to downstream packages that breaking changes will now follow
   semantic versioning.
2. **Top-level imports.** A curated public API is now re-exported from
   `space_ml_sim` itself. The submodule paths still work, but you can now write
   the shorter form:

   ```python
   # 0.x — still works in 1.0
   from space_ml_sim.compute.fault_injector import FaultInjector
   from space_ml_sim.environment.radiation import RadiationEnvironment

   # 1.0 — recommended for new code
   from space_ml_sim import FaultInjector, RadiationEnvironment
   ```

   The top-level import is lazy (PEP 562). `import space_ml_sim` does not pull
   in torch, numpy, or any heavy submodule until a name from that submodule is
   actually referenced.
3. **API stability snapshot.** `tests/test_public_api.py` now pins the public
   API surface. Any change to the set of public names will fail the test —
   forcing a conscious deprecation rather than an accidental break.
4. **Deprecation utility.** `space_ml_sim._deprecation.deprecated` is the
   sanctioned way to mark a symbol as deprecated for the 1.x cycle. Use it
   instead of ad-hoc `warnings.warn` calls so messages stay consistent.

### What did not change

- No public class signatures changed.
- No public function signatures changed.
- No constants were renamed.
- The CLI surface is identical.
- The on-disk file formats (CSV imports, ECSS report HTML, RTM exports) are
  unchanged.

If a 0.x notebook ran clean against `0.5.0`, it will run clean against
`1.0.0rc1`. Open an issue if you find a counter-example — that's a bug, not a
documented break.

### Deprecation policy going forward

Starting at 1.0:

- Breaking changes (removing a public name, changing a public signature, or
  changing observable behaviour) require a **major version bump** (2.0, 3.0).
- A symbol scheduled for removal in 2.0 will emit `DeprecationWarning` for at
  least one minor release (1.1, 1.2, ...) before removal.
- New symbols can be added in any minor release as long as they are additive
  and do not shadow an existing name.

The full contract lives in `ROADMAP.md` under the "v1.0 (Stability commitment)"
section.

## Future migrations

Each future major release (2.0, 3.0, ...) gets its own section in this file
when it ships. Until then, every 1.x release is a drop-in upgrade.
