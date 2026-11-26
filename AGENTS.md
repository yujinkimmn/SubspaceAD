# Repository development checks

- Whenever Python code changes, check all added, removed, or changed third-party imports against the affected model's `requirements.txt` (including any platform-specific requirements). Keep package names and compatible version constraints aligned; do not add standard-library or local modules as dependencies.
- Treat each model's environment as independent. The root `requirements.txt` delegates to SubspaceAD; use `models/csam_void/requirements.txt` for CSAM and `models/nagl/requirements*.txt` for NAGL. Do not combine unrelated model requirements or silently upgrade PyTorch, torchvision, CUDA, or NumPy in existing user environments.
- Verify dependency resolution in an isolated environment, run `python -m pip check`, and run the affected model's tests or smoke test. If a check cannot be performed, report that explicitly. Mention any requirements changes and new setup commands alongside the code changes.
