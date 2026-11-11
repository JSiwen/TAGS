# Project collaboration preferences

- After authorized code changes in this project, commit and push the task-related changes to the bound GitHub repository (currently remote `github`, https://github.com/JSiwen/TAGS.git), without asking for confirmation again. This preference was explicitly requested by the user on 2026-10-29.
- Keep unrelated local changes, datasets, checkpoints, credentials and generated training outputs out of these commits. Include required code dependencies and relevant usage documentation.
- Use a `codex/` branch when remote integration is uncertain; never force push or overwrite unrelated remote work. Verify the remote commit hash before reporting successful synchronization.
- Report checks performed and unverified GPU/training boundaries. Do not launch long training jobs merely to synchronize code.

## TAGS experiment stage-gate workflow

- For every TAGS experiment planning, execution, status review, or transition between experiment stages, automatically read and follow `.agents/skills/tags-experiment-stage-gate/SKILL.md`. Apply it without waiting for the user to invoke the skill by name. The user may explicitly ask to skip or narrow this workflow for a particular task.
