# Private ModelScope CPU Studio

The benchmark repository is public. The interactive demo Studio must stay
private. Do not add access tokens or credentials to this repository.

## Studio settings

- SDK: Gradio (or Docker only if the account is already eligible for Docker Studios)
- Visibility: Private
- Hardware: a free CPU option returned by ModelScope's hardware API
- Port: `7860`
- Entry point: `website/studio_app.py`

The full 674-task JSONL is stored in the repository with Git LFS. Before launch,
run `python website/prepare_tasks.py` to derive the 100-task demo catalog.
The site renderer is CPU-only. It does not need model API keys.

Use the `ms-studio-deploy` skill to query currently available free CPU hardware,
create the private Studio, sync code, deploy, and check logs. Supply
`MODELSCOPE_API_KEY` through the local environment or a Studio secret only; never
commit it. Do not change the Studio to public.
