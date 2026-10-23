# RoTA-Bench

<p align="center"><a href="README.md">English</a> · <a href="README_zh.md">简体中文</a></p>

<p align="center"><img src="assets/rota-bench-arcade-EN-preview.gif" alt="RoTA-Bench promo animation" width="920"></p>

<p align="center"><a href="https://www.modelscope.cn/studios/san2333333/rotabench-anon?mode=full&app_path=%2Fportal"><strong>▶ Open the interactive demo</strong></a></p>

<p align="center"><a href="https://github.com/MAC-AutoML/RoTA-Bench"><strong>⌘ Browse the code</strong></a> · <a href="https://arxiv.org/abs/2610.17665v1"><strong>Read the paper</strong></a> · <a href="https://huggingface.co/papers/2610.17665"><strong>Hugging Face Papers</strong></a></p>

Code and the fixed 674-task dataset accompanying **RoTA-Bench: Evaluating 3D Spatial Reasoning through Interactive Projection Alignment**.

The public release contains 674 tasks. Two tasks (`single_00345` and `single_00384`) were removed from the original 676-task collection because their icons could not be redistributed due to copyright restrictions.

**Paper:** [RoTA-Bench: Evaluating 3D Spatial Reasoning through Interactive Projection Alignment](https://arxiv.org/abs/2610.17665v1) · [Hugging Face Papers](https://huggingface.co/papers/2610.17665)<br>
**Authors:** Qingchuan Ma, Yuexiao Ma, Xiang Meng, Yixuan Zou, Tianyu Xie, Wang Chen, Yuhui Zeng, Yuhang Wu, Xiawu Zheng, Rongrong Ji.

## Dataset on Hugging Face

The fixed 674-task dataset is also available on [Hugging Face Datasets](https://huggingface.co/datasets/QingchuanMa/RoTA-Bench) and can be loaded with 🤗 Datasets:

```python
from datasets import load_dataset

dataset = load_dataset("QingchuanMa/RoTA-Bench")
```

## Benchmark Overview

RoTA-Bench evaluates an agent's ability to align a target 2D projection through a sequence of discrete 3D rotations. Each task provides a target silhouette, an auxiliary view of the object, and its current projection. The agent selects from a predefined set of rotation actions and continues until the projected contour aligns with the target.

<p align="center">
  <img src="website/assets/rota-overview.png" alt="RoTA-Bench benchmark overview" width="980">
</p>

*Benchmark overview: interactive observations, rotation actions, projection alignment, and the resulting evaluation signal.*

<p align="center">
  <img src="website/assets/rota-pipeline.png" alt="RoTA-Bench construction and evaluation pipeline" width="980">
</p>

*Construction and evaluation pipeline: 3D object generation, reachable reference-path search, frozen task-split construction, and closed-loop agent evaluation.*

The website provides an interactive leaderboard and task browser. This repository contains the evaluation code, task metadata, and fixed dataset for reproducible comparisons.

## Usage

Clone with Git LFS to download the dataset:

```bash
cd RoTA-Bench
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build compatible with your NVIDIA driver from https://pytorch.org/get-started/locally/. Formal evaluation and Beam search require CUDA.

Set `OPENAI_API_KEY` to your API key. In Bash use `export OPENAI_API_KEY="your-key"`; in PowerShell use `$env:OPENAI_API_KEY="your-key"`.

Run the frozen 100-task evaluation using an OpenAI-compatible endpoint:

```bash
python src/step_02_test_models.py --backend gemini_api --config data/run_config.json --formal-config src/final_benchmark_config.json --tasks-file data/tasks_674.jsonl --split-file data/test_100.json --base-url https://YOUR-ENDPOINT/v1 --api-model YOUR-MODEL --model-name YOUR-MODEL --metric-feedback-mode none --max-step-cap 48 --output-dir result/evaluation
```

To regenerate tasks from the original 2D icons, install the additional `cairosvg` dependency and run `src/step_01_generate_data.py`. The script downloads the upstream [game-icons](https://github.com/game-icons/icons) SVG repository when `data_sources/game-icons` is absent, rasterizes each SVG into deterministic 2D points, applies the icon-quality filters, constructs the spherical 3D geometry, renders the target and initial views, and writes task JSONL records. The game-icons assets remain under their upstream Creative Commons/public-domain licenses; review `data_sources/game-icons/license.txt` and the upstream attribution requirements before redistributing them.

### Reproducible pipeline, step by step

The following commands run the public pipeline without changing the frozen `data/tasks_674.jsonl` or `data/test_100.json` files. Generation outputs are written under `tasks/`, `data/single/`, `reports/`, and `logs/`.

1. Install the runtime dependencies. For CUDA rendering, install a CUDA-enabled PyTorch build compatible with your driver separately.

   ```bash
   pip install -r requirements.txt
   ```

2. Obtain the source icons. Step 1 performs this automatically by cloning `game-icons/icons` into `data_sources/game-icons` if that directory is absent. You may also clone it explicitly:

   ```bash
   git clone --depth 1 https://github.com/game-icons/icons data_sources/game-icons
   ```

   The upstream SVG files retain their original licenses and attribution requirements; they are not part of this repository.

3. Convert SVG icons to deterministic 2D points and generate tasks. This command also constructs the spherical 3D geometry, renders target/current/oblique views, and writes `tasks/tasks.jsonl`.

   For a one-task smoke test, first make a copy of the run configuration and limit the icon pool. The default `icon_pool_limit: -1` scans the full upstream icon repository and is intended for full regeneration; it can require substantial disk space even when `--sample-count 1` is used:

   ```bash
   python -c "import json; from pathlib import Path; p=Path('data/run_config.json'); q=Path('data/run_config_smoke.json'); cfg=json.loads(p.read_text(encoding='utf-8')); cfg.update({'icon_pool_limit':1,'sample_count':1}); q.write_text(json.dumps(cfg, indent=2)+'\n', encoding='utf-8')"
   ```

   ```bash
   python src/step_01_generate_data.py --root . --config data/run_config_smoke.json --sample-count 1 --save-init-images --save-init-oblique-images --no-resume
   ```

   Target images are saved by default (use `--no-save-target-images` only when they are not needed). The generated task contains `icon_points` (the sampled 2D shape), pose metadata, and the parameters used to construct the 3D object. Generated images are saved under `data/single/target/`, `data/single/init/`, and `data/single/init_oblique/`. Use a separate generation config or output checkout when producing new data at scale; do not overwrite the frozen release files.

4. Inspect the generated task and confirm that the expected sample exists:

   ```bash
   python -c "import json; from pathlib import Path; rows=[json.loads(x) for x in Path('tasks/tasks.jsonl').open(encoding='utf-8') if x.strip()]; print(len(rows), rows[0]['sample_id'], len(rows[0]['icon_points']))"
   ```

5. Create a split file for the generated task. For a one-task smoke test, the split must list the generated `sample_id` explicitly:

   ```bash
   python -c "import json; from pathlib import Path; sid=next(json.loads(x)['sample_id'] for x in Path('tasks/tasks.jsonl').open(encoding='utf-8') if x.strip()); Path('tasks/smoke_split.json').write_text(json.dumps({'split_name':'smoke','count':1,'sample_ids':[sid]},indent=2)+'\\n',encoding='utf-8')"
   ```

6. Run the local simulated evaluator on the generated task. This validates the complete generated-task → renderer → closed-loop evaluator path without an API key:

   ```bash
   python src/step_02_test_models.py --root . --config data/run_config.json --tasks-file tasks/tasks.jsonl --split-file tasks/smoke_split.json --backend simulated --max-tasks 1 --max-step-cap 48 --task-workers 1 --render-workers 1 --output-dir result/smoke_generated --no-resume
   ```

   The report is written to `result/smoke_generated/reports/model_raw.csv`; per-step frames and the trajectory are under `result/smoke_generated/logs/`.

7. For an API-backed evaluation, keep the same generated task and split paths, set `OPENAI_API_KEY`, and replace `--backend simulated` with `--backend gemini_api`, supplying `--base-url`, `--api-model`, and the required formal protocol options. The frozen benchmark command below remains the recommended command for official comparisons.

Run Beam-256 reference search:

```bash
python src/guided_iou_beam.py --tasks-file data/tasks_674.jsonl --beam-widths 256 --output-dir result/beam256
```

Use `--help` for additional options. The 100-task test IDs, remaining 574-task IDs, and saved Beam/BFS paths are in `data/`.

## Task Example

`single_00627`: target projection, current auxiliary view, and current projection.

![Projection-alignment example](assets/single_00627_panel.png)

The left panel shows the target silhouette, the middle panel shows the current 3D object, and the right panel shows its current projection. A successful trajectory aligns the current projection with the target while minimizing the number of rotation steps.

## Interactive Results and Resources

The accompanying website provides interactive leaderboard tables and cost–performance plots. Hover over a point to inspect the model, benchmark split, and metric values. Use the task browser to replay individual trajectories.

## Online demo

[Open the interactive demo](https://www.modelscope.cn/studios/san2333333/rotabench-anon?mode=full&app_path=%2Fportal) — powered by a free CPU Studio on ModelScope.
