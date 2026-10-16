# RoTA-Bench

<p align="center"><a href="#english">English</a> · <a href="#简体中文">简体中文</a></p>

<a id="english"></a>

<p align="center">
  <a href="https://www.modelscope.cn/studios/san2333333/rotabench-anon?mode=full&app_path=%2Fportal"><img src="website/assets/demo-home.png" alt="RoTA-Bench interactive demo" width="920"></a>
</p>

<p align="center"><a href="https://www.modelscope.cn/studios/san2333333/rotabench-anon?mode=full&app_path=%2Fportal"><strong>▶ Open the interactive demo</strong></a></p>

<p align="center"><a href="https://github.com/MAC-AutoML/RoTA-Bench"><strong>⌘ Browse the code</strong></a> · <a href="https://arxiv.org/abs/2610.17665v1"><strong>Read the paper</strong></a> · <a href="https://huggingface.co/papers/2610.17665"><strong>Hugging Face Papers</strong></a></p>

Code and the fixed 674-task dataset accompanying **RoTA-Bench: Evaluating 3D Spatial Reasoning through Interactive Projection Alignment**.

The public release contains 674 tasks. Two tasks (`single_00345` and `single_00384`) were removed from the original 676-task collection because their icons could not be redistributed due to copyright restrictions.

**Paper:** [RoTA-Bench: Evaluating 3D Spatial Reasoning through Interactive Projection Alignment](https://arxiv.org/abs/2610.17665v1) · [Hugging Face Papers](https://huggingface.co/papers/2610.17665)<br>
**Authors:** Qingchuan Ma, Yuexiao Ma, Xiang Meng, Yixuan Zou, Tianyu Xie, Wang Chen, Yuhui Zeng, Yuhang Wu, Xiawu Zheng, Rongrong Ji.

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

---

<a id="简体中文"></a>

# RoTA-Bench（简体中文）

<p align="center"><a href="#english">English</a> · <a href="#简体中文">简体中文</a></p>

<p align="center">
  <a href="https://www.modelscope.cn/studios/san2333333/rotabench-anon?mode=full&app_path=%2Fportal"><img src="website/assets/demo-home.png" alt="RoTA-Bench 交互演示" width="920"></a>
</p>

<p align="center"><a href="https://www.modelscope.cn/studios/san2333333/rotabench-anon?mode=full&app_path=%2Fportal"><strong>▶ 打开交互演示</strong></a></p>

<p align="center"><a href="https://github.com/MAC-AutoML/RoTA-Bench"><strong>⌘ 浏览代码</strong></a> · <a href="https://arxiv.org/abs/2610.17665v1"><strong>阅读论文</strong></a> · <a href="https://huggingface.co/papers/2610.17665"><strong>Hugging Face Papers</strong></a></p>

**RoTA-Bench：通过交互式投影对齐评估 3D 空间推理**的代码与固定 674-task 数据集。

公开版本包含 674 个任务。由于版权限制，原始 676 个任务中的两个图标无法再分发，因此移除了对应任务（`single_00345` 和 `single_00384`）。

**论文：**[RoTA-Bench: Evaluating 3D Spatial Reasoning through Interactive Projection Alignment](https://arxiv.org/abs/2610.17665v1) · [Hugging Face Papers](https://huggingface.co/papers/2610.17665)<br>
**作者：**Qingchuan Ma、Yuexiao Ma、Xiang Meng、Yixuan Zou、Tianyu Xie、Wang Chen、Yuhui Zeng、Yuhang Wu、Xiawu Zheng、Rongrong Ji。

## 基准概览

RoTA-Bench 评估智能体通过一系列离散 3D 旋转来对齐目标 2D 投影的能力。每个任务提供目标轮廓、物体的辅助视图和当前投影。智能体从预定义的旋转动作中进行选择，并持续操作，直到投影轮廓与目标对齐。

<p align="center">
  <img src="website/assets/rota-overview.png" alt="RoTA-Bench 基准概览" width="980">
</p>

*基准概览：交互式观测、旋转动作、投影对齐及相应的评估信号。*

<p align="center">
  <img src="website/assets/rota-pipeline.png" alt="RoTA-Bench 构建与评测流程" width="980">
</p>

*构建与评测流程：3D 物体生成、可达参考路径搜索、固定任务划分构建，以及智能体闭环评测。*

网站提供交互式排行榜和任务浏览器。本仓库包含用于可复现对比的评测代码、任务元数据和固定数据集。

## 使用方法

使用 Git LFS 克隆仓库以下载数据集：

```bash
cd RoTA-Bench
pip install -r requirements.txt
```

请从 https://pytorch.org/get-started/locally/ 安装与 NVIDIA 驱动兼容、支持 CUDA 的 PyTorch。正式评测和 Beam 搜索需要 CUDA。

将 `OPENAI_API_KEY` 设置为你的 API 密钥。Bash 中使用 `export OPENAI_API_KEY="your-key"`；PowerShell 中使用 `$env:OPENAI_API_KEY="your-key"`。

使用兼容 OpenAI API 的服务端点运行固定的 100-task 评测：

```bash
python src/step_02_test_models.py --backend gemini_api --config data/run_config.json --formal-config src/final_benchmark_config.json --tasks-file data/tasks_674.jsonl --split-file data/test_100.json --base-url https://YOUR-ENDPOINT/v1 --api-model YOUR-MODEL --model-name YOUR-MODEL --metric-feedback-mode none --max-step-cap 48 --output-dir result/evaluation
```

如需从原始 2D 图标重新生成任务，请安装额外依赖 `cairosvg` 并运行 `src/step_01_generate_data.py`。如果 `data_sources/game-icons` 不存在，脚本会自动下载上游 [game-icons](https://github.com/game-icons/icons) SVG 仓库，将 SVG 栅格化为确定性的 2D 点，应用图标质量筛选，构建球面 3D 几何，渲染目标视图和初始视图，并写出 JSONL 任务记录。game-icons 资源仍受其各自的 Creative Commons/公共领域许可及署名要求约束；重新分发前请查看 `data_sources/game-icons/license.txt` 和上游的署名要求。

### 可复现流程：分步说明

以下命令运行公开的生成流程，不会修改固定文件 `data/tasks_674.jsonl` 或 `data/test_100.json`。生成结果写入 `tasks/`、`data/single/`、`reports/` 和 `logs/`。

1. 安装运行依赖。若要使用 CUDA 渲染，请另行安装与本机驱动兼容的 CUDA 版 PyTorch。

   ```bash
   pip install -r requirements.txt
   ```

2. 获取源图标。若 `data_sources/game-icons` 目录不存在，Step 1 会自动克隆 `game-icons/icons`。也可以手动克隆：

   ```bash
   git clone --depth 1 https://github.com/game-icons/icons data_sources/game-icons
   ```

   上游 SVG 文件保留各自原有的许可证和署名要求；这些文件不包含在本仓库中。

3. 将 SVG 图标转换为确定性的 2D 点并生成任务。此步骤还会构建球面 3D 几何，渲染目标、当前和斜视图，并写出 `tasks/tasks.jsonl`。

   进行单任务 smoke test 时，请先复制运行配置并限制图标池。默认值 `icon_pool_limit: -1` 会扫描上游图标仓库中的全部图标，适用于完整重新生成；即使设置 `--sample-count 1`，也可能需要较大的磁盘空间：

   ```bash
   python -c "import json; from pathlib import Path; p=Path('data/run_config.json'); q=Path('data/run_config_smoke.json'); cfg=json.loads(p.read_text(encoding='utf-8')); cfg.update({'icon_pool_limit':1,'sample_count':1}); q.write_text(json.dumps(cfg, indent=2)+'\n', encoding='utf-8')"
   ```

   ```bash
   python src/step_01_generate_data.py --root . --config data/run_config_smoke.json --sample-count 1 --save-init-images --save-init-oblique-images --no-resume
   ```

   默认会保存目标图像（只有在不需要目标图时才使用 `--no-save-target-images`）。生成任务包含 `icon_points`（采样后的 2D 形状）、姿态元数据，以及用于构建 3D 物体的参数。生成图像保存在 `data/single/target/`、`data/single/init/` 和 `data/single/init_oblique/`。大规模生成数据时请使用单独的配置或输出工作区；不要覆盖固定发布文件。

4. 检查生成的任务，确认样例存在：

   ```bash
   python -c "import json; from pathlib import Path; rows=[json.loads(x) for x in Path('tasks/tasks.jsonl').open(encoding='utf-8') if x.strip()]; print(len(rows), rows[0]['sample_id'], len(rows[0]['icon_points']))"
   ```

5. 为生成的任务创建 split 文件。单任务 smoke test 的 split 必须显式列出生成的 `sample_id`：

   ```bash
   python -c "import json; from pathlib import Path; sid=next(json.loads(x)['sample_id'] for x in Path('tasks/tasks.jsonl').open(encoding='utf-8') if x.strip()); Path('tasks/smoke_split.json').write_text(json.dumps({'split_name':'smoke','count':1,'sample_ids':[sid]},indent=2)+'\\n',encoding='utf-8')"
   ```

6. 对生成的任务运行本地模拟评测。这会在不需要 API 密钥的情况下验证从生成任务、渲染器到闭环评测器的流程：

   ```bash
   python src/step_02_test_models.py --root . --config data/run_config.json --tasks-file tasks/tasks.jsonl --split-file tasks/smoke_split.json --backend simulated --max-tasks 1 --max-step-cap 48 --task-workers 1 --render-workers 1 --output-dir result/smoke_generated --no-resume
   ```

   报告写入 `result/smoke_generated/reports/model_raw.csv`；逐步帧和轨迹保存在 `result/smoke_generated/logs/` 下。

7. 如需使用 API 评测，保持生成任务和 split 路径不变，设置 `OPENAI_API_KEY`，并将 `--backend simulated` 替换为 `--backend gemini_api`，同时提供 `--base-url`、`--api-model` 和要求的正式评测协议选项。正式对比建议使用下方的固定基准命令。

运行 Beam-256 参考搜索：

```bash
python src/guided_iou_beam.py --tasks-file data/tasks_674.jsonl --beam-widths 256 --output-dir result/beam256
```

其他选项请使用 `--help` 查看。100-task 测试 ID、其余 574-task ID，以及保存的 Beam/BFS 路径均位于 `data/` 中。

## 任务示例

`single_00627`：目标投影、当前辅助视图和当前投影。

![投影对齐示例](assets/single_00627_panel.png)

左图显示目标轮廓，中间显示当前 3D 物体，右图显示其当前投影。成功的轨迹会在尽可能减少旋转步数的同时，使当前投影与目标对齐。

## 交互结果与资源

配套网站提供可交互的排行榜和成本–性能图表。将鼠标悬停在数据点上，可以查看模型、基准划分和指标值；使用任务浏览器可以回放单个轨迹。

## 在线演示

[打开交互演示](https://www.modelscope.cn/studios/san2333333/rotabench-anon?mode=full&app_path=%2Fportal) — 由 ModelScope 免费 CPU 创空间提供服务。
