# RoTA-Bench（简体中文）

<p align="center"><a href="README.md">English</a> · <a href="README_zh.md">简体中文</a></p>

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

![RoTA-Bench 基准概览](website/assets/rota-overview.png)

![RoTA-Bench 构建与评测流程](website/assets/rota-pipeline.png)

网站提供交互式排行榜和任务浏览器。本仓库包含用于可复现对比的评测代码、任务元数据和固定数据集。

## 使用方法

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

## 从 2D 图标生成任务

若 `data_sources/game-icons` 不存在，Step 1 会自动克隆上游图标仓库，将 SVG 转换为确定性的 2D 点，构建球面 3D 几何并渲染任务视图。上游图标不包含在本仓库中，请遵守其许可证和署名要求。

单任务 smoke test 建议复制配置并将 `icon_pool_limit` 设为 1；默认值 `-1` 会扫描全部图标，可能需要较大的磁盘空间：

```bash
python -c "import json; from pathlib import Path; p=Path('data/run_config.json'); q=Path('data/run_config_smoke.json'); cfg=json.loads(p.read_text(encoding='utf-8')); cfg.update({'icon_pool_limit':1,'sample_count':1}); q.write_text(json.dumps(cfg, indent=2)+'\n', encoding='utf-8')"
python src/step_01_generate_data.py --root . --config data/run_config_smoke.json --sample-count 1 --save-init-images --save-init-oblique-images --no-resume
```

检查任务并创建单任务 split：

```bash
python -c "import json; from pathlib import Path; rows=[json.loads(x) for x in Path('tasks/tasks.jsonl').open(encoding='utf-8') if x.strip()]; print(len(rows), rows[0]['sample_id'], len(rows[0]['icon_points']))"
python -c "import json; from pathlib import Path; sid=next(json.loads(x)['sample_id'] for x in Path('tasks/tasks.jsonl').open(encoding='utf-8') if x.strip()); Path('tasks/smoke_split.json').write_text(json.dumps({'split_name':'smoke','count':1,'sample_ids':[sid]},indent=2)+'\n',encoding='utf-8')"
```

如需运行本地模拟评测：

```bash
python src/step_02_test_models.py --root . --config data/run_config.json --tasks-file tasks/tasks.jsonl --split-file tasks/smoke_split.json --backend simulated --max-tasks 1 --max-step-cap 48 --task-workers 1 --render-workers 1 --output-dir result/smoke_generated --no-resume
```

## 任务示例

`single_00627`：目标投影、当前辅助视图和当前投影。

![投影对齐示例](assets/single_00627_panel.png)

左图显示目标轮廓，中间显示当前 3D 物体，右图显示当前投影。成功的轨迹会在尽可能减少旋转步数的同时，使当前投影与目标对齐。

## 在线演示

[打开交互演示](https://www.modelscope.cn/studios/san2333333/rotabench-anon?mode=full&app_path=%2Fportal) — 由 ModelScope 免费 CPU 创空间提供服务。
