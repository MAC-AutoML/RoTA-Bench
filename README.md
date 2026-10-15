# RoTA-Bench

<p align="center">
  <a href="https://www.modelscope.cn/studios/san2333333/rotabench-anon?mode=full&app_path=%2Fportal"><img src="website/assets/demo-home.png" alt="RoTA-Bench interactive demo" width="920"></a>
</p>

<p align="center"><a href="https://www.modelscope.cn/studios/san2333333/rotabench-anon?mode=full&app_path=%2Fportal"><strong>▶ Open the interactive demo</strong></a></p>

<p align="center"><a href="https://github.com/MAC-AutoML/RoTA-Bench"><strong>⌘ Browse the code</strong></a> · <a href="https://arxiv.org/abs/2610.17665v1"><strong>Read the paper</strong></a></p>

Code and the fixed 674-task dataset accompanying **RoTA-Bench: Evaluating 3D Spatial Reasoning through Interactive Projection Alignment**.

**Paper:** [RoTA-Bench: Evaluating 3D Spatial Reasoning through Interactive Projection Alignment](https://arxiv.org/abs/2610.17665v1)<br>
**Authors:** Qingchuan Ma, Yuexiao Ma, Xiang Meng, Yixuan Zou, Tianyu Xie, Wang Chen, Yuhui Zeng, Yuhang Wu, Xiawu Zheng, Rongrong Ji.

## Usage

Clone with Git LFS to download the dataset:

```bash
cd RoTA-Bench
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build compatible with your NVIDIA driver from https://pytorch.org/get-started/locally/. Formal evaluation and Beam search require CUDA.

Set `PROJECTION_API_KEY` to your own API key. In Bash use `export PROJECTION_API_KEY="your-key"`; in PowerShell use `$env:PROJECTION_API_KEY="your-key"`.

Run the frozen 100-task evaluation using an OpenAI-compatible endpoint:

```bash
python src/step_02_test_models.py --backend gemini_api --config data/run_config.json --formal-config src/final_benchmark_config.json --tasks-file data/tasks_674.jsonl --split-file data/test_100.json --base-url https://YOUR-ENDPOINT/v1 --api-model YOUR-MODEL --model-name YOUR-MODEL --metric-feedback-mode none --max-step-cap 48 --output-dir result/evaluation
```

Run Beam-256 reference search:

```bash
python src/guided_iou_beam.py --tasks-file data/tasks_674.jsonl --beam-widths 256 --output-dir result/beam256
```

Use `--help` for additional options. The 100-task test IDs, remaining 574-task IDs, and saved Beam/BFS paths are in `data/`.

## Example

`single_00627`: target projection, current auxiliary view, and current projection.

![Projection-alignment example](assets/single_00627_panel.png)

## Online demo

[Open the interactive demo](https://www.modelscope.cn/studios/san2333333/rotabench-anon?mode=full&app_path=%2Fportal) — powered by a free CPU Studio on ModelScope.
