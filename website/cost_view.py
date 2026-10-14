"""Cost–performance view using the paper's full-test API estimates."""
import json
from pathlib import Path


def render(bilingual):
    root = Path(__file__).resolve().parent
    data = json.loads((root / "assets/paper-data.json").read_text(encoding="utf-8"))["api_costs"]
    data.append({"method": "Human", "tasks": 100, "total_usd": None, "success": 100.0, "iou": 0.923})
    return f'''<section class="section cost-section"><h2>{bilingual("成本与性能", "Cost and performance")}</h2>
<p class="section-intro">{bilingual("完整 100-task 测试集。横轴为平均 API 费用（美元／任务，对数刻度）；左上方代表更低成本与更高性能。人类结果作为无 API 费用参照单独标记。", "Full 100-task test set. The horizontal axis shows mean API cost (USD/task, log scale); the upper left indicates lower cost and higher performance. Human results are shown separately as a non-API reference.")}</p>
<div class="chart-card"><div class="chart-controls" id="cost-controls"><button type="button" data-metric="success" aria-pressed="true">{bilingual("成功率", "Success rate")}</button><button type="button" data-metric="iou" aria-pressed="false">{bilingual("最终 IoU", "Final IoU")}</button></div>
<svg id="cost-chart" viewBox="0 0 960 480" role="img"></svg><div id="cost-detail" aria-live="polite"></div><div id="cost-legend" class="paper-legend"></div></div>
<p class="table-caption">{bilingual("费用取自论文 API 成本表：记录的输入／输出 token × 公开未缓存定价，智能体包含内部工作流。人类成功率和 IoU 依论文报告展示；无 API 费用不等于实际成本为零。", "Costs come from the paper’s API cost table: recorded input/output tokens × published uncached list prices, including internal agent workflows. Human success and IoU follow the paper; absence of API cost does not mean zero actual cost.")}</p>
</section><script>const costData={json.dumps(data)};</script><script>{(root / 'assets/cost-chart.js').read_text(encoding='utf-8')}</script>'''
