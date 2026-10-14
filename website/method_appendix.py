"""HTML tables from the method-referenced paper appendix."""


def render(bi) -> str:
    strata = [("<4", 32, 0), ("4", 53, 8), ("5", 122, 19), ("6", 181, 28),
              ("7", 124, 19), ("8", 69, 11), ("9", 38, 6), ("10", 21, 3),
              ("11", 9, 1), ("12", 10, 2), ("13", 5, 1), ("14", 5, 1),
              ("15", 1, 0), ("16", 4, 1)]
    rows = "".join(f'<tr><td>{length.replace("<", "&lt;")}</td><td>{all_count}</td><td>{test_count}</td></tr>' for length, all_count, test_count in strata)
    return f'''<section class="section method-appendix"><h2>{bi("方法附录 · 观察、参考搜索与测试集", "Method appendix · Observations, search references and test selection")}</h2>
<figure class="paper-figure"><h3>{bi("single_00627 · 提示词中的观察面板", "single_00627 · Observation panels in prompt examples")}</h3>
<img loading="lazy" src="/assets/method-prompt-01.png" alt="single_00627 observation 1"><img loading="lazy" src="/assets/method-prompt-02.png" alt="single_00627 observation 2"><img loading="lazy" src="/assets/method-prompt-03.png" alt="single_00627 observation 3">
<figcaption>{bi("论文附录中的前三步观察。每幅面板从左到右依次为目标投影、当前深度增强辅助视图和当前投影。", "The first three observations from the paper appendix. Each panel is ordered left-to-right as TARGET projection | CURRENT oblique view | CURRENT projection.")}</figcaption></figure>
<p>{bi("我们将深度增强的辅助观察称为‘斜视图’，因为它揭示了二值投影中缺失的深度和遮挡线索，其用途类似于三维形状的斜视描绘。这个名称描述的是它的视觉作用，而非相机旋转：该视图与当前投影使用相同的相机方向和物体朝向渲染，额外线索由深度相关着色提供。", "We call the depth-enhanced auxiliary observation the “oblique view” because it exposes depth and occlusion cues that are absent from the binary projection, similar in purpose to an oblique depiction of a 3D shape. The name describes its visual role rather than a camera rotation: the view is rendered with the same camera direction and object orientation as the current projection, while depth-dependent shading provides the additional cues.")}</p>
<figure class="paper-figure"><h3>{bi("人类评测界面", "Human evaluation interface")}</h3><img loading="lazy" src="/assets/method-human-interface.png" alt="Human evaluation interface from the paper appendix"><figcaption>{bi("人类评测界面使用与模型相同的提示词内容、当前观察、历史观察和离散动作空间。人类通过按钮提交动作，按钮映射到模型的动作标记与角度。", "Human evaluation interface with the same prompt content, current observation, historical observations, and discrete action space used by the models. Humans submit actions through buttons that map to the model action tokens and angles.")}</figcaption></figure>
<figure class="paper-figure"><h3>{bi("674 个任务的参考路径长度对比", "Search-reference and inverse-initialization path lengths across 674 tasks")}</h3>
<div class="table-wrap"><table class="paper-table"><thead><tr><th>{bi("对比", "Comparison")}</th><th>{bi("任务数", "Tasks")}</th><th>{bi("占比", "Percentage")}</th></tr></thead><tbody>
<tr><td>{bi("搜索参考路径更短", "Search reference shorter")}</td><td>585</td><td>86.8%</td></tr>
<tr><td>{bi("长度相同", "Equal length")}</td><td>89</td><td>13.2%</td></tr>
<tr><td>{bi("搜索参考路径更长", "Search reference longer")}</td><td>0</td><td>0.0%</td></tr>
</tbody></table></div><figcaption>{bi("逆初始化路径：平均 11.04 步，中位数 11 步，范围 6–16 步。搜索参考路径：平均 6.54 步，中位数 6 步，范围 2–16 步。", "Inverse mean / median / range: 11.04 / 11 / 6–16. Search mean / median / range: 6.54 / 6 / 2–16.")}</figcaption></figure>
<p>{bi("两种搜索都重放候选路径，并使用与评测相同的投影 IoU 阈值进行验证。我们保留更短的已验证候选路径作为搜索参考，但它不一定是最优路径。限制搜索使参考路径构建在计算上可行；内部姿态和参考路径不会提供给被评测的模型、智能体或人类。", "Both searches replay candidate paths and verify them with the same projection IoU threshold used in evaluation. We retain the shorter verified candidate as the Search reference, although it is not necessarily optimal. Restricting the search keeps reference construction computationally feasible, while internal poses and reference paths remain hidden from evaluated models, agents, and humans.")}</p>
<figure class="paper-figure"><h3>{bi("按搜索参考路径长度划分的任务数", "Task counts by Search-reference length")}</h3>
<div class="table-wrap"><table class="paper-table"><thead><tr><th>{bi("参考旋转步数", "Reference rotations")}</th><th>{bi("全部任务", "All tasks")}</th><th>{bi("测试任务", "Test tasks")}</th></tr></thead><tbody>{rows}
<tr><td>{bi("符合抽样条件（≥4）", "Eligible (≥4)")}</td><td>642</td><td>100</td></tr><tr><td>{bi("总计", "Total")}</td><td>674</td><td>100</td></tr></tbody></table></div></figure>
<p>{bi("标准任务集合包含 674 个具有不同源图标标识符的任务。其中，参考路径至少包含四次旋转的 642 个任务符合测试集抽样条件。我们使用 Hamilton 最大余数法，按照参考路径长度分组的比例分配 100 个测试名额：先分配比例配额的整数部分，再按小数余数从大到小分配剩余名额。每组内的任务按抽样种子与任务标识符的 SHA-256 哈希确定性排序。其余 574 个任务保留供后续训练与研究。", "The canonical collection contains 674 tasks with distinct source-icon identifiers. The 642 tasks whose reference paths contain at least four rotations are eligible for test selection. We allocate 100 test places proportionally across reference-length groups using the Hamilton largest-remainder method: first assign the integer part of each proportional quota, then assign remaining places in descending order of fractional remainder. Within each group, tasks are ranked deterministically by a SHA-256 hash of the selection seed and task identifier. The remaining 574 tasks are reserved for future training and research.")}</p></section>'''
