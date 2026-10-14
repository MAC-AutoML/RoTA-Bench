"""Render verbatim English excerpts from the project's paper TeX source."""

import html
import json
import re
from pathlib import Path
from functools import lru_cache
from latex2mathml.converter import convert as latex_to_mathml


EXCERPTS = json.loads((Path(__file__).parent / "assets/paper-source-excerpts.json").read_text(encoding="utf-8"))

ZH_EXCERPTS = {
    "abstract": """视觉语言模型能否将二维观察转化为支持精确操作的三维理解？这要求模型推断物体几何、预判旋转效果，并通过视觉反馈细化动作。我们提出 RoTA-Bench，通过交互式三维到二维投影对齐评估这些能力。给定目标轮廓，模型观察旋转后的物体，并在有限动作预算内选择连续旋转以恢复目标投影。我们的自动化流程从二维图标构造保留轮廓的三维物体，并通过可逆初始化保证任务可解，共生成 674 个任务及经分层抽样得到的 100 个测试任务。实验发现：（i）人类与模型之间仍存在显著差距：GPT-5.6-Sol 仅成功完成 17% 的任务，而 20 名本科生参与者合计完成了四轮完整测试集，成功率为 100%；（ii）图像空间与姿态空间的进展可能分歧：在匹配状态下，模型比人类更常出现轮廓重叠改善但姿态对齐变差的情况（19.4% 对 10.9%）；（iii）精细控制对交互历史敏感：移除交互历史后，模型 96.8% 的动作采用粗粒度的 30° 步长，观察到的任务成功率也大幅降低。这些发现突出了将视觉证据转化为精确三维控制的挑战，而不仅仅是识别或匹配单张图像。""",
    "introduction": """心理学与认知科学研究表明，空间感知是认知能力的基础领域，对于跨视角保持物体恒常性，以及通过对形状和方向的心理表征、操作与推断，从二维投影中恢复三维几何，均至关重要。随着视觉语言模型日益应用于视觉问答、交互式智能体和具身系统，具身探索、几何感知导航等交互场景中的智能体必须将空间理解转化为物理动作。然而，当前模型在推理三维位置、方向和空间关系方面仍面临挑战。因此，我们需要评测工具来衡量模型能否将二维观察转化为支持精确操作的三维理解。

然而，当前视觉与空间基准尚不能满足这一要求。静态基准通过对固定图像进行被动问答来评估空间推理，完全缺少动作执行和闭环反馈。相反，涵盖软件界面、视觉迷宫和滑块谜题的现有交互式基准，主要关注高层任务规划、工具调用或复杂环境中的空间寻路。在这些场景中，任务成功主要取决于编码能力、推理和规划，而非从视觉线索感知三维方向并将其转化为动作。

为弥补这一差距，受从二维投影和轮廓推断三维形状这一经典观点启发，我们提出如下问题：视觉语言模型能否将二维观察转化为支持精确操作的三维理解？为此，我们提出 RoTA-Bench，这是一个自动构建的交互式三维到二维投影对齐基准。我们的流程从二维图标构造三维物体，通过可逆旋转生成初始方向，并自动保证任务可解。给定目标轮廓以及旋转物体的视觉观察，模型在每一步从 24 种离散动作中选择一种，在下一步接收更新后的观察，并在有限动作预算内迭代改善对齐。该基准包含 674 个任务，其中测试集通过分层抽样选出 100 个任务。""",
    "motivation": """认知视觉与心理学的经典研究表明，人们能够从二维视觉轮廓推断其背后的三维几何与方向，并通过心理旋转进行比较。受这些基础研究启发，我们将三维方向对齐表述为投影匹配任务：旋转三维物体，直至其二维轮廓与目标图像相匹配。将物理交互限制为三维旋转带来两项关键的方法优势。第一，面外旋转会引起非线性的轮廓形变、宽高比变化和自遮挡，形成难以用文本表征、且本质上依赖视觉的挑战。第二，纯旋转将空间方向推理与平移漂移、抓取可供性和接触摩擦等额外物理因素清晰分离，使我们能够考察模型能否将二维观察转化为支持精确操作的三维理解。

为减少对语义捷径的依赖并鼓励视觉空间推理，RoTA-Bench 从二维矢量图标合成三维物体，而非直接采用常见的三维 CAD 模型库中的物体，以避免熟悉形状触发预训练先验。我们在球形体积内挤出二维图标，自动生成抽象且新颖的三维几何体，同时从数学上保证标准正视图能够复现原始图标，保留其精细轮廓。最后，通过可逆离散旋转序列生成初始姿态，从构造上保证每个任务都可解。""",
    "task": """沿用上述设计动机，我们将交互式投影匹配任务定义如下。每个任务实例由一个以坐标原点为中心的刚性三维物体、一个初始姿态和一个标准目标姿态构成。正交投影算子会根据物体姿态生成二值轮廓掩码。在第 t 步，环境提供由目标轮廓、当前物体的深度着色辅助视图和当前投影组成的三元观察。目标轮廓保持不变，当前投影表示物体在当前姿态下的轮廓。为补充二值轮廓缺失的深度与遮挡线索，智能体还会收到一张深度着色辅助视图；该视图与当前投影从相同视角渲染。

基于这些视觉反馈，智能体通过离散动作空间与物体交互。在每一步，智能体从由 24 种相机坐标系固定的旋转动作及终止指令 STOP 构成的动作集合中选择动作。这 24 种动作覆盖三个主要相机轴上的六个有向旋转：俯仰（向上/向下）、偏航（向左/向右）以及平面内滚转（顺时针/逆时针）；每个方向配有 1°、5°、10° 和 30° 四种步长，以支持粗略和精细的姿态调整。执行动作会更新三维姿态，并重新渲染下一步的当前投影和辅助视图。

为评估投影对齐，我们计算当前投影与目标轮廓之间的交并比（IoU）。当 IoU 不低于 0.90 时，回合即判为成功。该阈值在确保较高视觉对齐精度的同时，为离散动作空间提供必要的容差，因为有限步数内难以实现精确对齐。此外，几何对称性与正交投影下的自重叠可能使不同三维姿态产生相同轮廓，因此任务自然允许存在多个等价目标姿态。回合会在任务成功、主动执行 STOP、达到最大动作预算，或连续若干步未能改善最佳 IoU 后停滞时结束。""",
    "construction": """如图所示，我们的自动化流程分为三个连续阶段：任务生成、参考搜索和数据集构建。

首先，在任务生成阶段，我们从开源 Game-icons 图标库中筛选二维矢量图标，并根据连通性、占用率、长宽比和前景点数等几何条件筛选后，将图标栅格化为高分辨率二值掩码。为从每个二维图标构造三维物体，同时严格限制其在任意旋转下的空间范围，我们将平面点集缩放至半径为 r 的圆盘内，并沿深度轴在球形包络内对称挤出。所得实体被离散化为稠密三维点云。由于三维旋转保持欧几里得范数不变，任意旋转后的姿态仍位于同一球形范围内；这一球面对称性可避免视口裁切，并使所有任务共享固定的相机比例。为提供单目深度线索并弥补采样点之间的离散空隙，我们使用带有深度相关着色和高斯平滑的正交投影来渲染视觉观察。

为保留原始二维图标的精细轮廓，我们将标准目标姿态固定为单位姿态，并通过离散旋转序列生成初始姿态。序列长度在 6 至 16 步之间均匀采样，每一步从六个候选相机方向中均匀选择一个 30° 旋转，并禁止紧接着执行其逆操作，以避免旋转抵消并确保产生实际位移。初始投影已经达到成功阈值的姿态会被丢弃。关键在于，SO(3) 中的每个旋转均可逆，因此按相反顺序执行生成序列中各旋转的逆操作即可恢复目标姿态，从构造上确保所有任务均可达且可解。该流程共生成 674 个具有不同源图标的标准任务。

其次，在参考搜索阶段，我们使用姿态引导的束搜索和仅限 30° 操作的广度优先搜索计算参考路径；与直接反转生成序列相比，这些搜索能够找到显著更短的轨迹。

最后，在数据集构建阶段，我们按照参考路径长度进行分层，并使用 Hamilton 方法进行比例分层抽样，再通过确定性哈希排序，冻结由 100 个任务组成的测试集；其余 574 个任务留作后续训练与研究。""",
}


@lru_cache(maxsize=None)
def render_excerpt(section: str, language: str = "en") -> str:
    text = ZH_EXCERPTS[section] if language == "zh" else EXCERPTS[section]
    if language != "zh":
        return _render_english_excerpt(text)
    text = re.sub(r"\\label\{[^}]+\}", "", text)
    text = re.sub(r"\\citep\{[^}]+\}", "", text)
    text = re.sub(r"(?:Appendix Table|Table|Figure|Appendix|Section)~\\ref\{([^}]+)\}", lambda m: m.group(0).split("~")[0], text)
    text = text.replace(r"\%", "%").replace(r"\_", "_").replace(r"\&", "&")
    text = re.sub(r"\\texttt\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\textbf\{([^{}]+)\}", r"\1", text)
    text = text.replace("$^\\circ$", "°").replace("^\\circ", "°")
    text = re.sub(r"\\([,.])", r"\1", text)
    text = text.replace("~", " ").replace("---", "—").replace("--", "–")
    text = text.replace("$", "")
    text = re.sub(r"\\(?:mathrm|operatorname|text)\{([^{}]+)\}", r"\1", text)
    paragraphs = [html.escape(p.strip()).replace("\n", " ") for p in re.split(r"\r?\n\s*\r?\n", text) if p.strip()]
    return "".join(f"<p>{paragraph}</p>" for paragraph in paragraphs)


def _render_english_excerpt(text: str) -> str:
    math_blocks: list[str] = []

    def hold_equation(match: re.Match) -> str:
        body = re.sub(r"\\label\{[^}]+\}", "", match.group(1)).strip()
        math_blocks.append(_equation_mathml(body))
        return f" QZBLOCK{len(math_blocks) - 1}QZ "

    text = re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}", hold_equation, text, flags=re.S)

    def hold_inline(match: re.Match) -> str:
        math_blocks.append(_equation_mathml(match.group(1)))
        return f" QZBLOCK{len(math_blocks) - 1}QZ "

    text = re.sub(r"\$(.+?)\$", hold_inline, text)
    text = re.sub(r"\\label\{[^}]+\}", "", text)
    text = re.sub(r"\\citep\{[^}]+\}", "", text)
    text = re.sub(r"(?:Appendix Table|Table|Figure|Appendix|Section)~\\ref\{([^}]+)\}", lambda m: m.group(0).split("~")[0], text)
    text = text.replace(r"\%", "%").replace(r"\_", "_").replace(r"\&", "&")
    text = re.sub(r"\\texttt\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\textbf\{([^{}]+)\}", r"\1", text)
    text = text.replace("$^\\circ$", "°").replace("^\\circ", "°")
    text = re.sub(r"\\([,.])", r"\1", text)
    text = text.replace("~", " ").replace("---", "—").replace("--", "–")
    for latex, symbol in ((r"\subset", "⊂"), (r"\in", "∈"), (r"\times", "×"), (r"\quad", "　"), (r"\ge", "≥"), (r"\mid", "|"), (r"\mathbb{R}", "ℝ"), (r"\mathcal{A}", "𝒜"), (r"\mathcal{R}", "ℛ"), (r"\mathcal{G}", "𝒢"), (r"\mathrm{SO}(3)", "SO(3)"), (r"\operatorname{IoU}", "IoU"), (r"\Pi", "Π"), (r"\star", "★"), (r"\Delta", "Δ"), (r"\theta", "θ"), (r"\tau", "τ"), (r"\cap", "∩"), (r"\cup", "∪"), (r"\{", "{"), (r"\}", "}"), (r"\,", " ")):
        text = text.replace(latex, symbol)
    text = re.sub(r"\\(?:mathrm|operatorname|text|mathcal|mathbb)\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = text.replace("^\\star", "★").replace("^\\top", "ᵀ").replace("_t", "ₜ").replace("_0", "₀").replace("_{t+1}", "ₜ₊₁")
    text = re.sub(r"\\", "", text)
    paragraphs = [p.strip().replace("\n", " ") for p in re.split(r"\r?\n\s*\r?\n", text) if p.strip()]
    rendered = []
    for paragraph in paragraphs:
        safe = html.escape(paragraph)
        for index, equation in enumerate(math_blocks):
            safe = safe.replace(f"QZBLOCK{index}QZ", equation)
        rendered.append(f"<p>{safe}</p>")
    return "".join(rendered)


def _equation_mathml(source: str) -> str:
    """Convert paper math source into structural MathML."""
    source = re.sub(r"\\label\{[^}]+\}", "", source).strip()
    return latex_to_mathml(source).replace('display="inline"', 'display="inline" class="inline-equation"')
