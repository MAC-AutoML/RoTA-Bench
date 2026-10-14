/* The data is generated from docs/iclr/main.tex by build_paper_data.py. */
(async function () {
  const response = await fetch('/assets/paper-data.json');
  if (!response.ok) return;
  const data = await response.json();
  const svg = document.getElementById('paper-line-chart');
  const legend = document.getElementById('paper-legend');
  const focus = document.getElementById('series-focus');
  const tooltip = document.getElementById('paper-tooltip');
  const familyColor = name => {
    if (name === 'Human') return '#E5EDF4';
    if (name.startsWith('GPT-') || ['ReAct', 'Magentic'].includes(name)) return '#55D6E8';
    if (name.startsWith('Claude ')) return '#BCA9DE';
    if (name.startsWith('Gemini ')) return '#78D6A3';
    if (name.startsWith('Qwen ')) return '#FF4FD8';
    if (name.startsWith('SpatioLM')) return '#FF4FD8';
    return '#A6B8D5';
  };
  const ns = 'http://www.w3.org/2000/svg';
  const node = (tag, attributes, parent = svg) => {
    const element = document.createElementNS(ns, tag);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
    parent.appendChild(element);
    return element;
  };
  const plot = {left: 76, top: 26, width: 1080, height: 360};
  const px = step => plot.left + step * plot.width / 48;
  const py = iou => plot.top + (0.95 - iou) * plot.height / 0.5;
  for (const value of [0.45, 0.55, 0.65, 0.75, 0.85, 0.95]) {
    const y = py(value);
    node('line', {x1: plot.left, y1: y, x2: plot.left + plot.width, y2: y, class: 'gridline'});
    node('text', {x: plot.left - 12, y: y + 4, 'text-anchor': 'end'}).textContent = value.toFixed(2);
  }
  for (const step of [0, 8, 16, 24, 32, 40, 48]) {
    const x = px(step);
    node('line', {x1: x, y1: plot.top, x2: x, y2: plot.top + plot.height, class: 'gridline'});
    node('text', {x, y: 408, 'text-anchor': 'middle'}).textContent = String(step);
  }
  node('line', {x1: plot.left, y1: py(0.9), x2: plot.left + plot.width, y2: py(0.9), stroke: '#b91c1c', 'stroke-width': 2, 'stroke-dasharray': '7 5'});
  const thresholdLabel = node('text', {x: plot.left + plot.width - 5, y: py(0.9) - 7, 'text-anchor': 'end', fill: '#b91c1c'});
  const axisLabel = node('text', {x: plot.left + plot.width / 2, y: 432, 'text-anchor': 'middle'});
  function localizeChart() {
    const english = document.body.classList.contains('language-en');
    thresholdLabel.textContent = english ? 'Success 0.90' : '成功阈值 0.90';
    axisLabel.textContent = english ? 'Rotation steps' : '旋转步数';
    focus.options[0].textContent = english ? 'All methods' : '所有方法';
    svg.setAttribute('aria-label', english ? 'Mean IoU versus rotation steps' : '论文原始数据：平均 IoU 与旋转步数');
  }
  localizeChart();
  document.getElementById('language-toggle').addEventListener('click', () => requestAnimationFrame(localizeChart));
  const groups = [];
  const selected = new Set(data.series.map((_, index) => index));
  function updateVisibility() {
    const chosen = focus.value;
    groups.forEach((group, index) => {
      group.style.display = selected.has(index) ? '' : 'none';
      group.style.opacity = chosen === 'all' || chosen === String(index) ? '1' : '0.15';
    });
  }
  data.series.forEach((series, index) => {
    const color = familyColor(series.name);
    const group = node('g', {'data-series': index});
    groups.push(group);
    const d = series.values.map((value, step) => (step ? 'L' : 'M') + px(step).toFixed(2) + ' ' + py(value).toFixed(2)).join(' ');
    const path = node('path', {d, class: 'series', stroke: color}, group);
    if (series.style.includes('dotted')) path.setAttribute('stroke-dasharray', '2 4');
    else if (series.style.includes('dash')) path.setAttribute('stroke-dasharray', '8 5');
    series.values.forEach((value, step) => {
      const point = node('circle', {cx: px(step), cy: py(value), r: 3, fill: color, tabindex: 0}, group);
      const show = event => {
        const english = document.body.classList.contains('language-en');
        tooltip.textContent = series.name + (english ? ' · step ' : ' · 步数 ') + step + (english ? ' · mean IoU ' : ' · 平均 IoU ') + value.toFixed(6);
        tooltip.style.display = 'block';
        tooltip.style.left = Math.min((event.clientX || point.getBoundingClientRect().left) + 12, innerWidth - 250) + 'px';
        tooltip.style.top = Math.max((event.clientY || point.getBoundingClientRect().top) - 42, 8) + 'px';
      };
      point.addEventListener('pointerenter', show);
      point.addEventListener('pointermove', show);
      point.addEventListener('pointerleave', () => tooltip.style.display = 'none');
      point.addEventListener('focus', show);
      point.addEventListener('blur', () => tooltip.style.display = 'none');
    });
    const label = document.createElement('label');
    label.innerHTML = '<input type="checkbox" checked><i></i><span></span>';
    label.querySelector('i').style.background = color;
    const labelText = label.querySelector('span');
    labelText.dataset.en = series.name;
    labelText.dataset.zh = ({'Human':'人类', 'Uniform random':'均匀随机', 'Random + IoU rollback':'随机 + IoU 回退', 'IoU oracle':'IoU oracle', 'Pose oracle':'姿态 oracle'}[series.name] || series.name);
    labelText.textContent = document.body.classList.contains('language-en') ? labelText.dataset.en : labelText.dataset.zh;
    label.querySelector('input').addEventListener('change', event => {
      event.target.checked ? selected.add(index) : selected.delete(index);
      updateVisibility();
    });
    legend.appendChild(label);
    focus.add(new Option(series.name, index));
  });
  focus.addEventListener('change', updateVisibility);
  updateVisibility();

  const tbody = document.querySelector('#paper-results tbody');
  if (!tbody) {
    document.getElementById('language-toggle').addEventListener('click', () => requestAnimationFrame(() => {
      const english = document.body.classList.contains('language-en');
      document.querySelectorAll('#paper-legend label span').forEach(span => {
        span.textContent = english ? span.dataset.en : span.dataset.zh;
      });
    }));
    return;
  }
  const original = data.table.slice();
  function renderRows(rows) {
    tbody.replaceChildren(...rows.map(row => {
      const tr = document.createElement('tr');
      [row.method, ...row.values.map((v, i) => v === null ? '—' : i === 1 ? v.toFixed(3) : v.toFixed(2).replace(/\.00$/, '.0'))].forEach(value => {
        const td = document.createElement('td'); td.textContent = value; tr.appendChild(td);
      });
      return tr;
    }));
  }
  const metricNames = ['Method', 'Succ. (%)', 'Final IoU', 'Steps', 'Cov. (%)', 'Eff. dir. (%)', 'Best ang. (%)', 'Gap (°)', 'Loss (°)'];
  const metricNamesZh = ['方法', '成功率 (%)', '最终 IoU', '步数', '覆盖率 (%)', '有效方向 (%)', '最佳角度 (%)', '角度差 (°)', '角度损失 (°)'];
  function localizePaper() {
    const english = document.body.classList.contains('language-en');
    document.querySelectorAll('.paper-figure img[data-alt-en]').forEach(image => {
      image.alt = english ? image.dataset.altEn : image.dataset.altZh;
    });
    document.querySelectorAll('#paper-results th button').forEach((button, index) => {
      button.textContent = (english ? metricNames[index] : metricNamesZh[index]) + ' ↕';
    });
    document.querySelectorAll('#paper-legend label span').forEach(span => {
      span.textContent = english ? span.dataset.en : span.dataset.zh;
    });
    [...focus.options].slice(1).forEach((option, index) => {
      const series = data.series[index];
      option.textContent = english ? series.name : ({'Human':'人类', 'Uniform random':'均匀随机', 'Random + IoU rollback':'随机 + IoU 回退', 'IoU oracle':'IoU oracle', 'Pose oracle':'姿态 oracle'}[series.name] || series.name);
    });
  }
  renderRows(original);
  localizePaper();
  document.getElementById('language-toggle').addEventListener('click', () => requestAnimationFrame(localizePaper));
  document.querySelectorAll('#paper-results th button').forEach(button => {
    let descending = true;
    button.addEventListener('click', () => {
      const index = Number(button.dataset.sort);
      const rows = original.slice().sort((a, b) => index === 0 ? a.method.localeCompare(b.method) : (a.values[index - 1] ?? -Infinity) - (b.values[index - 1] ?? -Infinity));
      if (descending) rows.reverse();
      descending = !descending;
      renderRows(rows);
    });
  });
})();
