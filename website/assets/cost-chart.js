(() => {
  const svg = document.getElementById('cost-chart');
  const detail = document.getElementById('cost-detail');
  const ns = 'http://www.w3.org/2000/svg';
  let metric = 'success', selected = null;
  const familyColor = method => {
    if (method === 'Human') return '#E5EDF4';
    if (/^(GPT-|GPT-5\.6-Sol \+)/.test(method)) return '#55D6E8';
    if (method.startsWith('Claude ')) return '#BCA9DE';
    if (method.startsWith('Gemini ')) return '#78D6A3';
    if (method.startsWith('Qwen ')) return '#FF4FD8';
    return '#A6B8D5';
  };
  const en = () => document.body.classList.contains('language-en');
  const t = (zh, english) => en() ? english : zh;
  function add(tag, attrs, content = '') {
    const el = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
    el.textContent = content; svg.appendChild(el); return el;
  }
  function describe(d) {
    detail.textContent = `${d.method} · ${t('成功率','Success')} ${d.success.toFixed(1)}% · Final IoU ${d.iou.toFixed(3)} · $${(d.total_usd / d.tasks).toFixed(4)} / ${t('任务','task')} · ${t('总费用','Total')} $${d.total_usd.toFixed(4)} (${d.tasks} ${t('任务','tasks')})`;
  }
  function draw() {
    svg.replaceChildren();
    svg.setAttribute('aria-label', t('API 成本与模型性能散点图', 'API cost versus model performance scatterplot'));
    const x = cost => 185 + (Math.log10(cost) - Math.log10(.04)) / (Math.log10(5) - Math.log10(.04)) * 710;
    const y = value => metric === 'success'
      ? 408 - value / 100 * 320
      : 408 - (value - .45) / .5 * 320;
    add('rect',{x:98,y:78,width:52,height:330,class:'human-band'});
    add('text',{x:124,y:66,'text-anchor':'middle',class:'human-band-label'},'Human');
    add('text',{x:124,y:429,'text-anchor':'middle',class:'human-band-note'},t('无 API 费','No API fee'));
    [.05,.1,.2,.5,1,2,5].forEach(cost => {
      add('line',{x1:x(cost),x2:x(cost),y1:78,y2:408,class:'gridline'});
      add('text',{x:x(cost),y:435,'text-anchor':'middle',class:'chart-tick'},`$${cost}`);
    });
    const ticks = metric === 'success' ? [0,20,40,60,80,100] : [.45,.55,.65,.75,.85,.95];
    ticks.forEach(value => {
      add('line',{x1:150,x2:895,y1:y(value),y2:y(value),class:'gridline'});
      add('text',{x:82,y:y(value)+5,'text-anchor':'end',class:'chart-tick'},metric === 'success' ? `${value}%` : value.toFixed(2));
    });
    add('text',{x:150,y:40,class:'chart-axis-title'},metric === 'success' ? t('成功率 ↑','Success rate ↑') : 'Final IoU ↑');
    add('text',{x:555,y:473,'text-anchor':'middle',class:'chart-axis-title'},t('平均 API 费用 · 美元／任务（对数刻度）','Mean API cost · USD/task (log scale)'));
    costData.forEach((d,i) => {
      const isHuman = d.total_usd === null;
      const color = familyColor(d.method);
      const cx = isHuman ? 124 : x(d.total_usd/d.tasks);
      const point = add('circle',{cx,cy:y(d[metric]),r:isHuman ? 11 : 8,fill:color,class:isHuman ? 'human-point' : 'model-point',tabindex:0,role:'button','aria-label':isHuman ? `${d.method}: ${d.success}% success, IoU ${d.iou}; no API cost` : `${d.method}: $${(d.total_usd/d.tasks).toFixed(4)}/task, ${d.success}%, IoU ${d.iou}`});
      const show = () => {
        selected = d;
        if (isHuman) detail.textContent = `${d.method} · ${t('成功率','Success')} ${d.success.toFixed(1)}% · Final IoU ${d.iou.toFixed(3)} · ${t('人类参考，不适用 API 费用','Human reference; API cost not applicable')}`;
        else describe(d);
      };
      point.addEventListener('mouseenter',show); point.addEventListener('focus',show); point.addEventListener('click',show);
      point.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); show(); } });
      const title = document.createElementNS(ns,'title'); title.textContent=d.method; point.appendChild(title);
    });
    if (selected) {
      if (selected.total_usd === null) detail.textContent = `${selected.method} · ${t('成功率','Success')} ${selected.success.toFixed(1)}% · Final IoU ${selected.iou.toFixed(3)} · ${t('人类参考，不适用 API 费用','Human reference; API cost not applicable')}`;
      else describe(selected);
    } else detail.textContent=t('悬停、点击或用 Tab 选中散点，查看完整指标与费用。','Hover, click or Tab to a point to inspect metrics and costs.');
  }
  costData.forEach((d,i) => {
    const label=document.createElement('span'), swatch=document.createElement('i');
    swatch.style.background=familyColor(d.method); label.append(swatch,document.createTextNode(' '+d.method));
    document.getElementById('cost-legend').appendChild(label);
  });
  document.querySelectorAll('#cost-controls button').forEach(button => button.addEventListener('click',() => {
    metric=button.dataset.metric;
    document.querySelectorAll('#cost-controls button').forEach(b => b.setAttribute('aria-pressed',String(b===button)));
    draw();
  }));
  new MutationObserver(draw).observe(document.body,{attributes:true,attributeFilter:['class']});
  draw();
})();
