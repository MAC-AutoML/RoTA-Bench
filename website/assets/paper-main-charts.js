/* Interactive renderings of the paper's original main-text figures and table. */
(function () {
  const $ = id => document.getElementById(id);
  const ns = 'http://www.w3.org/2000/svg';
  const palette = ['#55D6E8', '#78D6A3', '#FF4FD8', '#BCA9DE'];
  const tooltip = $('paper-tooltip');
  const bilingual = (zh, en) => document.body.classList.contains('language-en') ? en : zh;
  const svgNode = (parent, tag, attrs = {}) => {
    const node = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
    parent.appendChild(node);
    return node;
  };
  function showTip(event, lines) {
    tooltip.replaceChildren(...lines.map(line => {
      const div = document.createElement('div'); div.textContent = line; return div;
    }));
    tooltip.style.display = 'block';
    tooltip.style.left = Math.min((event.clientX || 20) + 12, innerWidth - 330) + 'px';
    tooltip.style.top = Math.max((event.clientY || 50) - 24, 8) + 'px';
  }
  const hideTip = () => { tooltip.style.display = 'none'; };

  const ablation = [
    ['A: Full input', 16.67, .5656, 56.52, 12.18, 7.51, 17.56, 62.75, -.472, -.626, -.272],
    ['B: No aux. view', 3.33, .5490, 55.90, 3.54, 3.24, 7.08, 86.14, -.463, -.605, -.238],
    ['D: Neither input', 6.67, .5087, 55.02, 1.29, 1.29, .81, 96.60, -.295, -.468, -.066],
    ['C: No history', 0, .4822, 54.85, .16, .49, 2.59, 96.76, -.242, -.331, -.131],
    ['Uniform random', 0, .4808, 53.12, 24.24, 25.27, 27.10, 23.39, -.027, -.073, .019]
  ];
  const abHeaders = ['Condition', 'Success (%)', 'Final IoU', 'Effective direction (%)', '1° (%)', '5° (%)', '10° (%)', '30° (%)', 'ρ', '95% CI low', '95% CI high'];
  const abHeadersZh = ['条件', '成功率 (%)', '最终 IoU', '有效方向 (%)', '1° (%)', '5° (%)', '10° (%)', '30° (%)', 'ρ', '95% CI 下界', '95% CI 上界'];
  const abTable = $('ablation-table');
  let abSort = {index: 2, descending: true};
  function renderAblation() {
    const en = document.body.classList.contains('language-en');
    abTable.querySelector('thead').innerHTML = `<tr>${(en ? abHeaders : abHeadersZh).map((h, i) => `<th><button type="button" data-sort="${i}">${h} ↕</button></th>`).join('')}</tr>`;
    const sorted = ablation.slice().sort((a, b) => {
      const x = a[abSort.index], y = b[abSort.index];
      return (typeof x === 'string' ? x.localeCompare(y) : x - y) * (abSort.descending ? -1 : 1);
    });
    abTable.querySelector('tbody').replaceChildren(...sorted.map(row => {
      const tr = document.createElement('tr');
      row.forEach((value, i) => {
        const td = document.createElement('td');
        td.textContent = i === 0 ? value : i === 2 ? value.toFixed(4) : i >= 8 ? value.toFixed(3) : value.toFixed(2);
        td.title = td.textContent; tr.appendChild(td);
      });
      return tr;
    }));
    abTable.querySelectorAll('th button').forEach(button => button.addEventListener('click', () => {
      const index = Number(button.dataset.sort);
      abSort = {index, descending: abSort.index === index ? !abSort.descending : true}; renderAblation();
    }));
  }
  renderAblation();

  const corr = [
    ['Human',.696,.633,.763,.751,.675,.828], ['ReAct',.570,.453,.674,.320,.108,.512],
    ['Gemini 3.6 Flash',.562,.480,.633,.618,.496,.709], ['GPT-5.6-Sol',.536,.443,.617,.368,.166,.532],
    ['GPT-6-Astra',.533,.427,.621,.452,.266,.607], ['GPT-5.6-Terra',.438,.335,.526,.254,.056,.440],
    ['GPT-5.6-Luna',.471,.373,.561,.280,.088,.470], ['Magentic',.368,.215,.493,.250,.066,.403],
    ['Qwen 3.7 Plus',.383,.251,.495,.380,.232,.514], ['Claude Sonnet 5',.255,.140,.365,.154,0,.309],
    ['Claude Opus 5',.091,0,.229,.204,.043,.354], ['SpatioLM',.127,0,.309,.142,0,.268]
  ];
  const corrSvg = $('correlation-chart'), corrLegend = $('correlation-legend');
  const corrActive = [true, true];
  const corrNames = [['|ρ|: IoU–angle', '|ρ|: Pose–angle'], ['|ρ|：IoU–角度', '|ρ|：姿态–角度']];
  corrNames[0].forEach((name, i) => {
    const label = document.createElement('button'); label.type = 'button'; label.className = 'chart-legend-button';
    label.innerHTML = `<i style="background:${palette[i]}"></i>${name}`;
    label.addEventListener('click', () => { corrActive[i] = !corrActive[i]; label.classList.toggle('is-muted', !corrActive[i]); drawCorr(); });
    corrLegend.appendChild(label);
  });
  function drawCorr() {
    corrSvg.replaceChildren();
    const p = {l: 62, t: 25, w: 1000, h: 425}, y = v => p.t + (1 - v) * p.h;
    [0,.2,.4,.6,.8,1].forEach(v => {
      const yy = y(v); svgNode(corrSvg, 'line', {x1:p.l,y1:yy,x2:p.l+p.w,y2:yy,class:'gridline'});
      const tx = svgNode(corrSvg, 'text', {x:p.l-10,y:yy+4,'text-anchor':'end'}); tx.textContent = v.toFixed(1);
    });
    corr.forEach((r, i) => {
      const cx = p.l + (i + .5) * p.w / corr.length;
      [[r[1],r[2],r[3],0], [r[4],r[5],r[6],1]].forEach(([val, low, high, group]) => {
        const w = 22, x = cx + (group ? 12 : -34), color = palette[group];
        const bar = svgNode(corrSvg, 'rect', {x,y:y(val),width:w,height:y(0)-y(val),fill:color,opacity:corrActive[group]?'.88':'.10',tabindex:'0',class:'data-mark'});
        const tip = e => showTip(e, [`${r[0]} · ${corrNames[document.body.classList.contains('language-en')?0:1][group]}`, `|ρ| = ${Number(val).toFixed(3)}`, `95% CI [${Number(low).toFixed(3)}, ${Number(high).toFixed(3)}]`]);
        bar.addEventListener('pointerenter',tip); bar.addEventListener('pointermove',tip); bar.addEventListener('pointerleave',hideTip); bar.addEventListener('focus',tip); bar.addEventListener('blur',hideTip);
        const yy1=y(high), yy2=y(low), mid=x+w/2;
        svgNode(corrSvg,'line',{x1:mid,y1:yy1,x2:mid,y2:yy2,stroke:'#E5EDF4','stroke-width':1.5,opacity:corrActive[group]?1:.1});
        svgNode(corrSvg,'line',{x1:mid-5,y1:yy1,x2:mid+5,y2:yy1,stroke:'#E5EDF4',opacity:corrActive[group]?1:.1});
        svgNode(corrSvg,'line',{x1:mid-5,y1:yy2,x2:mid+5,y2:yy2,stroke:'#E5EDF4',opacity:corrActive[group]?1:.1});
      });
      const label=svgNode(corrSvg,'text',{x:cx+8,y:475,'text-anchor':'end',transform:`rotate(-48 ${cx+8} 475)`,'font-size':'12'}); label.textContent=r[0];
    });
    const axis=svgNode(corrSvg,'text',{x:18,y:230,transform:'rotate(-90 18 230)','text-anchor':'middle'}); axis.textContent='|ρ|';
  }
  drawCorr();

  const states = [
    {key:'UC', zh:'IoU 上升 · 姿态误差下降', en:'IoU ↑ · pose error ↓', color:palette[0], values:[50.24,34.60]},
    {key:'DC', zh:'IoU 下降 · 姿态误差下降', en:'IoU ↓ · pose error ↓', color:palette[1], values:[33.18,23.70]},
    {key:'UF', zh:'IoU 上升 · 姿态误差上升', en:'IoU ↑ · pose error ↑', color:palette[2], values:[10.90,19.43]},
    {key:'DF', zh:'IoU 下降 · 姿态误差上升', en:'IoU ↓ · pose error ↑', color:palette[3], values:[5.69,22.27]}
  ];
  const activeStates = new Set(states.map(s=>s.key)), stateLegend=$('state-legend'), stateSvg=$('state-chart');
  states.forEach(s=>{
    const b=document.createElement('button'); b.type='button'; b.className='chart-legend-button'; b.innerHTML=`<i style="background:${s.color}"></i>${s.key} · ${s.zh}`;
    b.addEventListener('click',()=>{activeStates.has(s.key)?activeStates.delete(s.key):activeStates.add(s.key);b.classList.toggle('is-muted',!activeStates.has(s.key));drawStates();}); stateLegend.appendChild(b);
  });
  function drawStates(){
    stateSvg.replaceChildren(); const left=160, scale=7.25, rows=[{name:'Human',vals:[50.24,33.18,10.90,5.69]},{name:'GPT-5.6-Sol',vals:[34.60,23.70,19.43,22.27]}];
    [0,25,50,75,100].forEach(v=>{const x=left+v*scale;svgNode(stateSvg,'line',{x1:x,y1:28,x2:x,y2:188,class:'gridline'});const t=svgNode(stateSvg,'text',{x,y:215,'text-anchor':'middle'});t.textContent=String(v);});
    rows.forEach((row,ri)=>{const yy=52+ri*72, name=svgNode(stateSvg,'text',{x:left-14,y:yy+23,'text-anchor':'end','font-size':'15'});name.textContent=row.name;let x=left;
      row.vals.forEach((value,i)=>{const s=states[i], width=value*scale, rect=svgNode(stateSvg,'rect',{x,y:yy,width,height:36,fill:s.color,opacity:activeStates.has(s.key)?'.9':'.12',tabindex:'0',class:'data-mark'});
        const tip=e=>showTip(e,[`${row.name} · ${s.key}`,s[document.body.classList.contains('language-en')?'en':'zh'],`${value.toFixed(2)}%`]);rect.addEventListener('pointerenter',tip);rect.addEventListener('pointermove',tip);rect.addEventListener('pointerleave',hideTip);rect.addEventListener('focus',tip);rect.addEventListener('blur',hideTip);
        if(value>8){const txt=svgNode(stateSvg,'text',{x:x+width/2,y:yy+23,'text-anchor':'middle','font-size':'12',fill:'#07101b','pointer-events':'none'});txt.textContent=value.toFixed(1)+'%';} x+=width;
      });
    });
  }
  drawStates();
  const context=$('state-context-table');
  context.innerHTML='<thead><tr><th>Measure</th><th>Human</th><th>GPT-5.6-Sol</th></tr></thead><tbody><tr><td>Prior rotations</td><td>2.40</td><td>4.18</td></tr><tr><td>Prior coverage (%)</td><td>42.01</td><td>46.21</td></tr><tr><td>ΔIoU</td><td>0.0207</td><td>0.0071</td></tr><tr><td>−Δpose (°)</td><td>10.0420</td><td>2.8740</td></tr></tbody>';
  document.getElementById('language-toggle').addEventListener('click',()=>requestAnimationFrame(()=>{
    renderAblation();
    corrLegend.querySelectorAll('button').forEach((b,i)=>b.textContent=(document.body.classList.contains('language-en')?corrNames[0]:corrNames[1])[i]);
    stateLegend.querySelectorAll('button').forEach((b,i)=>b.textContent=`${states[i].key} · ${states[i][document.body.classList.contains('language-en')?'en':'zh']}`);
    const en=document.body.classList.contains('language-en');
    context.querySelector('thead').innerHTML=`<tr><th>${en?'Measure':'指标'}</th><th>${en?'Human':'人类'}</th><th>GPT-5.6-Sol</th></tr>`;
    context.querySelectorAll('tbody tr').forEach((tr,i)=>tr.cells[0].textContent=(en?['Prior rotations','Prior coverage (%)','ΔIoU','−Δpose (°)']:['此前旋转步数','此前覆盖率 (%)','ΔIoU','−Δpose (°)'])[i]);
  }));
})();
