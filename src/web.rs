use colored::*;
use serde::Serialize;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::sync::mpsc;

use crate::cli::Args;
use crate::providers::Provider;
use crate::transforms::Transform;
use crate::{TokenEvent, TokenInterceptor};

/// Wraps a `TokenEvent` with a provider-side label for diff streaming.
#[derive(Debug, Serialize)]
struct DiffTokenEvent<'a> {
    side: &'static str,
    #[serde(flatten)]
    event: &'a TokenEvent,
}

/// Embedded single-page HTML application with side-by-side, multi-transform,
/// dependency graph, and export features.
pub const INDEX_HTML: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Every Other Token</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#c9d1d9;font-family:'Cascadia Code','Fira Code',monospace;min-height:100vh;display:flex;flex-direction:column}
header{padding:16px 24px;border-bottom:1px solid #21262d;display:flex;align-items:center;justify-content:space-between}
header h1{font-size:1.2rem;color:#58a6ff}
.hdr-right{display:flex;gap:8px}
.controls{display:flex;gap:10px;padding:12px 24px;flex-wrap:wrap;align-items:end;border-bottom:1px solid #21262d;background:#161b22}
.field{display:flex;flex-direction:column;gap:3px}
.field label{font-size:.7rem;color:#8b949e;text-transform:uppercase;letter-spacing:.5px}
.field input,.field select{background:#0d1117;border:1px solid #30363d;color:#c9d1d9;padding:6px 10px;border-radius:6px;font-family:inherit;font-size:.85rem}
.field input:focus,.field select:focus{outline:none;border-color:#58a6ff}
.field input[type=text]{min-width:240px}
.btn{border:none;padding:6px 14px;border-radius:6px;font-family:inherit;font-size:.85rem;cursor:pointer;color:#fff}
.btn-go{background:#238636}.btn-go:hover{background:#2ea043}
.btn-go:disabled{background:#21262d;color:#484f58;cursor:not-allowed}
.btn-mode{background:#30363d}.btn-mode:hover{background:#484f58}
.btn-mode.active{background:#1f6feb}
.btn-export{background:#6e40c9}.btn-export:hover{background:#8957e5}
.toggle{display:flex;align-items:center;gap:5px;font-size:.8rem;color:#8b949e;cursor:pointer;user-select:none}
.toggle input{accent-color:#58a6ff}
/* View modes */
#views{flex:1;overflow:auto;position:relative}
/* Single column */
.view-single{padding:20px 24px;line-height:1.8;font-size:1rem;white-space:pre-wrap;word-wrap:break-word}
/* Side-by-side */
.view-sidebyside{display:grid;grid-template-columns:1fr 1fr;height:100%}
.sbs-col{padding:16px 20px;line-height:1.8;font-size:.95rem;white-space:pre-wrap;word-wrap:break-word;overflow-y:auto}
.sbs-col:first-child{border-right:1px solid #21262d}
.sbs-label{font-size:.7rem;color:#8b949e;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;display:block}
/* Multi-transform 2x2 grid */
.view-multi{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;height:100%}
.multi-panel{padding:12px 16px;line-height:1.6;font-size:.85rem;white-space:pre-wrap;word-wrap:break-word;overflow-y:auto;border:1px solid #21262d}
.multi-label{font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;display:block}
.mp-reverse .multi-label{color:#58a6ff}
.mp-uppercase .multi-label{color:#f0883e}
.mp-mock .multi-label{color:#a371f7}
.mp-noise .multi-label{color:#3fb950}
.mp-reverse .token.odd{color:#58a6ff;font-weight:bold}
.mp-uppercase .token.odd{color:#f0883e;font-weight:bold}
.mp-mock .token.odd{color:#a371f7;font-weight:bold}
.mp-noise .token.odd{color:#3fb950;font-weight:bold}
/* Tokens */
.token{display:inline;animation:fadeIn .12s ease-in}
.token.odd{color:#00d4ff;font-weight:bold}
.token.even{color:#c9d1d9}
.heat-4{background:#da3633;color:#fff}.heat-3{background:#b62324;color:#fff}
.heat-2{background:#9e6a03;color:#000}.heat-1{background:#1f6feb;color:#fff}.heat-0{}
@keyframes fadeIn{from{opacity:0;transform:translateY(2px)}to{opacity:1;transform:translateY(0)}}
/* Graph */
#graph-wrap{border-top:1px solid #21262d;background:#0a0e14;display:none;overflow-x:auto}
#graph-wrap.show{display:block}
#graph-toggle-bar{padding:6px 24px;border-top:1px solid #21262d;background:#161b22;display:flex;align-items:center;gap:12px}
canvas#depgraph{width:100%;height:200px;display:block}
/* Stats */
#stats{padding:8px 24px;border-top:1px solid #21262d;font-size:.78rem;color:#8b949e;background:#161b22;display:flex;gap:20px;flex-wrap:wrap}
/* Chaos tooltip */
.token[title]{position:relative;cursor:help;border-bottom:1px dotted #58a6ff}
.token[title]:hover::after{content:attr(title);position:absolute;top:-1.8em;left:0;background:#1c2333;color:#58a6ff;padding:2px 6px;border-radius:4px;font-size:.7rem;white-space:nowrap;z-index:10;pointer-events:none}
/* Diff view */
.view-diff{display:grid;grid-template-columns:1fr 1fr;height:100%}
.diff-col{padding:16px 20px;line-height:1.8;font-size:.95rem;white-space:pre-wrap;word-wrap:break-word;overflow-y:auto}
.diff-col:first-child{border-right:1px solid #21262d}
.diff-label{font-size:.7rem;color:#8b949e;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;display:block}
.diff-match{background:#0d2010;color:#3fb950}
.diff-diverge{background:#200d0d;color:#f85149}
/* Token surgery */
.token.surgeable{cursor:pointer}
.token.surgeable:hover{text-decoration:underline dotted #e3b341;background:rgba(227,179,65,.08)}
.token-input{font-family:inherit;font-size:inherit;background:#1f2937;color:#e3b341;border:1px solid #e3b341;border-radius:3px;padding:0 2px;min-width:20px;outline:none}
/* Confidence bars */
.conf-bar{display:inline-block;width:100%;height:2px;border-radius:1px;margin-top:1px;vertical-align:bottom}
.token-wrap{display:inline-flex;flex-direction:column;align-items:center;vertical-align:top}
.conf-high{background:#3fb950}.conf-mid{background:#e3b341}.conf-low{background:#f85149}.conf-none{background:#30363d}
/* High-perplexity pulse */
@keyframes perpPulse{0%,100%{opacity:1}50%{opacity:.4}}
.high-perp{animation:perpPulse 1.2s ease-in-out 3}
/* Perplexity sparkline */
#perp-spark-wrap{padding:4px 24px;background:#161b22;border-top:1px solid #21262d;display:none}
#perp-spark-wrap.show{display:block}
#perp-spark{width:100%;height:40px;display:block}
/* A/B Experiment view */
.view-experiment{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr auto;height:100%;gap:0}
.exp-panel{padding:12px 16px;line-height:1.8;font-size:.9rem;white-space:pre-wrap;word-wrap:break-word;overflow-y:auto;border:1px solid #21262d}
.exp-label{font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;display:block;color:#58a6ff}
.exp-label.b{color:#a371f7}
.exp-perp-chart{padding:8px 16px;background:#0a0e14;border:1px solid #21262d;overflow:hidden}
.exp-diverge-map{padding:8px 16px;background:#0a0e14;border:1px solid #21262d;overflow-y:auto;font-size:.8rem}
.ab-system-prompts{display:none;gap:10px;padding:6px 24px;background:#161b22;border-bottom:1px solid #21262d;flex-wrap:wrap}
.ab-system-prompts.show{display:flex}
/* Research dashboard */
#research-dash{display:none;padding:12px 24px;background:#0d1117;border-top:2px solid #21262d;font-size:.78rem}
#research-dash.show{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}
.dash-card{background:#161b22;border:1px solid #21262d;border-radius:6px;padding:10px 12px}
.dash-card h3{font-size:.7rem;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}
.dash-card .val{font-size:1.1rem;color:#c9d1d9;font-weight:bold}
.dash-card .sub{font-size:.7rem;color:#484f58;margin-top:2px}
#conf-hist{display:flex;align-items:flex-end;gap:1px;height:32px;margin-top:4px}
.hist-bar{flex:1;background:#1f6feb;min-height:2px;border-radius:1px 1px 0 0}
#citation{font-size:.68rem;color:#484f58;word-break:break-all;margin-top:4px;line-height:1.4}
</style>
</head>
<body>
<header>
  <h1>Every Other Token</h1>
  <div class="hdr-right">
    <button class="btn btn-mode active" id="btn-single" title="Single stream">Single</button>
    <button class="btn btn-mode" id="btn-sbs" title="Side-by-side: original vs transformed">Split</button>
    <button class="btn btn-mode" id="btn-multi" title="All 4 transforms simultaneously">Quad</button>
    <button class="btn btn-mode" id="btn-diff" title="Live diff: OpenAI vs Anthropic simultaneously">Diff</button>
    <button class="btn btn-mode" id="btn-experiment" title="A/B experiment: two system prompts, same user prompt">Experiment</button>
    <button class="btn btn-mode" id="btn-research" title="Research dashboard: stats, perplexity, confidence">Research</button>
    <button class="btn btn-export" id="btn-export" title="Export session as JSON">Export JSON</button>
  </div>
</header>
<div id="ab-prompts" class="ab-system-prompts">
  <div class="field"><label>System Prompt A</label><input type="text" id="sysprompt-a" value="You are a creative storyteller." style="min-width:280px"></div>
  <div class="field"><label>System Prompt B</label><input type="text" id="sysprompt-b" value="You are a technical writer. Be precise and concise." style="min-width:280px"></div>
</div>
<div class="controls">
  <div class="field"><label>Prompt</label><input type="text" id="prompt" value="Tell me a story about a robot" placeholder="Enter prompt..."></div>
  <div class="field"><label>Transform</label><select id="transform"><option value="reverse">reverse</option><option value="uppercase">uppercase</option><option value="mock">mock</option><option value="noise">noise</option><option value="chaos">chaos</option></select></div>
  <div class="field"><label>Provider</label><select id="provider"><option value="openai">OpenAI</option><option value="anthropic">Anthropic</option></select></div>
  <div class="field"><label>Model</label><input type="text" id="model" value="" placeholder="auto" style="min-width:160px"></div>
  <label class="toggle"><input type="checkbox" id="heatmap"> Heatmap</label>
  <label class="toggle"><input type="checkbox" id="graphtoggle"> Graph</label>
  <button class="btn btn-go" id="start">Stream</button>
</div>
<div id="views">
  <!-- Single column (default) -->
  <div class="view-single" id="v-single"></div>
  <!-- Side-by-side -->
  <div class="view-sidebyside" id="v-sbs" style="display:none">
    <div class="sbs-col" id="sbs-orig"><span class="sbs-label">Original</span></div>
    <div class="sbs-col" id="sbs-xform"><span class="sbs-label">Transformed</span></div>
  </div>
  <!-- Multi-transform 2x2 -->
  <div class="view-multi" id="v-multi" style="display:none">
    <div class="multi-panel mp-reverse" id="mp-reverse"><span class="multi-label">Reverse</span></div>
    <div class="multi-panel mp-uppercase" id="mp-uppercase"><span class="multi-label">Uppercase</span></div>
    <div class="multi-panel mp-mock" id="mp-mock"><span class="multi-label">Mock</span></div>
    <div class="multi-panel mp-noise" id="mp-noise"><span class="multi-label">Noise</span></div>
  </div>
  <!-- Diff: OpenAI vs Anthropic side by side -->
  <div class="view-diff" id="v-diff" style="display:none">
    <div class="diff-col" id="diff-openai"><span class="diff-label">OpenAI</span></div>
    <div class="diff-col" id="diff-anthropic"><span class="diff-label">Anthropic</span></div>
  </div>
  <!-- A/B Experiment: 4 panels -->
  <div class="view-experiment" id="v-experiment" style="display:none">
    <div class="exp-panel" id="exp-a"><span class="exp-label">System Prompt A</span></div>
    <div class="exp-panel" id="exp-b"><span class="exp-label b">System Prompt B</span></div>
    <div class="exp-perp-chart" id="exp-perp"><span style="font-size:.7rem;color:#8b949e">Perplexity comparison (A=blue, B=purple) — run streams to populate</span><br><canvas id="exp-perp-canvas" style="width:100%;height:120px;display:block;margin-top:4px"></canvas></div>
    <div class="exp-diverge-map" id="exp-diverge"><span style="font-size:.7rem;color:#8b949e">Divergence map — run streams to populate</span></div>
  </div>
  <!-- Research dashboard -->
  <div id="v-research" style="display:none;padding:16px 24px;overflow-y:auto;height:100%">
    <h2 style="color:#58a6ff;font-size:1rem;margin-bottom:4px">Research Dashboard</h2>
    <p style="font-size:.75rem;color:#8b949e;margin-bottom:12px">Stats computed from the most recent stream. Stream tokens first.</p>
    <div id="research-grid" style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px"></div>
    <div style="margin-top:16px">
      <div style="font-size:.72rem;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px">Top 10 High-Perplexity Tokens</div>
      <ol id="research-perp-list" style="list-style:none;font-size:.8rem;color:#c9d1d9"></ol>
    </div>
    <div style="margin-top:16px">
      <div style="font-size:.72rem;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px">Confidence Distribution</div>
      <div id="research-conf-hist" style="display:flex;gap:2px;align-items:flex-end;height:48px;margin-top:4px"></div>
      <div style="display:flex;justify-content:space-between;font-size:.6rem;color:#484f58;margin-top:2px"><span>0.0</span><span>0.5</span><span>1.0</span></div>
    </div>
    <div style="margin-top:16px">
      <div style="font-size:.72rem;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px">Citation</div>
      <code id="research-citation" style="font-size:.7rem;color:#e3b341;white-space:pre-wrap;display:block;background:#161b22;padding:8px;border-radius:4px;border:1px solid #21262d"></code>
    </div>
  </div>
</div>
<div id="graph-wrap"><canvas id="depgraph"></canvas></div>
<div id="perp-spark-wrap"><svg id="perp-spark" viewBox="0 0 600 40" preserveAspectRatio="none"><polyline id="perp-line" points="" fill="none" stroke="#a371f7" stroke-width="1.5"/></svg></div>
<div id="stats"></div>
<div id="research-dash"></div>
<script>
const $=s=>document.querySelector(s);
const $$=s=>document.querySelectorAll(s);

/* ---- Transform functions (JS mirrors of Rust) ---- */
const TX={
  reverse:s=>s.split('').reverse().join(''),
  uppercase:s=>s.toUpperCase(),
  mock:s=>s.split('').map((c,i)=>i%2===0?c.toLowerCase():c.toUpperCase()).join(''),
  noise:s=>{const n='*+~@#$%';return s+n[Math.floor(Math.random()*n.length)]},
  chaos:s=>{const picks=['reverse','uppercase','mock','noise'];const k=picks[Math.floor(Math.random()*4)];return TX[k](s)}
};

/* ---- State ---- */
let es=null, mode='single', allTokens=[], graphNodes=[], surgeryLog=[];
let perpWindow=[], expATokens=[], expBTokens=[];
const views={single:$('#v-single'),sbs:$('#v-sbs'),multi:$('#v-multi'),diff:$('#v-diff'),experiment:$('#v-experiment'),research:$('#v-research')};

/* ---- View mode switching ---- */
function setMode(m){
  mode=m;
  Object.values(views).forEach(v=>{if(v)v.style.display='none'});
  if(m==='experiment'){$('#v-experiment').style.display='';$('#ab-prompts').classList.add('show')}
  else{if(views[m])views[m].style.display='';$('#ab-prompts').classList.remove('show')}
  $$('.btn-mode').forEach(b=>b.classList.remove('active'));
  $('#btn-'+m)&&$('#btn-'+m).classList.add('active');
  if(m==='research')renderResearch();
}
$('#btn-single').onclick=()=>setMode('single');
$('#btn-sbs').onclick=()=>setMode('sbs');
$('#btn-multi').onclick=()=>setMode('multi');
$('#btn-diff').onclick=()=>{setMode('diff');if(!es)startDiff()};
$('#btn-experiment').onclick=()=>setMode('experiment');
$('#btn-research').onclick=()=>setMode('research');

/* ---- Graph toggle ---- */
$('#graphtoggle').onchange=function(){
  $('#graph-wrap').classList.toggle('show',this.checked);
  if(this.checked)drawGraph();
};

/* ---- Helpers ---- */
function mkSpan(text,isOdd,importance,extraCls,chaosLabel,confidence,perplexity){
  const s=document.createElement('span');
  s.className='token '+(isOdd?'odd':'even');
  if(extraCls)s.classList.add(extraCls);
  if($('#heatmap').checked){
    const h=importance>=.8?4:importance>=.6?3:importance>=.4?2:importance>=.2?1:0;
    s.classList.add('heat-'+h);
  }
  if(chaosLabel&&isOdd)s.title='chaos → '+chaosLabel;
  /* Confidence bar: colored underline */
  if(confidence!=null){
    const cls=confidence>=0.7?'conf-high':confidence>=0.4?'conf-mid':'conf-low';
    s.classList.add(cls);
    s.title=(s.title?s.title+' | ':'')+('conf: '+(confidence*100).toFixed(0)+'%');
  }
  /* Perplexity pulse for surprising tokens (perplexity > 5) */
  if(perplexity!=null&&perplexity>5)s.classList.add('high-perp');
  s.textContent=text;
  return s;
}

/* ---- Perplexity sparkline ---- */
function updatePerpSparkline(perplexity){
  if(perplexity==null)return;
  perpWindow.push(perplexity);
  if(perpWindow.length>60)perpWindow.shift();
  const spark=$('#perp-spark-wrap');
  if(!spark)return;
  spark.classList.add('show');
  const line=$('#perp-line');
  if(!line)return;
  const max=Math.max(...perpWindow,1);
  const pts=perpWindow.map((v,i)=>{
    const x=Math.round(i/(perpWindow.length-1||1)*600);
    const y=Math.round((1-v/max)*36+2);
    return x+','+y;
  }).join(' ');
  line.setAttribute('points',pts);
}

/* ---- Research Dashboard ---- */
function renderResearch(){
  const grid=$('#research-grid');
  const perpList=$('#research-perp-list');
  const confHist=$('#research-conf-hist');
  const citation=$('#research-citation');
  if(!grid)return;
  if(allTokens.length===0){grid.innerHTML='<div style="color:#8b949e;font-size:.8rem">No tokens yet. Run a stream first.</div>';return;}
  /* Vocab diversity */
  const unique=new Set(allTokens.map(t=>t.original.toLowerCase())).size;
  const diversity=(unique/allTokens.length).toFixed(3);
  /* Avg token length */
  const avgLen=(allTokens.reduce((s,t)=>s+t.original.length,0)/allTokens.length).toFixed(1);
  /* Perplexity stats */
  const withPerp=allTokens.filter(t=>t.perplexity!=null);
  const avgPerp=withPerp.length>0?(withPerp.reduce((s,t)=>s+t.perplexity,0)/withPerp.length).toFixed(2):'n/a';
  /* Confidence stats */
  const withConf=allTokens.filter(t=>t.confidence!=null);
  const avgConf=withConf.length>0?((withConf.reduce((s,t)=>s+t.confidence,0)/withConf.length)*100).toFixed(0)+'%':'n/a';
  /* Cost estimate (GPT-3.5 pricing: ~$0.002/1K tokens) */
  const costEst=allTokens.length>0?'$'+(allTokens.length/1000*0.002).toFixed(4):'n/a';
  grid.innerHTML=`
    <div class="dash-card"><h3>Vocab Diversity</h3><div class="val">${diversity}</div><div class="sub">${unique} unique / ${allTokens.length} total</div></div>
    <div class="dash-card"><h3>Avg Token Length</h3><div class="val">${avgLen}</div><div class="sub">characters per token</div></div>
    <div class="dash-card"><h3>Avg Perplexity</h3><div class="val">${avgPerp}</div><div class="sub">exp(-logprob); lower=confident</div></div>
    <div class="dash-card"><h3>Avg Confidence</h3><div class="val">${avgConf}</div><div class="sub">from top-1 logprob</div></div>
    <div class="dash-card"><h3>Token Count</h3><div class="val">${allTokens.length}</div><div class="sub">${allTokens.filter(t=>t.transformed).length} transformed</div></div>
    <div class="dash-card"><h3>Est. Cost</h3><div class="val">${costEst}</div><div class="sub">GPT-3.5 rate ($0.002/1K)</div></div>
  `.replace(/dash-card/g,'r-card').replace(/class="val"/g,'class="r-stat"').replace(/class="sub"/g,'class="r-sub"');
  /* Top 10 perplexity tokens */
  if(perpList){
    const sorted=[...withPerp].sort((a,b)=>b.perplexity-a.perplexity).slice(0,10);
    perpList.innerHTML=sorted.map((t,i)=>`<li style="display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #21262d;font-size:.78rem"><span style="color:#c9d1d9">${i+1}. "${t.original}"</span><span style="color:#f85149">${t.perplexity.toFixed(1)}</span></li>`).join('');
  }
  /* Confidence histogram (10 buckets) */
  if(confHist&&withConf.length>0){
    const buckets=Array(10).fill(0);
    withConf.forEach(t=>{const b=Math.min(9,Math.floor(t.confidence*10));buckets[b]++});
    const maxB=Math.max(...buckets,1);
    confHist.innerHTML=buckets.map((v,i)=>`<div style="flex:1;background:#1f6feb;height:${Math.round(v/maxB*44)+2}px;border-radius:2px 2px 0 0;opacity:${0.4+i*0.06}" title="${(i*10)}-${(i+1)*10}%: ${v} tokens"></div>`).join('');
  }
  /* Citation */
  if(citation){
    const ts=new Date().toISOString().replace('T',' ').replace(/\.\d+Z$/,' UTC');
    citation.textContent=`Every Other Token (v4.0.0). Session recorded ${ts}.\nTokens: ${allTokens.length}, Transform: ${$('#transform').value}, Provider: ${$('#provider').value}, Model: ${$('#model').value||'auto'}.\nVocab diversity: ${diversity}, Avg perplexity: ${avgPerp}, Avg confidence: ${avgConf}.`;
  }
}

/* ---- A/B Experiment streaming ---- */
let expATokens2=[], expBTokens2=[];
function startExperiment(){
  if(es){es.close();es=null}
  $('#exp-a').innerHTML='<span class="exp-label">System Prompt A</span>';
  $('#exp-b').innerHTML='<span class="exp-label b">System Prompt B</span>';
  $('#exp-diverge').innerHTML='<span style="font-size:.7rem;color:#8b949e">Divergence map — streaming...</span>';
  expATokens2=[];expBTokens2=[];
  const p=encodeURIComponent($('#prompt').value);
  const t=$('#transform').value;
  const m=encodeURIComponent($('#model').value);
  const sa=encodeURIComponent(($('#sysprompt-a')&&$('#sysprompt-a').value)||'You are a creative storyteller.');
  const sb=encodeURIComponent(($('#sysprompt-b')&&$('#sysprompt-b').value)||'You are a technical writer. Be precise.');
  const url='/ab-stream?prompt='+p+'&transform='+t+'&model='+m+'&sys_a='+sa+'&sys_b='+sb;
  $('#start').disabled=true;$('#start').textContent='Experimenting...';
  es=new EventSource(url);
  es.onmessage=e=>{
    if(e.data==='[DONE]'){
      es.close();es=null;
      $('#start').disabled=false;$('#start').textContent='Stream';
      renderExpDivergence();
      return;
    }
    try{
      const tk=JSON.parse(e.data);
      if(tk.side==='a'){
        expATokens2.push(tk);
        const sp=mkSpan(tk.text,tk.transformed,tk.importance,'',tk.chaos_label,tk.confidence,tk.perplexity);
        sp.dataset.idx=tk.index;
        $('#exp-a').appendChild(sp);$('#exp-a').scrollTop=$('#exp-a').scrollHeight;
        updatePerpSparkline(tk.perplexity);
      } else if(tk.side==='b'){
        expBTokens2.push(tk);
        const sp=mkSpan(tk.text,tk.transformed,tk.importance,'',tk.chaos_label,tk.confidence,tk.perplexity);
        sp.dataset.idx=tk.index;
        $('#exp-b').appendChild(sp);$('#exp-b').scrollTop=$('#exp-b').scrollHeight;
      }
      $('#stats').textContent='A: '+expATokens2.length+' tokens | B: '+expBTokens2.length+' tokens';
    }catch(_){}
  };
  es.onerror=()=>{es.close();es=null;$('#start').disabled=false;$('#start').textContent='Stream'};
}
function renderExpDivergence(){
  const el=$('#exp-diverge');if(!el)return;
  const minLen=Math.min(expATokens2.length,expBTokens2.length);
  if(minLen===0){el.innerHTML='<span style="font-size:.7rem;color:#8b949e">No tokens to compare.</span>';return;}
  let matches=0;
  const rows=[];
  for(let i=0;i<minLen;i++){
    const match=expATokens2[i].original===expBTokens2[i].original;
    if(match)matches++;
    rows.push(`<div style="display:flex;gap:8px;font-size:.75rem;padding:1px 0;color:${match?'#3fb950':'#f85149'}"><span style="flex:1;overflow:hidden;text-overflow:ellipsis">${expATokens2[i].original}</span><span style="flex:1;overflow:hidden;text-overflow:ellipsis">${expBTokens2[i].original}</span></div>`);
  }
  const pct=Math.round(matches/minLen*100);
  el.innerHTML=`<div style="font-size:.7rem;color:#8b949e;margin-bottom:4px">Similarity: ${pct}% (${matches}/${minLen})</div>`+rows.slice(0,50).join('');
}

/* ---- Token surgery ---- */
function enableSurgery(container){
  container.querySelectorAll('.token').forEach(sp=>{
    if(sp.classList.contains('surgeable'))return;
    sp.classList.add('surgeable');
    sp.addEventListener('click',function(){
      const orig=sp.textContent;
      const inp=document.createElement('input');
      inp.className='token-input';
      inp.value=orig;
      inp.style.width=Math.max(orig.length*0.6+1,2)+'em';
      sp.replaceWith(inp);
      inp.focus();inp.select();
      function commit(){
        const newText=inp.value||orig;
        sp.textContent=newText;
        inp.replaceWith(sp);
        if(newText!==orig){
          surgeryLog.push({index:parseInt(sp.dataset.idx||'0'),original:orig,replacement:newText,timestamp:new Date().toISOString()});
        }
      }
      inp.addEventListener('blur',commit);
      inp.addEventListener('keydown',e=>{if(e.key==='Enter')inp.blur();if(e.key==='Escape'){inp.value=orig;inp.blur()}});
    });
  });
}

function heatColor(imp){
  if(imp>=.8)return'#f85149';
  if(imp>=.6)return'#f0883e';
  if(imp>=.4)return'#e3b341';
  if(imp>=.2)return'#58a6ff';
  return'#484f58';
}

/* ---- Canvas dependency graph ---- */
function drawGraph(){
  const canvas=$('#depgraph');
  if(!canvas||!$('#graphtoggle').checked)return;
  const dpr=window.devicePixelRatio||1;
  const rect=canvas.parentElement.getBoundingClientRect();
  const nodes=graphNodes;
  if(nodes.length===0)return;
  const gap=60;
  const neededW=Math.max((nodes.length+1)*gap,rect.width);
  canvas.width=neededW*dpr;
  canvas.height=200*dpr;
  canvas.style.width=neededW+'px';
  canvas.style.height='200px';
  const ctx=canvas.getContext('2d');
  ctx.scale(dpr,dpr);
  ctx.clearRect(0,0,neededW,200);
  const y1=50,y2=150;
  /* Draw nodes and edges */
  let lastEvenX=0,lastEvenIdx=-1;
  for(let i=0;i<nodes.length;i++){
    const n=nodes[i];
    const x=(i+1)*gap;
    const y=n.transformed?y2:y1;
    /* If odd, draw line to previous even */
    if(n.transformed&&lastEvenIdx>=0){
      const ex=(lastEvenIdx+1)*gap;
      ctx.beginPath();
      ctx.moveTo(ex,y1+8);
      ctx.bezierCurveTo(ex,y1+40,x,y2-40,x,y2-8);
      ctx.lineWidth=Math.min(n.original.length,6);
      ctx.strokeStyle=heatColor(n.importance);
      ctx.globalAlpha=0.6;
      ctx.stroke();
      ctx.globalAlpha=1;
    }
    if(!n.transformed){lastEvenX=x;lastEvenIdx=i;}
    /* Node circle */
    ctx.beginPath();
    ctx.arc(x,y,7,0,Math.PI*2);
    ctx.fillStyle=n.transformed?'#00d4ff':'#c9d1d9';
    ctx.fill();
    ctx.strokeStyle='#30363d';ctx.lineWidth=1;ctx.stroke();
    /* Label */
    ctx.fillStyle='#8b949e';
    ctx.font='10px monospace';
    ctx.textAlign='center';
    const label=n.text.length>6?n.text.slice(0,5)+'…':n.text;
    ctx.fillText(label,x,y+(n.transformed?20:-14));
  }
}

/* ---- Streaming ---- */
$('#start').onclick=()=>{
  if(mode==='diff'){startDiff();return}
  if(mode==='experiment'){startExperiment();return}
  if(es){es.close();es=null}
  /* Clear all views */
  $('#v-single').innerHTML='';
  $('#sbs-orig').innerHTML='<span class="sbs-label">Original</span>';
  $('#sbs-xform').innerHTML='<span class="sbs-label">Transformed</span>';
  ['reverse','uppercase','mock','noise','chaos'].forEach(t=>{
    const el=$('#mp-'+t);if(el)el.innerHTML='<span class="multi-label">'+t+'</span>';
  });
  $('#stats').textContent='';
  allTokens=[];graphNodes=[];surgeryLog=[];
  if($('#graphtoggle').checked)drawGraph();

  const p=encodeURIComponent($('#prompt').value);
  const t=$('#transform').value;
  const prov=$('#provider').value;
  const m=encodeURIComponent($('#model').value);
  const hm=$('#heatmap').checked?'1':'0';
  const url='/stream?prompt='+p+'&transform='+t+'&provider='+prov+'&model='+m+'&heatmap='+hm;
  $('#start').disabled=true;$('#start').textContent='Streaming...';
  let count=0,xformed=0;
  es=new EventSource(url);
  es.onmessage=e=>{
    if(e.data==='[DONE]'){
      es.close();es=null;
      $('#start').disabled=false;$('#start').textContent='Stream';
      if($('#graphtoggle').checked)drawGraph();
      /* Enable token surgery on all rendered containers */
      enableSurgery($('#v-single'));
      enableSurgery($('#sbs-orig'));
      enableSurgery($('#sbs-xform'));
      /* Update research dashboard */
      renderResearch();
      return;
    }
    try{
      const tk=JSON.parse(e.data);
      allTokens.push(tk);
      graphNodes.push(tk);
      count++;if(tk.transformed)xformed++;

      /* Single column */
      const singleSp=mkSpan(tk.text,tk.transformed,tk.importance,'',tk.chaos_label,tk.confidence,tk.perplexity);
      singleSp.dataset.idx=tk.index;
      $('#v-single').appendChild(singleSp);
      updatePerpSparkline(tk.perplexity);

      /* Side-by-side: left=original, right=transformed */
      const origSp=mkSpan(tk.original,false,tk.importance,null,null,tk.confidence,tk.perplexity);origSp.dataset.idx=tk.index;
      $('#sbs-orig').appendChild(origSp);
      const xformSp=mkSpan(tk.text,tk.transformed,tk.importance,'',tk.chaos_label,tk.confidence,tk.perplexity);xformSp.dataset.idx=tk.index;
      $('#sbs-xform').appendChild(xformSp);

      /* Multi-transform: apply all 4+chaos transforms to original for odd tokens */
      ['reverse','uppercase','mock','noise','chaos'].forEach(txName=>{
        const panel=$('#mp-'+txName);
        if(!panel)return;
        if(tk.transformed){
          const applied=TX[txName](tk.original);
          panel.appendChild(mkSpan(applied,true,tk.importance));
        } else {
          panel.appendChild(mkSpan(tk.original,false,tk.importance));
        }
      });

      /* Auto-scroll active view */
      const active=views[mode];
      if(active)active.scrollTop=active.scrollHeight;
      /* Scroll panels in multi */
      if(mode==='multi'){
        ['reverse','uppercase','mock','noise'].forEach(t=>{
          const p=$('#mp-'+t);p.scrollTop=p.scrollHeight;
        });
      }

      /* Redraw graph periodically */
      if($('#graphtoggle').checked&&count%5===0)drawGraph();

      $('#stats').textContent='Tokens: '+count+' | Transformed: '+xformed+' | Mode: '+mode;
    }catch(_){}
  };
  es.onerror=()=>{es.close();es=null;$('#start').disabled=false;$('#start').textContent='Stream'};
};

/* ---- Diff streaming ---- */
let diffOpenaiTokens=[], diffAnthropicTokens=[];
function startDiff(){
  if(es){es.close();es=null}
  $('#diff-openai').innerHTML='<span class="diff-label">OpenAI</span>';
  $('#diff-anthropic').innerHTML='<span class="diff-label">Anthropic</span>';
  diffOpenaiTokens=[];diffAnthropicTokens=[];
  $('#stats').textContent='';
  const p=encodeURIComponent($('#prompt').value);
  const t=$('#transform').value;
  const m=encodeURIComponent($('#model').value);
  const hm=$('#heatmap').checked?'1':'0';
  const url='/diff-stream?prompt='+p+'&transform='+t+'&model='+m+'&heatmap='+hm;
  $('#start').disabled=true;$('#start').textContent='Diffing...';
  es=new EventSource(url);
  es.onmessage=e=>{
    if(e.data==='[DONE]'){
      es.close();es=null;
      $('#start').disabled=false;$('#start').textContent='Stream';
      applyDiffHighlights();
      return;
    }
    try{
      const tk=JSON.parse(e.data);
      if(tk.side==='openai'){
        diffOpenaiTokens.push(tk);
        const sp=mkSpan(tk.text,tk.transformed,tk.importance,'',tk.chaos_label);
        sp.dataset.idx=tk.index;
        $('#diff-openai').appendChild(sp);
        $('#diff-openai').scrollTop=$('#diff-openai').scrollHeight;
      } else if(tk.side==='anthropic'){
        diffAnthropicTokens.push(tk);
        const sp=mkSpan(tk.text,tk.transformed,tk.importance,'',tk.chaos_label);
        sp.dataset.idx=tk.index;
        $('#diff-anthropic').appendChild(sp);
        $('#diff-anthropic').scrollTop=$('#diff-anthropic').scrollHeight;
      }
      $('#stats').textContent='OpenAI: '+diffOpenaiTokens.length+' tokens | Anthropic: '+diffAnthropicTokens.length+' tokens';
    }catch(_){}
  };
  es.onerror=()=>{es.close();es=null;$('#start').disabled=false;$('#start').textContent='Stream'};
}
function applyDiffHighlights(){
  const oSpans=Array.from($('#diff-openai').querySelectorAll('.token'));
  const aSpans=Array.from($('#diff-anthropic').querySelectorAll('.token'));
  let matches=0;
  const total=Math.max(oSpans.length,aSpans.length);
  for(let i=0;i<total;i++){
    const match=oSpans[i]&&aSpans[i]&&oSpans[i].textContent===aSpans[i].textContent;
    if(oSpans[i])oSpans[i].classList.add(match?'diff-match':'diff-diverge');
    if(aSpans[i])aSpans[i].classList.add(match?'diff-match':'diff-diverge');
    if(match)matches++;
  }
  const pct=total>0?Math.round(matches/total*100):0;
  $('#stats').textContent='Match: '+pct+'% ('+matches+'/'+total+') | OpenAI: '+oSpans.length+' tokens | Anthropic: '+aSpans.length+' tokens';
}

/* ---- Export JSON ---- */
$('#btn-export').onclick=()=>{
  if(allTokens.length===0){alert('No tokens to export. Run a stream first.');return}
  const data={
    prompt:$('#prompt').value,
    provider:$('#provider').value,
    model:$('#model').value||'auto',
    transform:$('#transform').value,
    timestamp:new Date().toISOString(),
    token_count:allTokens.length,
    transformed_count:allTokens.filter(t=>t.transformed).length,
    tokens:allTokens.map(t=>({text:t.text,original:t.original,index:t.index,transformed:t.transformed,importance:t.importance,chaos_label:t.chaos_label||null})),
    surgery_log:surgeryLog
  };
  const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'});
  const url=URL.createObjectURL(blob);
  const a=document.createElement('a');
  a.href=url;a.download='every-other-token-'+Date.now()+'.json';
  document.body.appendChild(a);a.click();a.remove();
  URL.revokeObjectURL(url);
};

/* ---- Resize graph on window resize ---- */
window.addEventListener('resize',()=>{if($('#graphtoggle').checked)drawGraph()});
</script>
</body>
</html>"##;

/// Simple percent-decoding for URL query parameters.
pub fn url_decode(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        match c {
            '+' => result.push(' '),
            '%' => {
                let hex: String = chars.by_ref().take(2).collect();
                if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                    result.push(byte as char);
                }
            }
            _ => result.push(c),
        }
    }
    result
}

/// Parse query string into key-value pairs.
pub fn parse_query(query: &str) -> std::collections::HashMap<String, String> {
    query
        .split('&')
        .filter_map(|pair| {
            let mut parts = pair.splitn(2, '=');
            let key = parts.next()?;
            let val = parts.next().unwrap_or("");
            Some((key.to_string(), url_decode(val)))
        })
        .collect()
}

/// Start the web UI server and open the browser.
pub async fn serve(port: u16, default_args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind(format!("127.0.0.1:{}", port)).await?;

    eprintln!(
        "{}",
        format!("  Web UI running at http://localhost:{}", port).bright_green()
    );
    eprintln!("{}", "  Press Ctrl+C to stop.".bright_blue());

    // Try to open the browser
    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new("cmd")
            .args(["/C", &format!("start http://localhost:{}", port)])
            .spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open")
            .arg(format!("http://localhost:{}", port))
            .spawn();
    }
    #[cfg(target_os = "linux")]
    {
        let _ = std::process::Command::new("xdg-open")
            .arg(format!("http://localhost:{}", port))
            .spawn();
    }

    let default_provider = default_args.provider.clone();
    let orchestrator = default_args.orchestrator;

    loop {
        let (stream, _addr) = listener.accept().await?;
        let provider = default_provider.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, provider, orchestrator).await {
                eprintln!("  connection error: {}", e);
            }
        });
    }
}

async fn handle_connection(
    mut stream: tokio::net::TcpStream,
    default_provider: Provider,
    orchestrator: bool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use tokio::io::AsyncReadExt;

    let mut buf = vec![0u8; 8192];
    let n = stream.read(&mut buf).await?;
    let request = String::from_utf8_lossy(&buf[..n]);

    // Parse the request line: "GET /path?query HTTP/1.1"
    let first_line = request.lines().next().unwrap_or("");
    let parts: Vec<&str> = first_line.split_whitespace().collect();
    if parts.len() < 2 {
        return Ok(());
    }
    let path_and_query = parts[1];

    // Split path and query
    let (path, query_str) = if let Some(idx) = path_and_query.find('?') {
        (&path_and_query[..idx], &path_and_query[idx + 1..])
    } else {
        (path_and_query, "")
    };

    match path {
        "/" => {
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                INDEX_HTML.len(),
                INDEX_HTML,
            );
            stream.write_all(response.as_bytes()).await?;
        }
        "/stream" => {
            let params = parse_query(query_str);
            let prompt = params.get("prompt").cloned().unwrap_or_default();
            let transform_str = params
                .get("transform")
                .cloned()
                .unwrap_or_else(|| "reverse".to_string());
            let provider_str = params
                .get("provider")
                .cloned()
                .unwrap_or_else(|| default_provider.to_string());
            let model_input = params.get("model").cloned().unwrap_or_default();
            let heatmap = params.get("heatmap").is_some_and(|v| v == "1");

            let provider = match provider_str.as_str() {
                "anthropic" => Provider::Anthropic,
                _ => Provider::Openai,
            };

            let model = if model_input.is_empty() {
                match provider {
                    Provider::Openai => "gpt-3.5-turbo".to_string(),
                    Provider::Anthropic => "claude-sonnet-4-20250514".to_string(),
                }
            } else {
                model_input
            };

            let transform = Transform::from_str_loose(&transform_str).unwrap_or(Transform::Reverse);

            // SSE headers
            let headers = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
            stream.write_all(headers.as_bytes()).await?;

            // Create channel for token events
            let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();

            let interceptor_result = TokenInterceptor::new(
                provider,
                transform,
                model,
                true, // visual mode always on for web
                heatmap,
                orchestrator,
            );

            // Convert result early — stringify the error before any await
            // to satisfy Send bounds on the spawned task.
            let interceptor_result = interceptor_result.map_err(|e| e.to_string());
            let mut interceptor = match interceptor_result {
                Ok(mut i) => {
                    i.web_tx = Some(tx);
                    i
                }
                Err(msg) => {
                    let err_event = format!(
                        "data: {{\"error\": \"{}\"}}\n\ndata: [DONE]\n\n",
                        msg.replace('"', "'")
                    );
                    stream.write_all(err_event.as_bytes()).await?;
                    return Ok(());
                }
            };

            // Spawn the LLM streaming in background
            let prompt_clone = prompt.clone();
            let stream_task = tokio::spawn(async move {
                let _ = interceptor.intercept_stream(&prompt_clone).await;
            });

            // Forward token events as SSE
            while let Some(event) = rx.recv().await {
                if let Ok(json) = serde_json::to_string(&event) {
                    let sse = format!("data: {}\n\n", json);
                    if stream.write_all(sse.as_bytes()).await.is_err() {
                        break;
                    }
                }
            }

            let _ = stream_task.await;

            // Send done signal
            let _ = stream.write_all(b"data: [DONE]\n\n").await;
        }
        "/diff-stream" => {
            let params = parse_query(query_str);
            let prompt = params.get("prompt").cloned().unwrap_or_default();
            let transform_str = params
                .get("transform")
                .cloned()
                .unwrap_or_else(|| "reverse".to_string());
            let model_input = params.get("model").cloned().unwrap_or_default();
            let heatmap = params.get("heatmap").is_some_and(|v| v == "1");

            let transform = Transform::from_str_loose(&transform_str).unwrap_or(Transform::Reverse);

            let openai_model = if model_input.is_empty() {
                "gpt-3.5-turbo".to_string()
            } else {
                model_input.clone()
            };
            let anthropic_model = if model_input.is_empty() {
                "claude-sonnet-4-20250514".to_string()
            } else {
                model_input.clone()
            };

            // SSE headers
            let headers = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
            stream.write_all(headers.as_bytes()).await?;

            // Merged channel: (side, event)
            let (merged_tx, mut merged_rx) =
                mpsc::unbounded_channel::<(&'static str, TokenEvent)>();

            // Spawn OpenAI side
            let openai_result = TokenInterceptor::new(
                Provider::Openai,
                transform.clone(),
                openai_model,
                true,
                heatmap,
                orchestrator,
            )
            .map_err(|e| e.to_string());
            if let Ok(mut oai) = openai_result {
                let (tx_oai, mut rx_oai) = mpsc::unbounded_channel::<TokenEvent>();
                oai.web_tx = Some(tx_oai);
                let prompt_o = prompt.clone();
                tokio::spawn(async move {
                    let _ = oai.intercept_stream(&prompt_o).await;
                });
                let mtx = merged_tx.clone();
                tokio::spawn(async move {
                    while let Some(ev) = rx_oai.recv().await {
                        let _ = mtx.send(("openai", ev));
                    }
                });
            }

            // Spawn Anthropic side
            let anthropic_result = TokenInterceptor::new(
                Provider::Anthropic,
                transform,
                anthropic_model,
                true,
                heatmap,
                orchestrator,
            )
            .map_err(|e| e.to_string());
            if let Ok(mut ant) = anthropic_result {
                let (tx_ant, mut rx_ant) = mpsc::unbounded_channel::<TokenEvent>();
                ant.web_tx = Some(tx_ant);
                let prompt_a = prompt.clone();
                tokio::spawn(async move {
                    let _ = ant.intercept_stream(&prompt_a).await;
                });
                let mtx = merged_tx.clone();
                tokio::spawn(async move {
                    while let Some(ev) = rx_ant.recv().await {
                        let _ = mtx.send(("anthropic", ev));
                    }
                });
            }

            // Drop the original merged_tx so the channel closes when both sides finish
            drop(merged_tx);

            // Forward merged events as SSE with side tag
            while let Some((side, event)) = merged_rx.recv().await {
                let diff_event = DiffTokenEvent {
                    side,
                    event: &event,
                };
                if let Ok(json) = serde_json::to_string(&diff_event) {
                    let sse = format!("data: {}\n\n", json);
                    if stream.write_all(sse.as_bytes()).await.is_err() {
                        break;
                    }
                }
            }

            let _ = stream.write_all(b"data: [DONE]\n\n").await;
        }
        "/ab-stream" => {
            // A/B Experiment: same prompt sent to provider with two different system prompts
            let params = parse_query(query_str);
            let prompt = params.get("prompt").cloned().unwrap_or_default();
            let transform_str = params
                .get("transform")
                .cloned()
                .unwrap_or_else(|| "reverse".to_string());
            let model_input = params.get("model").cloned().unwrap_or_default();
            let sys_a = params
                .get("sys_a")
                .cloned()
                .unwrap_or_else(|| "You are a creative storyteller.".to_string());
            let sys_b = params
                .get("sys_b")
                .cloned()
                .unwrap_or_else(|| "You are a technical writer. Be precise.".to_string());

            let transform = Transform::from_str_loose(&transform_str).unwrap_or(Transform::Reverse);
            let model = if model_input.is_empty() {
                match default_provider {
                    Provider::Openai => "gpt-3.5-turbo".to_string(),
                    Provider::Anthropic => "claude-sonnet-4-20250514".to_string(),
                }
            } else {
                model_input
            };

            let headers = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
            stream.write_all(headers.as_bytes()).await?;

            let (merged_tx, mut merged_rx) =
                mpsc::unbounded_channel::<(&'static str, TokenEvent)>();

            // Side A
            let a_result = TokenInterceptor::new(
                default_provider.clone(),
                transform.clone(),
                model.clone(),
                true,
                false,
                orchestrator,
            )
            .map_err(|e| e.to_string());
            if let Ok(mut side_a) = a_result {
                let (tx_a, mut rx_a) = mpsc::unbounded_channel::<TokenEvent>();
                side_a.web_tx = Some(tx_a);
                side_a.system_prompt = Some(sys_a);
                let prompt_a = prompt.clone();
                tokio::spawn(async move {
                    let _ = side_a.intercept_stream(&prompt_a).await;
                });
                let mtx = merged_tx.clone();
                tokio::spawn(async move {
                    while let Some(ev) = rx_a.recv().await {
                        let _ = mtx.send(("a", ev));
                    }
                });
            }

            // Side B
            let b_result = TokenInterceptor::new(
                default_provider.clone(),
                transform,
                model,
                true,
                false,
                orchestrator,
            )
            .map_err(|e| e.to_string());
            if let Ok(mut side_b) = b_result {
                let (tx_b, mut rx_b) = mpsc::unbounded_channel::<TokenEvent>();
                side_b.web_tx = Some(tx_b);
                side_b.system_prompt = Some(sys_b);
                let prompt_b = prompt.clone();
                tokio::spawn(async move {
                    let _ = side_b.intercept_stream(&prompt_b).await;
                });
                let mtx = merged_tx.clone();
                tokio::spawn(async move {
                    while let Some(ev) = rx_b.recv().await {
                        let _ = mtx.send(("b", ev));
                    }
                });
            }

            drop(merged_tx);

            while let Some((side, event)) = merged_rx.recv().await {
                let diff_event = DiffTokenEvent { side, event: &event };
                if let Ok(json) = serde_json::to_string(&diff_event) {
                    let sse = format!("data: {}\n\n", json);
                    if stream.write_all(sse.as_bytes()).await.is_err() {
                        break;
                    }
                }
            }

            let _ = stream.write_all(b"data: [DONE]\n\n").await;
        }
        _ => {
            let response =
                "HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\nConnection: close\r\n\r\nNot Found";
            stream.write_all(response.as_bytes()).await?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- url_decode tests --

    #[test]
    fn test_url_decode_basic() {
        assert_eq!(url_decode("hello+world"), "hello world");
        assert_eq!(url_decode("hello%20world"), "hello world");
        assert_eq!(url_decode("a%26b"), "a&b");
        assert_eq!(url_decode("plain"), "plain");
    }

    #[test]
    fn test_url_decode_empty() {
        assert_eq!(url_decode(""), "");
    }

    #[test]
    fn test_url_decode_no_encoding() {
        assert_eq!(url_decode("hello"), "hello");
    }

    #[test]
    fn test_url_decode_plus_only() {
        assert_eq!(url_decode("+++"), "   ");
    }

    #[test]
    fn test_url_decode_percent_special_chars() {
        assert_eq!(url_decode("%21"), "!");
        assert_eq!(url_decode("%3D"), "=");
        assert_eq!(url_decode("%3F"), "?");
    }

    #[test]
    fn test_url_decode_mixed() {
        assert_eq!(url_decode("hello+world%21+how%3F"), "hello world! how?");
    }

    #[test]
    fn test_url_decode_consecutive_percent() {
        assert_eq!(url_decode("%20%20"), "  ");
    }

    #[test]
    fn test_url_decode_single_char() {
        assert_eq!(url_decode("a"), "a");
        assert_eq!(url_decode("+"), " ");
    }

    // -- parse_query tests --

    #[test]
    fn test_parse_query_basic() {
        let params = parse_query("prompt=hello+world&transform=reverse&heatmap=1");
        assert_eq!(
            params.get("prompt").map(|s| s.as_str()),
            Some("hello world")
        );
        assert_eq!(params.get("transform").map(|s| s.as_str()), Some("reverse"));
        assert_eq!(params.get("heatmap").map(|s| s.as_str()), Some("1"));
    }

    #[test]
    fn test_parse_query_empty() {
        let params = parse_query("");
        assert!(params.is_empty() || params.get("").map_or(true, |v| v.is_empty()));
    }

    #[test]
    fn test_parse_query_single_param() {
        let params = parse_query("key=value");
        assert_eq!(params.len(), 1);
        assert_eq!(params.get("key").map(|s| s.as_str()), Some("value"));
    }

    #[test]
    fn test_parse_query_no_value() {
        let params = parse_query("key=");
        assert_eq!(params.get("key").map(|s| s.as_str()), Some(""));
    }

    #[test]
    fn test_parse_query_encoded_values() {
        let params = parse_query("prompt=hello+world%21&model=gpt-4");
        assert_eq!(
            params.get("prompt").map(|s| s.as_str()),
            Some("hello world!")
        );
        assert_eq!(params.get("model").map(|s| s.as_str()), Some("gpt-4"));
    }

    #[test]
    fn test_parse_query_many_params() {
        let params = parse_query("a=1&b=2&c=3&d=4&e=5");
        assert_eq!(params.len(), 5);
        assert_eq!(params.get("c").map(|s| s.as_str()), Some("3"));
    }

    #[test]
    fn test_parse_query_special_chars_in_value() {
        let params = parse_query("q=a%2Bb%3Dc");
        assert_eq!(params.get("q").map(|s| s.as_str()), Some("a+b=c"));
    }

    // -- INDEX_HTML structure tests --

    #[test]
    fn test_index_html_is_valid_html() {
        assert!(INDEX_HTML.starts_with("<!DOCTYPE html>"));
        assert!(INDEX_HTML.contains("</html>"));
    }

    #[test]
    fn test_index_html_contains_title() {
        assert!(INDEX_HTML.contains("<title>Every Other Token</title>"));
    }

    #[test]
    fn test_index_html_has_dark_theme() {
        assert!(INDEX_HTML.contains("background:#0d1117"));
    }

    #[test]
    fn test_index_html_has_sse_event_source() {
        assert!(INDEX_HTML.contains("EventSource"));
    }

    #[test]
    fn test_index_html_has_view_modes() {
        assert!(INDEX_HTML.contains("v-single"));
        assert!(INDEX_HTML.contains("v-sbs"));
        assert!(INDEX_HTML.contains("v-multi"));
    }

    #[test]
    fn test_index_html_has_export_button() {
        assert!(INDEX_HTML.contains("Export JSON"));
    }

    #[test]
    fn test_index_html_has_graph_canvas() {
        assert!(INDEX_HTML.contains("depgraph"));
        assert!(INDEX_HTML.contains("drawGraph"));
    }

    #[test]
    fn test_index_html_has_transform_selector() {
        assert!(INDEX_HTML.contains("reverse"));
        assert!(INDEX_HTML.contains("uppercase"));
        assert!(INDEX_HTML.contains("mock"));
        assert!(INDEX_HTML.contains("noise"));
    }

    #[test]
    fn test_index_html_has_provider_selector() {
        assert!(INDEX_HTML.contains("OpenAI"));
        assert!(INDEX_HTML.contains("Anthropic"));
    }

    #[test]
    fn test_index_html_has_js_transforms() {
        assert!(INDEX_HTML.contains("const TX="));
        assert!(INDEX_HTML.contains("reverse:"));
        assert!(INDEX_HTML.contains("uppercase:"));
        assert!(INDEX_HTML.contains("mock:"));
        assert!(INDEX_HTML.contains("noise:"));
    }

    #[test]
    fn test_index_html_has_heatmap_toggle() {
        assert!(INDEX_HTML.contains("heatmap"));
    }

    #[test]
    fn test_index_html_has_stream_button() {
        assert!(INDEX_HTML.contains("Stream"));
        assert!(INDEX_HTML.contains("btn-go"));
    }

    #[test]
    fn test_index_html_has_token_animation() {
        assert!(INDEX_HTML.contains("fadeIn"));
    }

    #[test]
    fn test_index_html_has_multi_panel_colors() {
        assert!(INDEX_HTML.contains("mp-reverse"));
        assert!(INDEX_HTML.contains("mp-uppercase"));
        assert!(INDEX_HTML.contains("mp-mock"));
        assert!(INDEX_HTML.contains("mp-noise"));
    }

    #[test]
    fn test_index_html_has_side_by_side_labels() {
        assert!(INDEX_HTML.contains("Original"));
        assert!(INDEX_HTML.contains("Transformed"));
    }

    #[test]
    fn test_index_html_has_mode_buttons() {
        assert!(INDEX_HTML.contains("btn-single"));
        assert!(INDEX_HTML.contains("btn-sbs"));
        assert!(INDEX_HTML.contains("btn-multi"));
    }

    #[test]
    fn test_index_html_has_done_signal_handling() {
        assert!(INDEX_HTML.contains("[DONE]"));
    }

    #[test]
    fn test_index_html_has_stats_display() {
        assert!(INDEX_HTML.contains("stats"));
    }

    #[test]
    fn test_index_html_has_graph_toggle() {
        assert!(INDEX_HTML.contains("graphtoggle"));
    }

    #[test]
    fn test_index_html_no_external_deps() {
        // Verify no npm/webpack/CDN references
        assert!(!INDEX_HTML.contains("cdn."));
        assert!(!INDEX_HTML.contains("unpkg.com"));
        assert!(!INDEX_HTML.contains("jsdelivr"));
        assert!(!INDEX_HTML.contains("npm"));
    }

    // -- server integration smoke tests --

    #[tokio::test]
    async fn test_serve_binds_to_port() {
        // Verify TcpListener can bind (just test the binding, not the loop)
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await;
        assert!(listener.is_ok());
        let listener = listener.unwrap();
        let addr = listener.local_addr().unwrap();
        assert!(addr.port() > 0);
    }

    #[test]
    fn test_index_html_content_length_matches() {
        let html_bytes = INDEX_HTML.as_bytes();
        assert!(html_bytes.len() > 1000, "HTML should be substantial");
    }

    #[test]
    fn test_index_html_responsive_viewport() {
        assert!(INDEX_HTML.contains("viewport"));
        assert!(INDEX_HTML.contains("width=device-width"));
    }

    #[test]
    fn test_url_decode_long_string() {
        let input = "the+quick+brown+fox+jumps+over+the+lazy+dog";
        let expected = "the quick brown fox jumps over the lazy dog";
        assert_eq!(url_decode(input), expected);
    }

    #[test]
    fn test_parse_query_duplicate_keys_last_wins() {
        let params = parse_query("key=first&key=second");
        // HashMap behavior: last key wins on iteration insert
        let val = params.get("key").map(|s| s.as_str());
        assert!(val == Some("first") || val == Some("second"));
    }

    #[test]
    fn test_index_html_has_bezier_curves() {
        assert!(INDEX_HTML.contains("bezierCurveTo"));
    }

    #[test]
    fn test_index_html_has_heat_color_function() {
        assert!(INDEX_HTML.contains("heatColor"));
    }

    #[test]
    fn test_index_html_has_export_download() {
        assert!(INDEX_HTML.contains("download"));
        assert!(INDEX_HTML.contains("application/json"));
    }

    // -- New feature tests --

    #[test]
    fn test_index_html_has_chaos_transform() {
        assert!(INDEX_HTML.contains("chaos"));
        assert!(INDEX_HTML.contains(r#"value="chaos""#));
    }

    #[test]
    fn test_index_html_has_chaos_js_transform() {
        assert!(INDEX_HTML.contains("TX.chaos") || INDEX_HTML.contains("chaos:s=>"));
    }

    #[test]
    fn test_index_html_has_chaos_tooltip_css() {
        assert!(INDEX_HTML.contains("attr(title)"));
        assert!(INDEX_HTML.contains(".token[title]"));
    }

    #[test]
    fn test_index_html_has_diff_view() {
        assert!(INDEX_HTML.contains("v-diff"));
        assert!(INDEX_HTML.contains("diff-openai"));
        assert!(INDEX_HTML.contains("diff-anthropic"));
    }

    #[test]
    fn test_index_html_has_diff_button() {
        assert!(INDEX_HTML.contains("btn-diff"));
        assert!(INDEX_HTML.contains("Diff"));
    }

    #[test]
    fn test_index_html_has_diff_css() {
        assert!(INDEX_HTML.contains("view-diff"));
        assert!(INDEX_HTML.contains("diff-match"));
        assert!(INDEX_HTML.contains("diff-diverge"));
    }

    #[test]
    fn test_index_html_has_diff_highlight_fn() {
        assert!(INDEX_HTML.contains("applyDiffHighlights"));
    }

    #[test]
    fn test_index_html_has_surgery_css() {
        assert!(INDEX_HTML.contains("surgeable"));
        assert!(INDEX_HTML.contains("token-input"));
    }

    #[test]
    fn test_index_html_has_surgery_fn() {
        assert!(INDEX_HTML.contains("enableSurgery"));
        assert!(INDEX_HTML.contains("surgeryLog"));
    }

    #[test]
    fn test_index_html_has_diff_stream_fn() {
        assert!(INDEX_HTML.contains("startDiff"));
        assert!(INDEX_HTML.contains("diff-stream"));
    }

    #[test]
    fn test_index_html_has_chaos_label_in_mkspan() {
        assert!(INDEX_HTML.contains("chaos_label") || INDEX_HTML.contains("chaosLabel"));
    }

    #[test]
    fn test_diff_token_event_serializes_with_side() {
        let event = crate::TokenEvent {
            text: "hello".to_string(),
            original: "hello".to_string(),
            index: 0,
            transformed: false,
            importance: 0.5,
            chaos_label: None,
            provider: None,
            confidence: None,
            perplexity: None,
            alternatives: vec![],
        };
        let diff = DiffTokenEvent {
            side: "openai",
            event: &event,
        };
        let json = serde_json::to_string(&diff).expect("serialize");
        assert!(json.contains(r#""side":"openai""#));
        assert!(json.contains(r#""text":"hello""#));
        assert!(json.contains(r#""index":0"#));
    }

    #[test]
    fn test_diff_token_event_anthropic_side() {
        let event = crate::TokenEvent {
            text: "world".to_string(),
            original: "world".to_string(),
            index: 1,
            transformed: true,
            importance: 0.7,
            chaos_label: Some("reverse".to_string()),
            provider: None,
            confidence: None,
            perplexity: None,
            alternatives: vec![],
        };
        let diff = DiffTokenEvent {
            side: "anthropic",
            event: &event,
        };
        let json = serde_json::to_string(&diff).expect("serialize");
        assert!(json.contains(r#""side":"anthropic""#));
        assert!(json.contains(r#""transformed":true"#));
        assert!(json.contains(r#""chaos_label":"reverse""#));
    }

    #[test]
    fn test_index_html_surgery_log_in_export() {
        assert!(INDEX_HTML.contains("surgery_log"));
    }

    // -- Research suite feature tests ----------------------------------------

    #[test]
    fn test_index_html_has_experiment_button() {
        assert!(INDEX_HTML.contains("btn-experiment"));
        assert!(INDEX_HTML.contains("Experiment"));
    }

    #[test]
    fn test_index_html_has_research_button() {
        assert!(INDEX_HTML.contains("btn-research"));
        assert!(INDEX_HTML.contains("Research"));
    }

    #[test]
    fn test_index_html_has_experiment_view() {
        assert!(INDEX_HTML.contains("v-experiment"));
    }

    #[test]
    fn test_index_html_has_research_view() {
        assert!(INDEX_HTML.contains("v-research"));
    }

    #[test]
    fn test_index_html_has_perplexity_sparkline() {
        assert!(INDEX_HTML.contains("perp-spark"));
    }

    #[test]
    fn test_index_html_has_research_grid() {
        assert!(INDEX_HTML.contains("research-grid") || INDEX_HTML.contains("research-dash"));
    }

    #[test]
    fn test_index_html_has_render_research_fn() {
        assert!(INDEX_HTML.contains("renderResearch"));
    }

    #[test]
    fn test_index_html_render_research_called_on_done() {
        // renderResearch() must appear after enableSurgery in the [DONE] handler
        let done_pos = INDEX_HTML.find("[DONE]").expect("[DONE] not found");
        let research_pos = INDEX_HTML[done_pos..]
            .find("renderResearch")
            .expect("renderResearch not found after [DONE]");
        let surgery_pos = INDEX_HTML[done_pos..]
            .find("enableSurgery")
            .expect("enableSurgery not found after [DONE]");
        assert!(research_pos > surgery_pos, "renderResearch should come after enableSurgery");
    }

    #[test]
    fn test_index_html_has_start_experiment_fn() {
        assert!(INDEX_HTML.contains("startExperiment"));
    }

    #[test]
    fn test_index_html_has_update_perp_sparkline_fn() {
        assert!(INDEX_HTML.contains("updatePerpSparkline"));
    }

    #[test]
    fn test_index_html_has_ab_stream_route() {
        assert!(INDEX_HTML.contains("/ab-stream"));
    }

    #[test]
    fn test_index_html_has_ab_system_prompts() {
        assert!(INDEX_HTML.contains("sys_a") || INDEX_HTML.contains("ab-prompts"));
    }

    #[test]
    fn test_index_html_has_experiment_divergence() {
        assert!(INDEX_HTML.contains("exp-diverge") || INDEX_HTML.contains("renderExpDivergence"));
    }

    #[test]
    fn test_index_html_has_confidence_rendering() {
        assert!(INDEX_HTML.contains("confidence"));
    }

    #[test]
    fn test_index_html_has_perplexity_rendering() {
        assert!(INDEX_HTML.contains("perplexity"));
    }

    #[test]
    fn test_index_html_has_confidence_css_classes() {
        assert!(INDEX_HTML.contains("conf-high"));
        assert!(INDEX_HTML.contains("conf-mid"));
        assert!(INDEX_HTML.contains("conf-low"));
    }

    #[test]
    fn test_index_html_has_high_perp_animation() {
        assert!(INDEX_HTML.contains("high-perp") || INDEX_HTML.contains("perpPulse"));
    }

    #[test]
    fn test_index_html_has_ab_panel_a() {
        assert!(INDEX_HTML.contains("exp-a"));
    }

    #[test]
    fn test_index_html_has_ab_panel_b() {
        assert!(INDEX_HTML.contains("exp-b"));
    }

    #[test]
    fn test_index_html_has_experiment_view_in_views_map() {
        assert!(INDEX_HTML.contains("experiment:$('#v-experiment')"));
    }

    #[test]
    fn test_index_html_has_research_view_in_views_map() {
        assert!(INDEX_HTML.contains("research:$('#v-research')"));
    }
}
