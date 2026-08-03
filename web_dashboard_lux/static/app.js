const VERSION_POLL_MS = 60000;
let data = window.__INITIAL__ || {};
let currentDataVersion = localStorage.getItem('fl_data_version') || '';
let latestVersionPayload = null;
let predictionFilter = localStorage.getItem('fl_filter') || 'all';
let predictionSort = localStorage.getItem('fl_sort') || 'default';
const PREDICTIONS_PAGE_SIZE = 10;
let predictionPage = 1;
let failCount = 0;
const $ = (id) => document.getElementById(id);
const money = (v) => Number(v || 0).toLocaleString('pt-BR',{style:'currency',currency:'BRL'});
const pct = (v) => `${(Number(v || 0)*100).toFixed(1)}%`;
const esc = (s='') => String(s).replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const clsNum = (v) => Number(v || 0) >= 0 ? 'positive' : 'negative';
const buzz = (ms=8) => { try{ navigator.vibrate && navigator.vibrate(ms); }catch(e){} };

const MESES = ['Janeiro','Fevereiro','Março','Abril','Maio','Junho','Julho','Agosto','Setembro','Outubro','Novembro','Dezembro'];

/* ---------- Tema (claro/escuro) ---------- */
function applyTheme(theme){
  document.documentElement.setAttribute('data-theme', theme);
  const meta = document.querySelector('meta[name="theme-color"]');
  if(meta) meta.setAttribute('content', theme==='dark' ? '#17130F' : '#FFF6EA');
  const btn = $('themeToggle');
  if(btn) btn.textContent = theme==='dark' ? '☀️' : '🌙';
}
(function initTheme(){
  const current = document.documentElement.getAttribute('data-theme') || 'light';
  applyTheme(current);
  $('themeToggle')?.addEventListener('click', ()=>{
    const next = document.documentElement.getAttribute('data-theme')==='dark' ? 'light' : 'dark';
    localStorage.setItem('fl_theme', next);
    applyTheme(next);
    buzz();
  });
})();

/* ---------- Resumo ao abrir o app ---------- */
function showDailyBrief(d){
  const el = $('dayBrief'); if(!el || !d) return;
  const today = new Date().toISOString().slice(0,10);
  if(localStorage.getItem('fl_brief_date') === today) return;
  const hora = new Date().getHours();
  const saudacao = hora < 12 ? 'Bom dia' : hora < 18 ? 'Boa tarde' : 'Boa noite';
  const sinais = d.previsoes?.sinais || 0;
  const saldo = money(d.historico?.saldo || 0);
  const roi = pct(d.historico?.roi || 0);
  const jogos = d.previsoes?.jogos || 0;
  $('dayBriefText').textContent = sinais > 0
    ? `☀️ ${saudacao}! ${sinais} indicação(ões) oficial(is) hoje em ${jogos} jogos · saldo ${saldo} · retorno ${roi}`
    : `☀️ ${saudacao}! Sem indicações oficiais no momento · ${jogos} jogos analisados · saldo ${saldo} · retorno ${roi}`;
  el.classList.add('show');
  localStorage.setItem('fl_brief_date', today);
}
$('dayBriefClose')?.addEventListener('click', ()=> $('dayBrief').classList.remove('show'));

/* ---------- Alerta de novo sinal ---------- */
let alertsEnabled = localStorage.getItem('fl_alerts_enabled') !== '0';
let firstSignalPass = true;
let knownSignalKeys = new Set();
try{ knownSignalKeys = new Set(JSON.parse(localStorage.getItem('fl_known_signals')||'[]')); }catch(e){}

function signalKey(p){ return `${p.jogo}||${p.mercado}||${p.data}`; }

function playAlertBeep(){
  try{
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if(!Ctx) return;
    const ctx = new Ctx();
    const o = ctx.createOscillator(), g = ctx.createGain();
    o.type = 'sine'; o.frequency.value = 880;
    g.gain.setValueAtTime(0.0001, ctx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.18, ctx.currentTime + 0.02);
    g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.32);
    o.connect(g); g.connect(ctx.destination);
    o.start(); o.stop(ctx.currentTime + 0.34);
    setTimeout(()=>ctx.close(), 500);
  }catch(e){}
}

function checkNewSignals(topList){
  const list = topList || [];
  const currentKeys = list.map(signalKey);
  if(firstSignalPass){
    // primeira carga: só registra o que já existe, sem alertar
    knownSignalKeys = new Set(currentKeys);
    localStorage.setItem('fl_known_signals', JSON.stringify([...knownSignalKeys].slice(-300)));
    firstSignalPass = false;
    return;
  }
  const novos = list.filter(p => !knownSignalKeys.has(signalKey(p)));
  if(novos.length){
    novos.forEach(p => knownSignalKeys.add(signalKey(p)));
    localStorage.setItem('fl_known_signals', JSON.stringify([...knownSignalKeys].slice(-300)));
    if(alertsEnabled){
      buzz(60);
      playAlertBeep();
      const p = novos[0];
      const evTxt = p.ev ? `vantagem estimada ${(Number(p.ev)*100).toFixed(1)}%` : '';
      toast(novos.length>1 ? `${novos.length} novos sinais com boa vantagem estimada` : `Novo sinal: ${p.jogo} · ${evTxt}`);
      $('alertToggle')?.classList.add('has-new');
      setTimeout(()=>$('alertToggle')?.classList.remove('has-new'), 6000);
    }
  }
}


function initials(name=''){
  const parts=String(name).trim().split(/\s+/).filter(Boolean);
  return (parts.slice(0,2).map(x=>x[0]).join('')||'?').toUpperCase();
}
function crest(url,name,kind='team'){
  const fallbackText=esc(initials(name));
  if(!url) return `<span class="crest-fallback ${kind}">${fallbackText}</span>`;
  return `<span class="crest-wrap ${kind}"><img src="${esc(url)}" alt="" loading="lazy" onerror="this.style.display='none';this.nextElementSibling.style.display='grid'"><span class="crest-fallback ${kind} crest-inline-fallback" style="display:none">${fallbackText}</span></span>`;
}

const MARKET_NAMES = {
  R_FT_H:'Vitória do mandante', R_FT_D:'Empate no jogo', R_FT_A:'Vitória do visitante',
  R_HT_H:'Mandante vence o 1º tempo', R_HT_D:'Empate no 1º tempo', R_HT_A:'Visitante vence o 1º tempo',
  BTTS_Y:'Ambos marcam — Sim', BTTS_N:'Ambos marcam — Não',
  O_BTTS_Y:'Ambos marcam — Sim', O_BTTS_N:'Ambos marcam — Não',
  DC_1X:'Dupla chance — Mandante ou empate', DC_12:'Dupla chance — Mandante ou visitante', DC_X2:'Dupla chance — Empate ou visitante'
};

function decimalLine(code=''){
  const raw=String(code).replace(/[^0-9]/g,'');
  if(!raw) return '';
  if(raw.length===1) return raw;
  return `${parseInt(raw.slice(0,-1),10)},${raw.slice(-1)}`;
}
function signedLine(sign,raw){
  const line=decimalLine(raw);
  if(!line) return '';
  return `${sign==='NEG'?'-':'+'}${line}`;
}
function marketName(code=''){
  const c=String(code||'').trim().toUpperCase();
  if(!c) return 'Mercado não informado';
  if(MARKET_NAMES[c]) return MARKET_NAMES[c];
  let m;
  if((m=c.match(/^TG_(FT|HT)_([OU])(\d+)$/))){
    const periodo=m[1]==='HT'?'no 1º tempo':'no jogo';
    return `${m[2]==='O'?'Mais':'Menos'} de ${decimalLine(m[3])} gols ${periodo}`;
  }
  if((m=c.match(/^G_([HA])_(FT|HT)_([OU])(\d+)$/))){
    const equipe=m[1]==='H'?'Mandante':'Visitante';
    const periodo=m[2]==='HT'?' no 1º tempo':'';
    return `${equipe} marca ${m[3]==='O'?'mais':'menos'} de ${decimalLine(m[4])} gols${periodo}`;
  }
  if((m=c.match(/^AH_(HOME|AWAY)_(NEG|POS)_(\d+)$/))){
    return `${m[1]==='HOME'?'Mandante':'Visitante'} ${signedLine(m[2],m[3])} — Handicap Asiático`;
  }
  if((m=c.match(/^EH_(HOME|AWAY|DRAW)_(NEG|POS)_(\d+)$/))){
    const lado={HOME:'Mandante',AWAY:'Visitante',DRAW:'Empate'}[m[1]];
    return `${lado} ${signedLine(m[2],m[3])} gols — Handicap Europeu`;
  }
  if((m=c.match(/^CS_(\d+)_(\d+)$/))) return `Placar exato — ${m[1]} x ${m[2]}`;
  if((m=c.match(/^(?:CORNERS?|CK|TG_CORNERS?)_(FT|HT)?_?([OU])_?(\d+)$/))){
    const periodo=m[1]==='HT'?' no 1º tempo':' no jogo';
    return `${m[2]==='O'?'Mais':'Menos'} de ${decimalLine(m[3])} escanteios${periodo}`;
  }
  if(c.includes('CORN') || c.includes('CORNER')) return c.replaceAll('_',' ').replace(/O(\d+)/g,(_,x)=>`Mais de ${decimalLine(x)}`).replace(/U(\d+)/g,(_,x)=>`Menos de ${decimalLine(x)}`);
  return c.toLowerCase().replaceAll('_',' ').replace(/\w/g,x=>x.toUpperCase());
}

const STATUS_NAMES = {
  ATIVA:'Mercado ativo', ATIVO:'Mercado ativo', APPROVED:'Modelo aprovado', APROVADO:'Modelo aprovado',
  BLOQUEADA:'Mercado bloqueado', BLOQUEADO:'Mercado bloqueado', OBSERVACAO:'Em observação',
  APOSENTADA:'Mercado desativado', CALIBRADOR_NAO_ENCONTRADO:'Calibração pendente',
  SEM_CALIBRADOR:'Calibração pendente', TESTE_REAL:'Modo de teste', DESCONHECIDA:'Situação em análise',
  COLETANDO_DADOS:'Coletando dados', AGUARDANDO:'Aguardando', SEM_AMOSTRA:'Sem amostra suficiente',
  SEM_HEARTBEAT:'Sem sinal da automação', EXECUTANDO:'Executando', ATRASADA:'Automação atrasada',
  PAPER_ONLY:'Somente simulação', OK:'Tudo certo'
};
function humanStatus(raw=''){
  const key=String(raw||'').trim().toUpperCase();
  return STATUS_NAMES[key] || (key ? key.toLowerCase().replaceAll('_',' ').replace(/\w/g,x=>x.toUpperCase()) : 'Em análise');
}
function displayMode(raw=''){
  const key=String(raw||'').trim().toUpperCase();
  return {PAPER_ONLY:'Somente simulação',MONITOR:'Monitoramento',RESEARCH:'Pesquisa'}[key] || humanStatus(key);
}
const TASK_LABELS = {
  fetch_today:'Atualizar jogos', predict_today:'Gerar previsões', import_bank:'Registrar simulação',
  settle:'Atualizar resultados', settle_flashscore_preview:'Consultar resultados',
  settle_flashscore_apply:'Aplicar resultados encontrados', scores_context:'Avaliar contextos',
  profile_markets:'Analisar mercados', rebuild_bank:'Recalcular banca',
  model_registry:'Atualizar catálogo de modelos', full_today:'Rotina completa',
  update_bank:'Atualizar simulação'
};
const STEP_LABELS = {
  '01_fetch_futpython_daily.py':'Atualizando jogos do dia',
  'paper_predict.py':'Gerando previsões da simulação',
  '06_registrar_paper.py':'Registrando previsões na banca simulada',
  '05_settle_historico.py':'Atualizando resultados encerrados',
  '05_settle_flashscore.py':'Consultando resultados no Flashscore',
  '04_banca.py':'Recalculando a banca simulada',
  '09_gerar_perfil_operacional_mercados.py':'Analisando mercados',
  '10_gerar_scores_contexto.py':'Avaliando contextos',
  '14_model_registry.py':'Atualizando o catálogo de modelos',
  iniciando:'Iniciando', finalizado:'Finalizado', erro:'Erro'
};
function taskLabel(raw=''){ return TASK_LABELS[String(raw||'')] || 'Rotina do sistema'; }
function friendlyStep(raw=''){ return STEP_LABELS[String(raw||'')] || String(raw||'Processando'); }
function humanReasonToken(token=''){
  const t=String(token).trim(); if(!t) return '';
  const low=t.toLowerCase();
  const fixed=[
    [/^odd_fora_da_faixa/, 'A odd está fora da faixa recomendada para este mercado.'],
    [/^odd_(real_)?ausente|sem_odd/, 'A odd real não estava disponível para esta análise.'],
    [/roi_backtest_negativo/, 'O histórico global deste mercado ainda está negativo no backtest.'],
    [/confianca_colada_no_threshold/, 'A confiança ficou muito próxima do limite mínimo de aprovação.'],
    [/confianca_abaixo|probabilidade_abaixo/, 'A confiança do modelo ficou abaixo do mínimo exigido.'],
    [/auc_abaixo/, 'A qualidade estatística do modelo ficou abaixo do mínimo exigido.'],
    [/calibrador_nao_encontrado|sem_calibrador/, 'O modelo ainda não possui calibração validada.'],
    [/sem_contexto_positivo|contexto_positivo_nao_encontrado/, 'Nenhum contexto positivo forte foi encontrado para esta entrada.'],
    [/contexto_bloqueado|context_block/, 'O contexto deste jogo não passou pelos filtros operacionais.'],
    [/mercado_bloqueado/, 'Este mercado está bloqueado pela governança operacional.'],
    [/mercado_em_observacao|observacao/, 'Este mercado está em observação e ainda não foi liberado.'],
    [/limite_1_aposta_por_jogo/, 'Outra oportunidade mais forte já foi escolhida para este jogo.'],
    [/ev_abaixo|min_ev/, 'O valor esperado ficou abaixo do mínimo exigido.'],
    [/sem_historico|amostra_insuficiente/, 'Ainda não há histórico suficiente para validar este contexto.']
  ];
  for(const [rx,msg] of fixed) if(rx.test(low)) return msg;
  let m=low.match(/^(liga_)?(home|away)_ruim_roi_(-?\d+(?:\.\d+)?)%?(?:_bets_(\d+))?/);
  if(m){
    const origem=m[1]?'A combinação entre liga e ':'O ';
    const lado=m[2]==='home'?'mandante':'visitante';
    const roi=Number(m[3]).toLocaleString('pt-BR',{minimumFractionDigits:1,maximumFractionDigits:1});
    const amostra=m[4]?` em ${m[4]} análises`:'';
    return `${origem}${lado} apresenta desempenho desfavorável neste mercado (ROI ${roi}%${amostra}).`;
  }
  m=low.match(/^(liga_)?(home|away)_bom_roi_(-?\d+(?:\.\d+)?)%?(?:_bets_(\d+))?/);
  if(m){
    const origem=m[1]?'A combinação entre liga e ':'O ';
    const lado=m[2]==='home'?'mandante':'visitante';
    const roi=Number(m[3]).toLocaleString('pt-BR',{minimumFractionDigits:1,maximumFractionDigits:1});
    const amostra=m[4]?` em ${m[4]} análises`:'';
    return `${origem}${lado} apresenta desempenho favorável neste mercado (ROI +${roi}%${amostra}).`;
  }
  return t.replaceAll('_',' ').replace(/\w/g,x=>x.toUpperCase()) + '.';
}
function humanDiagnostics(...values){
  const tokens=values.filter(Boolean).flatMap(v=>String(v).split(/[;|]+/)).map(x=>x.trim()).filter(Boolean);
  const out=[];
  for(const token of tokens){ const msg=humanReasonToken(token); if(msg && !out.includes(msg)) out.push(msg); }
  return out.length?out:['Nenhum impedimento adicional foi identificado nesta análise.'];
}
function diagnosticHtml(p){
  return `<ul class="diagnostic-list">${humanDiagnostics(p.motivo,p.contexto,p.status).map(x=>`<li>${esc(x)}</li>`).join('')}</ul>`;
}

function matchResultHtml(p){
  const hasScore = p.home_score !== '' && p.away_score !== '';
  const status = p.match_status ? humanStatus(p.match_status) : '';
  if(!hasScore && !status) return '';
  return `<div class="match-result-summary">
    ${hasScore ? `<strong>${esc(p.home_score)} × ${esc(p.away_score)}</strong>` : ''}
    ${status ? `<span>${esc(status)}</span>` : ''}
  </div>`;
}

function matchStatsButton(p){
  if(!p.stats_url) return '';
  return `<a class="match-stats-link" href="${esc(p.stats_url)}" target="_blank" rel="noopener noreferrer">
    <span>▣</span><b>Ver placar e estatísticas</b><small>${esc(p.stats_source||'Abre em nova aba')}${p.flashscore_id?` · ID ${esc(p.flashscore_id)}`:''}</small>
  </a>`;
}

function toast(msg){ const t=$('toast'); t.textContent=msg; t.classList.add('show'); clearTimeout(window.__toast); window.__toast=setTimeout(()=>t.classList.remove('show'),2500); }

/* ---------- datas ---------- */
function normDate(raw){
  if(!raw) return null;
  const s=String(raw).trim();
  let m=s.match(/^(\d{4})-(\d{2})-(\d{2})/); if(m) return `${m[1]}-${m[2]}-${m[3]}`;
  m=s.match(/^(\d{2})\/(\d{2})\/(\d{4})/); if(m) return `${m[3]}-${m[2]}-${m[1]}`;
  m=s.match(/^(\d{2})-(\d{2})-(\d{4})/); if(m) return `${m[3]}-${m[2]}-${m[1]}`;
  return null;
}
function isoToBR(iso){ if(!iso) return '-'; const [y,m,d]=iso.split('-'); return `${d}/${m}/${y}`; }
function pad2(n){ return String(n).padStart(2,'0'); }

/* ---------- gráfico ---------- */
function renderChart(id, points){
  const svg=$(id); if(!svg) return;
  if(!points || points.length < 2){svg.innerHTML='<text x="20" y="120" class="chart-label">Sem dados suficientes</text>';return;}
  const W=720,H=id==='bankChartFull'?280:240,pad=24;
  const ys=points.map(p=>Number(p.y||0)); const min=Math.min(...ys), max=Math.max(...ys); const span=Math.max(max-min,1);
  const coords=points.map((p,i)=>({x:pad+(i/(points.length-1))*(W-pad*2),y:H-pad-((Number(p.y)-min)/span)*(H-pad*2)}));
  const line=coords.map((p,i)=>`${i?'L':'M'}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
  const area=`M${coords[0].x},${H-pad} `+coords.map((p)=>`L${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')+` L${coords.at(-1).x},${H-pad} Z`;
  const grid=[.25,.5,.75].map(f=>`<line class="chart-gridline" x1="${pad}" y1="${(H-pad)-(H-pad*2)*f}" x2="${W-pad}" y2="${(H-pad)-(H-pad*2)*f}"/>`).join('');
  const last=coords.at(-1);
  svg.innerHTML=`<defs><linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#29A866" stop-opacity=".28"/><stop offset="1" stop-color="#29A866" stop-opacity="0"/></linearGradient></defs>${grid}<path class="chart-area" d="${area}"/><path class="chart-line" d="${line}"/><circle class="chart-dot" cx="${last.x}" cy="${last.y}" r="4"/><text x="${pad}" y="${H-3}" class="chart-label">${money(min)}</text><text x="${W-145}" y="20" class="chart-label">${money(max)}</text>`;
}

/* ---------- Spotlight: sinal em destaque (estilo pôster) ---------- */
function renderSpotlight(){
  const el = $('spotlight'); if(!el) return;
  if(!data.previsoes?.oficial){
    el.style.display='';
    el.innerHTML=`<div class="market-card source-empty"><b>Nenhuma aposta aprovada neste arquivo</b><small>As análises antigas continuam guardadas para auditoria, mas não são exibidas como apostas. O ciclo paper atual só aparecerá aqui quando aprovar uma entrada dentro da janela operacional.</small></div>`;
    return;
  }
  let pool = (data.previsoes?.top?.length ? data.previsoes.top : (data.previsoes?.linhas||[]));
  if(dateFilterJogos) pool = pool.filter(p=>normDate(p.data)===dateFilterJogos);
  if(!pool.length){
    el.style.display='';
    el.innerHTML=`<div class="market-card">Nenhuma previsão disponível para ${dateFilterJogos?isoToBR(dateFilterJogos):'a data selecionada'}.</div>`;
    return;
  }
  el.style.display='';
  const p = pool.slice().sort((a,b)=>(Number(b.ev)||0)-(Number(a.ev)||0))[0];

  const modelProb = p.prob ? Number(p.prob)*100 : null;
  const impliedProb = p.odd ? (100/Number(p.odd)) : null;
  const edge = (modelProb!=null && impliedProb!=null) ? (modelProb-impliedProb) : null;
  const modelPct = Math.max(2, Math.min(96, modelProb ?? 50));
  const marketPct = Math.max(0, Math.min(96-modelPct, impliedProb ?? 0));

  const bgWords = `${(p.home||'GOL').toUpperCase()} ${(p.away||'BOLA').toUpperCase()} `.repeat(4);

  el.innerHTML = `
    <div class="spotlight-bgtext">${esc(bgWords)}<br>${esc(bgWords)}</div>
    <div class="spotlight-top">
      <span class="spotlight-league">${crest(p.league_crest,p.liga,'league')}<span>${esc(p.liga||'Liga')}</span></span>
      <span class="spotlight-chip">${esc(marketName(p.mercado))}</span>
    </div>
    <div class="spotlight-kicker">★★★★★ TOP PICK</div><h1 class="spotlight-title">MELHOR SINAL <span class="hl">DO DIA</span></h1>
    <div class="spotlight-bar">
      <div class="spotlight-track">
        <i class="seg-model" style="width:${modelPct}%"></i><i class="seg-gap" style="width:${Math.max(0,96-modelPct-marketPct)}%"></i><i class="seg-market" style="width:${marketPct}%"></i>
      </div>
      <div class="spotlight-nums">
        <div><b>${modelProb!=null?modelProb.toFixed(0)+'%':'-'}</b><span>MODELO</span></div>
        <div><b>${edge!=null?(edge>=0?'+':'')+edge.toFixed(1)+'%':'-'}</b><span>EDGE</span></div>
        <div><b>${impliedProb!=null?impliedProb.toFixed(0)+'%':'-'}</b><span>MERCADO</span></div>
      </div>
    </div>
    <div class="spotlight-card">
      <div class="sc-row"><span class="sc-comp">${crest(p.league_crest,p.liga,'league')} ${esc(p.liga||'')}</span><span class="sc-date">${esc(p.data||'Hoje')}</span></div>
      <div class="sc-mid"><b>${esc(p.odd||'-')}</b><span>COTAÇÃO</span></div><div class="sc-value-row"><span>${esc(marketName(p.mercado))}</span><b class="${clsNum(p.ev)}">Vantagem ${p.ev?`${(Number(p.ev)*100).toFixed(1)}%`:'-'}</b></div>
      <div class="sc-teams">
        <div class="sc-team"><span class="crest-ring">${crest(p.home_crest,p.home)}</span><b>${esc(p.home||'-')}</b></div>
        <span class="sc-vs">vs</span>
        <div class="sc-team"><span class="crest-ring">${crest(p.away_crest,p.away)}</span><b>${esc(p.away||'-')}</b></div>
      </div>
      <button class="sc-cta ${p.sinal==='Sim'?'':'muted'}" onclick="document.querySelector('[data-tab=jogos]').click();const s=document.getElementById('searchPredictions');s.value='${esc(p.home||'')}';s.dispatchEvent(new Event('input'));">Ver análise completa</button>
      ${p.stats_url?`<a class="spotlight-stats" href="${esc(p.stats_url)}" target="_blank" rel="noopener noreferrer">${p.stats_source==='Flashscore'?'Abrir no Flashscore':'Ver placar e estatísticas'} ↗</a>`:''}
    </div>`;
}


function reasonChips(p){
  const raw = `${p.contexto||''};${p.motivo||''};${p.status||''}`.toLowerCase();
  const items=[];
  const add=(label,ok=true)=>{ if(!items.some(x=>x.label===label)) items.push({label,ok}); };
  if(raw.includes('liga')) add('Liga analisada');
  if(raw.includes('home')||raw.includes('mandante')) add('Mandante analisado');
  if(raw.includes('away')||raw.includes('visitante')) add('Visitante analisado');
  if(raw.includes('context')||raw.includes('especialista')) add('Contexto aplicado');
  if(p.auc && Number(p.auc)>=0.58) add(`Qualidade (AUC) ${Number(p.auc).toFixed(3)}`);
  if(p.ev && Number(p.ev)>0) add('Vantagem estimada positiva');
  if(p.sinal==='Sim') add('Entrada aprovada');
  if(!items.length) add('Análise global', false);
  return items.slice(0,5).map(x=>`<span class="reason-chip ${x.ok?'ok':'neutral'}">${x.ok?'✓':'•'} ${esc(x.label)}</span>`).join('');
}

/* ---------- previsões ---------- */
function predictionCard(p){
  const leagueLogo=crest(p.league_crest,p.liga,'league');
  const isSignal = p.indicacao_oficial===true;
  const isLegacy = p.origem_previsao==='legado_pre_ciclo';
  const modelProb = p.prob ? Number(p.prob)*100 : null;
  const impliedProb = p.odd ? (100/Number(p.odd)) : null;
  const edge = (modelProb!=null && impliedProb!=null) ? (modelProb-impliedProb) : null;
  const fillPct = Math.max(4, Math.min(100, modelProb ?? 0));
  const markPct = impliedProb!=null ? Math.max(0, Math.min(100, impliedProb)) : null;
  return `<details class="ticket ${isSignal?'signal-ticket':''}"><summary>
    <div class="match-top">
      <div class="league-line">${leagueLogo}<span>${esc(p.liga)} · ${esc(marketName(p.mercado))}</span></div>
      <span class="signal-pill ${isSignal?'yes':''}">${isSignal?'⚡ INDICAÇÃO OFICIAL':(isLegacy?'PRÉ-CICLO · CONSULTA':'ANÁLISE')}</span>
    </div>
    <div class="face-off">
      <div class="fo-side home"><span class="crest-ring">${crest(p.home_crest,p.home)}</span><b>${esc(p.home||'-')}</b></div>
      <span class="fo-vs">VS</span>
      <div class="fo-side away"><span class="crest-ring">${crest(p.away_crest,p.away)}</span><b>${esc(p.away||'-')}</b></div>
    </div>
    <div class="confidence">
      <div class="confidence-track">
        <i class="confidence-fill" style="width:${fillPct}%"></i>
        ${markPct!=null?`<i class="confidence-mark" style="left:${markPct}%" title="Probabilidade implícita da odd"></i>`:''}
      </div>
      <div class="confidence-labels"><span>Modelo <b>${modelProb!=null?modelProb.toFixed(1)+'%':'-'}</b></span><span class="${edge!=null?clsNum(edge):''}">${edge!=null?`Vantagem ${edge>=0?'+':''}${edge.toFixed(1)}%`:'Mercado '+(impliedProb!=null?impliedProb.toFixed(1)+'%':'-')}</span></div>
    </div>
    <div class="stat-pills">
      <div class="stat-pill orange"><span>COTAÇÃO</span><b class="num">${esc(p.odd||'-')}</b></div>
      <div class="stat-pill purple"><span>PROB.</span><b class="num">${modelProb!=null?modelProb.toFixed(1)+'%':'-'}</b></div>
      <div class="stat-pill ${clsNum(p.ev)==='positive'?'teal':'red'}"><span>VANTAGEM</span><b class="num ${clsNum(p.ev)}">${p.ev?`${(Number(p.ev)*100).toFixed(1)}%`:'-'}</b></div>
    </div>
  </summary><div class="ticket-body"><div class="metric metric-wide"><span>Mercado recomendado</span><b>${esc(marketName(p.mercado))}</b><small class="market-code">${esc(p.mercado)}</small></div><div class="metric"><span>Confiança</span><b>${p.conf?`${(Number(p.conf)*100).toFixed(1)}%`:'-'}</b></div><div class="metric"><span>Qualidade (AUC)</span><b>${p.auc||'-'}</b></div><div class="metric"><span>Situação</span><b>${esc(humanStatus(p.status))}</b></div><div class="reason-block"><span>Por que</span><div class="reason-chips">${reasonChips(p)}</div></div><div class="metric metric-wide diagnostic"><span>Leitura da análise</span>${diagnosticHtml(p)}</div>${matchResultHtml(p)}${matchStatsButton(p)}</div></details>`;
}

let dateFilterJogos = null;
let dateFilterBanca = null;

function renderPredictions(){
  const q=($('searchPredictions')?.value||'').trim().toLowerCase();
  let rows=[...(data.previsoes?.linhas||[])];
  if(predictionFilter==='signals') rows=rows.filter(p=>p.sinal==='Sim');
  if(dateFilterJogos) rows=rows.filter(p=>normDate(p.data)===dateFilterJogos);
  if(q) rows=rows.filter(p=>`${p.jogo} ${p.liga} ${p.mercado} ${marketName(p.mercado)} ${p.descricao}`.toLowerCase().includes(q));
  const sortKey={ev:'ev',odd:'odd',conf:'conf'}[predictionSort];
  if(sortKey) rows.sort((a,b)=>(parseFloat(b[sortKey])||-Infinity)-(parseFloat(a[sortKey])||-Infinity));

  const totalRows=rows.length;
  const totalPages=Math.max(1, Math.ceil(totalRows / PREDICTIONS_PAGE_SIZE));
  predictionPage=Math.min(Math.max(1, predictionPage), totalPages);
  const start=(predictionPage-1)*PREDICTIONS_PAGE_SIZE;
  const pageRows=rows.slice(start, start+PREDICTIONS_PAGE_SIZE);

  $('predictions').innerHTML=pageRows.length
    ? pageRows.map(predictionCard).join('')
    : '<div class="market-card">Nenhuma aposta aprovada para esta data.</div>';

  const pager=$('predictionsPager');
  if(pager){
    pager.classList.toggle('hidden', totalRows===0);
    $('predictionsPageInfo').textContent=`${predictionPage} / ${totalPages}`;
    $('predictionsCountInfo').textContent=totalRows===1 ? '1 previsão' : `${totalRows} previsões`;
    $('predictionsPrev').disabled=predictionPage<=1;
    $('predictionsNext').disabled=predictionPage>=totalPages;
  }
}

function resetPredictionPage(){ predictionPage=1; }

function renderRanks(target, rows, key){
  $(target).innerHTML=(rows||[]).map(r=>`<div class="rank-item"><div><b>${esc(key==='mercado'?marketName(r[key]):r[key])}</b><small> · ${r.apostas} apostas · WR ${pct(r.winrate)}</small></div><div class="rank-right"><b class="num ${clsNum(r.roi)}">${pct(r.roi)}</b><small class="num">${money(r.lucro)}</small></div></div>`).join('')||'<div class="rank-item">Sem histórico liquidado.</div>';
}

/* ---------- Heatmap de performance (estilo GitHub) ---------- */
function renderHeatmap(rows){
  const el = $('heatmap');
  if(!el) return;
  const map = new Map((rows||[]).map(r=>[r.data, r]));
  const totalDays = 168; // ~24 semanas, cabe bem na tela do celular
  const today = new Date(); today.setHours(0,0,0,0);
  const start = new Date(today); start.setDate(start.getDate() - totalDays + 1);
  start.setDate(start.getDate() - start.getDay()); // alinha no domingo

  const cells = [];
  for(let d = new Date(start); d <= today; d.setDate(d.getDate()+1)){
    const iso = `${d.getFullYear()}-${pad2(d.getMonth()+1)}-${pad2(d.getDate())}`;
    const rec = map.get(iso);
    cells.push({ iso, lucro: rec ? rec.lucro : null, apostas: rec ? rec.apostas : 0 });
  }

  const maxAbs = Math.max(1, ...cells.filter(c=>c.lucro!==null).map(c=>Math.abs(c.lucro)));
  const levelOf = (v) => { const r = Math.abs(v)/maxAbs; return r>0.66?3:r>0.33?2:1; };

  let html = '';
  for(let i=0;i<cells.length;i+=7){
    html += '<div class="heatmap-col">';
    for(let j=i;j<i+7 && j<cells.length;j++){
      const c = cells[j];
      let cls = 'heatmap-cell';
      let title;
      if(c.lucro===null){ cls+=' empty'; title = `${isoToBR(c.iso)} · sem apostas liquidadas`; }
      else if(c.lucro>0){ cls+=` win l${levelOf(c.lucro)}`; title = `${isoToBR(c.iso)} · ${c.apostas} apostas · ${money(c.lucro)}`; }
      else if(c.lucro<0){ cls+=` loss l${levelOf(c.lucro)}`; title = `${isoToBR(c.iso)} · ${c.apostas} apostas · ${money(c.lucro)}`; }
      else { cls+=' flat'; title = `${isoToBR(c.iso)} · ${c.apostas} apostas · empatado`; }
      html += `<div class="${cls}" title="${esc(title)}"></div>`;
    }
    html += '</div>';
  }
  el.innerHTML = html;
  const first = cells[0]?.iso, last = cells.at(-1)?.iso;
  if($('heatmapRange') && first && last) $('heatmapRange').textContent = `${isoToBR(first)} – ${isoToBR(last)}`;
}

function renderBankRows(){
  let rows=[...(data.historico.linhas||[])];
  if(dateFilterBanca) rows=rows.filter(r=>normDate(r.data)===dateFilterBanca);
  const resultLink = (r) => r.resultado_url
    ? `<a class="result-search-link" href="${esc(r.resultado_url)}" target="_blank" rel="noopener noreferrer">Ver resultado</a>`
    : '';
  $('bankRows').innerHTML=rows.length?rows.map(r=>`<tr><td>${esc(r.data)}</td><td><span>${esc(r.jogo)}</span>${resultLink(r)}</td><td title="${esc(r.mercado)}">${esc(marketName(r.mercado))}</td><td>${esc(r.resultado)}</td><td class="${clsNum(r.lucro)}">${esc(r.lucro)}</td></tr>`).join(''):'<tr><td colspan="5">Nenhuma aposta nesse dia.</td></tr>';
}

function renderPaper(paper){
  const p=paper||{}, m=p.metrics||{}, progress=p.sample_progress||{}, automation=p.automation||{};
  if($('paperStatus')) $('paperStatus').textContent=humanStatus(p.status||'AGUARDANDO');
  if($('paperCycle')) $('paperCycle').textContent=p.cycle_id?`Ciclo ${p.cycle_id}`:'Ciclo não iniciado';
  if($('paperTiming')) $('paperTiming').textContent=`Previsões entre ${p.timing?.capture_min_minutes||'-'} e ${p.timing?.capture_max_minutes||'-'} min antes · resultados após ${p.timing?.settlement_after_minutes||'-'} min`;
  if($('paperAutomation')) $('paperAutomation').textContent=`Automação: ${humanStatus(automation.status||'SEM_HEARTBEAT')}`;
  if($('paperCalibration')) $('paperCalibration').textContent=`Calibração: ${humanStatus(p.calibration_status||'SEM_AMOSTRA')}`;
  if($('paperSettled')) $('paperSettled').textContent=Number(m.settled||0).toLocaleString('pt-BR');
  if($('paperPending')) $('paperPending').textContent=`${Number(m.pending||0)} pendentes`;
  if($('paperProfit')) { $('paperProfit').textContent=money(m.profit); $('paperProfit').className=clsNum(m.profit); }
  if($('paperStake')) $('paperStake').textContent=`Total simulado ${money(m.stake)}`;
  if($('paperRoi')) { $('paperRoi').textContent=m.roi==null?'-':pct(m.roi); $('paperRoi').className=clsNum(m.roi); }
  if($('paperRoiCi')) $('paperRoiCi').textContent=m.ci95_low==null?'Faixa de confiança aguardando':`Faixa de confiança: ${pct(m.ci95_low)} a ${pct(m.ci95_high)}`;
  if($('paperBrier')) $('paperBrier').textContent=m.brier==null?'-':Number(m.brier).toFixed(3);
  if($('paperCalError')) $('paperCalError').textContent=m.calibration_error==null?'Erro -':`Erro ${pct(m.calibration_error)}`;
  const fraction=Math.max(0,Math.min(1,Number(progress.decision_pct||0)));
  if($('paperProgress')) $('paperProgress').style.width=`${fraction*100}%`;
  if($('paperProgressText')) $('paperProgressText').textContent=`${Math.round(fraction*100)}%`;
  if($('paperMessage')) $('paperMessage').textContent=p.message||'Coletando dados prospectivos.';
  renderRanks('paperMarkets',p.por_mercado||[],'mercado');
  renderRanks('paperLeagues',p.por_liga||[],'liga');
  renderRanks('paperMonths',p.por_mes||[],'mes');
  const paperRows=$('paperRows');
  if(paperRows){
    const rows=p.linhas||[];
    paperRows.innerHTML=rows.length?rows.map(r=>{
      const result=String(r.resultado||'pendente').toLowerCase();
      const label=result==='ganhou'?'Ganhou':result==='perdeu'?'Perdeu':'Pendente';
      const resultClass=result==='ganhou'?'positive':result==='perdeu'?'negative':'';
      return `<tr><td>${esc(r.data)}</td><td>${esc(r.jogo)}</td><td>${esc(marketName(r.mercado))}</td><td class="${resultClass}"><b>${label}</b></td><td class="${clsNum(r.lucro)}">${money(r.lucro)}</td></tr>`;
    }).join(''):'<tr><td colspan="5">Nenhuma aposta registrada neste ciclo.</td></tr>';
  }
}

function render(d){
  data=d;
  if(d.dashboard_version){
    latestVersionPayload=d.dashboard_version;
    renderVersionStatus(d.dashboard_version);
  }
  $('lastUpdate').textContent=`Sincronizado ${d.agora}`;
  $('health').textContent=d.health;
  $('healthLabel').textContent=d.health_label;
  $('healthMode').textContent=displayMode(d.modo);
  const hcolor=d.health>=90?'var(--sage)':d.health>=70?'var(--brass)':'var(--wine)';
  $('healthDot').style.background=hcolor;
  $('saldo').textContent=money(d.historico.saldo); $('lucro').textContent=`Lucro ${money(d.historico.lucro)}`;
  $('roi').textContent=pct(d.historico.roi); $('winrate').textContent=`Taxa de acerto ${pct(d.historico.winrate)}`;
  $('sinais').textContent=d.previsoes.sinais; $('previsoesTotal').textContent=`${d.previsoes.total} previsões analisadas · ${d.previsoes.jogos} jogos`;
  $('prevFile').textContent=d.previsoes.arquivo; $('modelCount').textContent=`${d.modelos.total} modelos`;
  $('topSignalCount').textContent=`${d.previsoes.top.length} oficiais`;
  const sourceNotice=$('predictionSourceNotice');
  if(sourceNotice){
    const official=Boolean(d.previsoes.oficial);
    const markets=(d.previsoes.mercados_oficiais||[]).map(marketName).join(', ');
    sourceNotice.className=`prediction-source-notice ${official?'official':'legacy'}`;
    sourceNotice.innerHTML=`<b>${esc(d.previsoes.rotulo_origem||'Origem não identificada')}</b><span>${esc(d.previsoes.mensagem_origem||'')}${markets?` Mercado do ciclo: ${esc(markets)}.`:''}</span>${!official&&d.previsoes.sinais_no_arquivo?`<small>${d.previsoes.sinais_no_arquivo} marcação(ões) antiga(s) ignorada(s) como aposta oficial.</small>`:''}`;
  }
  $('statusStrip').innerHTML=[`Base ${Number(d.base_linhas).toLocaleString('pt-BR')} linhas`,`Base ${d.idade_base}`,`Teste histórico ${d.idade_backtest}`,`Modelos ${d.idade_modelos}`,`Previsões ${d.idade_previsoes}`,`IP ${d.ip}`].map(x=>`<span class="chip">${esc(x)}</span>`).join('');
  $('topPredictions').innerHTML=d.previsoes.top.length?d.previsoes.top.map(predictionCard).join(''):'<div class="market-card">Nenhuma indicação oficial do ciclo paper no momento.</div>';
  renderPredictions();
  renderSpotlight();
  checkNewSignals(d.previsoes.top);

  $('drawdownChip').textContent=`Queda ${pct(d.historico.drawdown)}`;
  renderChart('bankChart',d.historico.curva); renderChart('bankChartFull',d.historico.curva);
  $('homeBankStats').innerHTML=[['Apostas',d.historico.total],['Ganhos',d.historico.ganhou],['Perdas',d.historico.perdeu],['Pendentes',d.historico.pendente],['Valor médio',money(d.historico.stake_media)],['Modelos',d.modelos.total]].map(([a,b])=>`<div class="mini-stat"><span>${a}</span><b>${b}</b></div>`).join('');

  $('bankSaldo').textContent=money(d.historico.saldo); $('bankLucro').textContent=`Lucro ${money(d.historico.lucro)}`;
  $('bankRoi').textContent=pct(d.historico.roi); $('bankStake').textContent=`Total simulado ${money(d.historico.stake_total)}`;
  $('bankWinrate').textContent=pct(d.historico.winrate); $('bankRecord').textContent=`${d.historico.ganhou}G · ${d.historico.perdeu}P`;
  $('bankDrawdown').textContent=pct(d.historico.drawdown); $('bankPending').textContent=`${d.historico.pendente} pendentes`;
  renderRanks('bankMarkets',d.historico.por_mercado,'mercado'); renderRanks('bankLeagues',d.historico.por_liga,'liga');
  renderBankRows();
  renderHeatmap(d.historico.heatmap);
  renderPaper(d.paper);

  $('modelStats').innerHTML=[['Modelos',d.modelos.total],['Aprovados',d.modelos.aprovados],['Especialistas',d.modelos.contextos],['Qualidade média (AUC)',d.modelos.auc_medio.toFixed(3)],['Erro prob. (Brier)',d.modelos.brier?d.modelos.brier.toFixed(3):'-'],['Mercados ativos',d.mercados.qtd]].map(([a,b])=>`<div class="mini-stat"><span>${a}</span><b>${b}</b></div>`).join('');
  $('models').innerHTML=(d.modelos.itens||[]).map(m=>`<div class="market-card"><div><b>${esc(marketName(m.mercado))}</b><small>${esc(m.mercado)} · ${esc(m.modelo)} · ${m.contextos} especialistas</small></div><div class="rank-right"><b class="num">Qualidade ${Number(m.auc).toFixed(3)}</b><small class="num">Retorno histórico ${pct(m.roi_bt)}</small></div></div>`).join('')||'<div class="market-card">Nenhum modelo encontrado.</div>';
  $('markets').innerHTML=(d.mercados.itens||[]).map(m=>`<div class="market-card ${m.ativo?'':'off'}"><div><b>${esc(marketName(m.mercado))}</b><small>${esc(m.mercado)} · ${esc(m.modelo||humanStatus(m.motivo)||'')}</small></div><div class="rank-right"><b class="num ${m.ativo?'status-on':'status-off'}">${m.ativo?'ATIVO':'BLOQUEADO'}</b><small class="num">Qualidade ${Number(m.auc||0).toFixed(3)} · Retorno ${pct(m.roi)}</small></div></div>`).join('');

  const tm=d.telemetry||{};
  if($('cpuUsage')) $('cpuUsage').textContent=tm.cpu==null?'-':`${tm.cpu}%`;
  if($('ramUsage')) $('ramUsage').textContent=tm.ram==null?'-':`${tm.ram}%`;
  if($('processMemory')) $('processMemory').textContent=tm.process_mb==null?'-':`${tm.process_mb} MB`;
  if($('uptimeChip')) $('uptimeChip').textContent=`Ligado ${tm.uptime||'-'}`;
  if($('cpuBar')) $('cpuBar').style.width=`${Math.min(100,Number(tm.cpu||0))}%`;
  if($('ramBar')) $('ramBar').style.width=`${Math.min(100,Number(tm.ram||0))}%`;
  const dashboardTaskRunning=Boolean(d.task?.running);
  const externalTaskRunning=Boolean(tm.external_pipeline_running);
  const pipelineRunning=dashboardTaskRunning||externalTaskRunning;
  if($('pipelineState')) $('pipelineState').textContent=pipelineRunning?'Executando':'Parado';
  if($('pipelineStep')) {
    $('pipelineStep').textContent=dashboardTaskRunning
      ? friendlyStep(d.task.step||d.task.task||'Processando')
      : externalTaskRunning
        ? (tm.external_pipeline_name||tm.external_pipeline_detail||'Processo externo detectado')
        : 'Aguardando';
  }

  $('alerts').innerHTML=(d.alertas||[]).map(a=>`<div class="alert ${a==='Tudo certo no momento'?'ok':''}">${esc(a)}</div>`).join('');
  $('logs').textContent=(d.logs||[]).join('\n');
  renderTask(d.task||{},tm);
}

function renderTask(t,tm={}){
  const external=Boolean(tm.external_pipeline_running);
  const running=Boolean(t.running)||external;
  $('taskLabel').textContent=t.running
    ? `${taskLabel(t.task)} · ${friendlyStep(t.step)}`
    : external
      ? `${tm.external_pipeline_name||'Processo externo'} · em execução pelo terminal`
      : (t.success===true?'Última execução concluída':t.error?`Erro: ${t.error}`:'Nenhuma execução');
  const p=t.running?Number(t.progress||0):0;
  $('taskProgress').style.width=`${p}%`;
  $('taskProgressText').textContent=external&&!t.running?'Em andamento':`${p}%`;
  document.querySelectorAll('[data-task]').forEach(b=>b.disabled=running);
}

function setConn(state){
  const dot=$('connDot'); dot.classList.remove('off','pending');
  if(state==='off') dot.classList.add('off');
  if(state==='pending') dot.classList.add('pending');
}

function renderVersionStatus(v){
  if(!v) return;
  const tm=data.task||{};
  const status=v.pipeline_status || (tm.running?'rodando':'monitoramento');
  const etapa=v.etapa_atual || tm.step || 'monitoramento';
  const concl=v.datas_concluidas ?? '-';
  const total=v.datas_totais ?? '-';
  const pctVal=v.percentual;
  const pctTxt=pctVal===null || pctVal===undefined ? '-' : `${Number(pctVal).toFixed(1)}%`;
  const last=v.ultima_previsao_data || '-';
  const next=v.proxima_data || '-';
  const el=$('pipelineStep');
  if(el) el.textContent=`${etapa} · ${concl}/${total} · ${pctTxt} · última ${last} · próxima ${next}`;
  const state=$('pipelineState');
  if(state) state.textContent=humanStatus(status);
}

function showVersionBanner(v){
  latestVersionPayload=v;
  const text=$('versionBannerText');
  if(text){
    const last=v?.ultima_previsao_data ? `Última data: ${v.ultima_previsao_data}.` : 'Dados novos detectados.';
    text.textContent=`${last} Toque para carregar a tela.`;
  }
  $('versionBanner')?.classList.add('show');
}

function hideVersionBanner(){
  $('versionBanner')?.classList.remove('show');
}

async function checkVersion(){
  try{
    const r=await fetch('/api/version',{cache:'no-store'});
    if(!r.ok) throw new Error('bad version');
    const v=await r.json();
    renderVersionStatus(v);
    const nextVersion=String(v.versao_dados||'');
    if(!currentDataVersion){
      currentDataVersion=nextVersion;
      localStorage.setItem('fl_data_version', currentDataVersion);
      return;
    }
    if(nextVersion && nextVersion!==currentDataVersion) showVersionBanner(v);
  }catch(e){}
}

async function refresh(selectedDate=dateFilterJogos){
  setConn('pending');
  try{
    const query=selectedDate?`?date=${encodeURIComponent(selectedDate)}`:'';
    const r=await fetch(`/api/status${query}`,{cache:'no-store'});
    if(r.ok){
      render(await r.json()); failCount=0; setConn('on'); $('staleBanner').classList.remove('show');
      const versionPayload=latestVersionPayload;
      if(versionPayload?.versao_dados){
        currentDataVersion=String(versionPayload.versao_dados);
        localStorage.setItem('fl_data_version', currentDataVersion);
        hideVersionBanner();
      } else {
        await checkVersion();
      }
    }
    else throw new Error('bad status');
  }catch(e){
    failCount++; setConn('off');
    if(failCount>=2) $('staleBanner').classList.add('show');
  }
}
async function refreshTaskLog(){try{const r=await fetch('/api/task-log',{cache:'no-store'});if(r.ok)$('taskMiniLog').textContent=await r.text();}catch(e){}}

for(const b of document.querySelectorAll('[data-tab]')) b.onclick=()=>{
  buzz();
  document.querySelectorAll('[data-tab]').forEach(x=>{x.classList.toggle('active',x===b); x.setAttribute('aria-pressed',x===b?'true':'false');});
  document.querySelectorAll('.tab-panel').forEach(x=>x.classList.toggle('active',x.dataset.panel===b.dataset.tab));
  localStorage.setItem('fl_tab', b.dataset.tab);
  window.scrollTo({top:0,behavior:'smooth'});
};
for(const b of document.querySelectorAll('[data-filter]')) b.onclick=()=>{predictionFilter=b.dataset.filter; localStorage.setItem('fl_filter',predictionFilter); document.querySelectorAll('[data-filter]').forEach(x=>x.classList.toggle('active',x===b)); resetPredictionPage(); renderPredictions();};
$('searchPredictions').oninput=()=>{ resetPredictionPage(); renderPredictions(); };
$('predictionsPrev')?.addEventListener('click',()=>{ if(predictionPage>1){ predictionPage--; renderPredictions(); window.scrollTo({top:$('predictions').offsetTop-120,behavior:'smooth'}); } });
$('predictionsNext')?.addEventListener('click',()=>{ predictionPage++; renderPredictions(); window.scrollTo({top:$('predictions').offsetTop-120,behavior:'smooth'}); });
$('sortPredictions').value=predictionSort;
$('sortPredictions').onchange=()=>{predictionSort=$('sortPredictions').value; localStorage.setItem('fl_sort',predictionSort); renderPredictions();};

document.querySelectorAll('[data-bank-segment]').forEach(btn=>btn.onclick=()=>{document.querySelectorAll('[data-bank-segment]').forEach(x=>x.classList.toggle('active',x===btn));['markets','leagues','history'].forEach(k=>$(`bank${k[0].toUpperCase()+k.slice(1)}`).classList.toggle('hidden',k!==btn.dataset.bankSegment));});
document.querySelectorAll('[data-paper-segment]').forEach(btn=>btn.onclick=()=>{document.querySelectorAll('[data-paper-segment]').forEach(x=>x.classList.toggle('active',x===btn));['markets','leagues','months'].forEach(k=>$(`paper${k[0].toUpperCase()+k.slice(1)}`).classList.toggle('hidden',k!==btn.dataset.paperSegment));});
document.querySelectorAll('[data-ai-segment]').forEach(btn=>btn.onclick=()=>{document.querySelectorAll('[data-ai-segment]').forEach(x=>x.classList.toggle('active',x===btn));$('models').classList.toggle('hidden',btn.dataset.aiSegment!=='models');$('markets').classList.toggle('hidden',btn.dataset.aiSegment!=='markets');});

document.querySelectorAll('[data-task]').forEach(btn=>btn.onclick=async()=>{
  buzz(15); btn.disabled=true;
  try{
    if(btn.dataset.task==='settle_flashscore_apply'){
      const confirmed=window.confirm('Aplicar somente os resultados finalizados e validados pelo Flashscore? Um backup do histórico será criado antes da alteração.');
      if(!confirmed) return;
    }
    const r=await fetch(`/api/run/${btn.dataset.task}`,{method:'POST'});
    const j=await r.json();
    if(!r.ok) toast(j.detail||'Falha ao iniciar');
    else{ toast('Execução iniciada'); await refresh(); await refreshTaskLog(); }
  }catch(e){ toast('Falha de conexão'); }
  finally{ setTimeout(()=>btn.disabled=false,700); }
});
$('refreshNow').onclick=()=>{refresh();refreshTaskLog();toast('Painel atualizado');};
$('versionRefreshNow')?.addEventListener('click',()=>{refresh();refreshTaskLog();toast('Painel atualizado');});
$('copyLogs')?.addEventListener('click',async()=>{
  try{ await navigator.clipboard.writeText($('logs').textContent||''); toast('Registros copiados'); }
  catch(e){ toast('Não foi possível copiar'); }
});

/* ---------- Toggle de alertas ---------- */
(function initBell(){
  const btn=$('alertToggle'); if(!btn) return;
  btn.classList.toggle('muted', !alertsEnabled);
  btn.onclick=()=>{
    alertsEnabled=!alertsEnabled;
    localStorage.setItem('fl_alerts_enabled', alertsEnabled?'1':'0');
    btn.classList.toggle('muted', !alertsEnabled);
    buzz();
    toast(alertsEnabled?'Alertas de sinal ativados':'Alertas de sinal desativados');
  };
})();

/* ---------- Exportar histórico da banca em CSV ---------- */
$('exportBankCsv')?.addEventListener('click',()=>{
  const rows=data.historico?.linhas||[];
  if(!rows.length){ toast('Sem histórico para exportar'); return; }
  const head=['Data','Jogo','Mercado','Liga','Cotação','Resultado','Valor simulado','Lucro'];
  const body=rows.map(r=>[r.data,r.jogo,marketName(r.mercado),r.liga,r.odd,r.resultado,r.stake,r.lucro]
    .map(v=>`"${String(v??'').replace(/"/g,'""')}"`).join(','));
  const csv=[head.join(','),...body].join('\n');
  const blob=new Blob(['\ufeff'+csv],{type:'text/csv;charset=utf-8;'});
  const url=URL.createObjectURL(blob);
  const a=document.createElement('a');
  a.href=url; a.download=`football-lab-banca-${new Date().toISOString().slice(0,10)}.csv`;
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
  buzz(); toast('CSV exportado');
});

/* ---------- Calendário lateral ---------- */
(function calendar(){
  const backdrop=$('drawerBackdrop'), drawer=$('calDrawer');
  let context='jogos'; // 'jogos' | 'banca'
  let view=new Date();

  function datesFor(ctx){
    if(ctx==='jogos'){
      const available=data.previsoes?.datas_disponiveis||[];
      if(available.length) return new Set(available.map(normDate).filter(Boolean));
    }
    const src = ctx==='jogos' ? (data.previsoes?.linhas||[]) : (data.historico?.linhas||[]);
    return new Set(src.map(r=>normDate(r.data)).filter(Boolean));
  }

  function open(ctx){
    context=ctx; view=(ctx==='jogos'&&dateFilterJogos)?new Date(dateFilterJogos+'T12:00:00'):new Date();
    backdrop.classList.add('show'); drawer.classList.add('show'); drawer.setAttribute('aria-hidden','false');
    draw();
  }
  function close(){
    backdrop.classList.remove('show'); drawer.classList.remove('show'); drawer.setAttribute('aria-hidden','true');
  }

  function draw(){
    const y=view.getFullYear(), m=view.getMonth();
    $('calMonthLabel').textContent=`${MESES[m]} ${y}`;
    const marked=datesFor(context);
    const selected = context==='jogos' ? dateFilterJogos : dateFilterBanca;
    const todayIso=new Date().toISOString().slice(0,10);
    const firstWeekday=new Date(y,m,1).getDay();
    const daysInMonth=new Date(y,m+1,0).getDate();
    let html='';
    for(let i=0;i<firstWeekday;i++) html+='<div class="cal-day empty"></div>';
    for(let day=1;day<=daysInMonth;day++){
      const iso=`${y}-${pad2(m+1)}-${pad2(day)}`;
      const cls=['cal-day'];
      if(marked.has(iso)) cls.push('has-data');
      if(iso===todayIso) cls.push('today');
      if(iso===selected) cls.push('selected');
      html+=`<button type="button" class="${cls.join(' ')}" data-iso="${iso}">${day}</button>`;
    }
    $('calGrid').innerHTML=html;
    $('calGrid').querySelectorAll('.cal-day[data-iso]').forEach(btn=>btn.onclick=async()=>{
      buzz();
      const iso=btn.dataset.iso;
      if(context==='jogos'){
        dateFilterJogos=iso;
        updateChipJogos();
        resetPredictionPage();
        close();
        await refresh(iso);
        renderPredictions();
        renderSpotlight();
      } else {
        dateFilterBanca=iso;
        updateChipBanca();
        renderBankRows();
        close();
      }
    });
  }

  function updateChipJogos(){
    const chip=$('dateChipJogos');
    if(dateFilterJogos){ chip.classList.add('show'); $('dateChipJogosText').textContent=`Filtrando ${isoToBR(dateFilterJogos)}`; }
    else chip.classList.remove('show');
  }
  function updateChipBanca(){
    const chip=$('dateChipBanca');
    if(dateFilterBanca){ chip.classList.add('show'); $('dateChipBancaText').textContent=`Filtrando ${isoToBR(dateFilterBanca)}`; }
    else chip.classList.remove('show');
  }

  $('openCalJogos').onclick=()=>open('jogos');
  $('openCalBanca').onclick=()=>open('banca');
  $('calPrev').onclick=()=>{ view=new Date(view.getFullYear(),view.getMonth()-1,1); draw(); };
  $('calNext').onclick=()=>{ view=new Date(view.getFullYear(),view.getMonth()+1,1); draw(); };
  $('calClose').onclick=close;
  backdrop.onclick=close;
  $('calClear').onclick=()=>{
    if(context==='jogos'){ dateFilterJogos=null; updateChipJogos(); resetPredictionPage(); refresh(null); }
    else { dateFilterBanca=null; updateChipBanca(); renderBankRows(); }
    close();
  };
  $('dateChipJogosClear').onclick=()=>{ dateFilterJogos=null; updateChipJogos(); resetPredictionPage(); refresh(null); };
  $('dateChipBancaClear').onclick=()=>{ dateFilterBanca=null; updateChipBanca(); renderBankRows(); };
})();

/* restaura tab e filtro salvos */
(function restore(){
  const savedTab=localStorage.getItem('fl_tab');
  if(savedTab){ const btn=document.querySelector(`[data-tab="${savedTab}"]`); if(btn) btn.click(); }
  const savedFilter=document.querySelector(`[data-filter="${predictionFilter}"]`);
  if(savedFilter) document.querySelectorAll('[data-filter]').forEach(x=>x.classList.toggle('active',x===savedFilter));
})();

/* online/offline nativo do navegador */
window.addEventListener('online', ()=>{ setConn('on'); checkVersion(); });
window.addEventListener('offline', ()=> setConn('off'));

/* pull-to-refresh (somente quando o topo da página está visível) */
(function pullToRefresh(){
  const ptr=$('ptr'); let startY=0, pulling=false;
  document.addEventListener('touchstart',(e)=>{ if(window.scrollY<=0){ startY=e.touches[0].clientY; pulling=true; } }, {passive:true});
  document.addEventListener('touchmove',(e)=>{
    if(!pulling) return;
    const dy=e.touches[0].clientY-startY;
    if(dy>60) ptr.classList.add('show');
  }, {passive:true});
  document.addEventListener('touchend',()=>{
    if(pulling && ptr.classList.contains('show')){ buzz(); refresh(); refreshTaskLog(); }
    pulling=false; setTimeout(()=>ptr.classList.remove('show'),400);
  });
})();

render(data); setConn('on');
showDailyBrief(data);
if(data.dashboard_version) showVersionBanner(data.dashboard_version);
refresh();
setInterval(checkVersion,VERSION_POLL_MS); setInterval(refreshTaskLog,30000); refreshTaskLog();
