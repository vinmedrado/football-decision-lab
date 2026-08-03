// Football Decision Lab — Painel ao vivo
// Poll simples via fetch; sem framework, sem build step.

const REFRESH_RAPIDO_MS = 2500;   // status/log do processo
const REFRESH_MEDIO_MS  = 8000;   // banca, apostas do dia
const REFRESH_LENTO_MS  = 20000;  // histórico recente

function fmtMoeda(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return v.toLocaleString('pt-BR', { style: 'currency', currency: 'BRL' });
}
function fmtPct(v, casas = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return `${v.toFixed(casas)}%`;
}
function fmtNum(v, casas = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return Number(v).toFixed(casas);
}

async function getJSON(url) {
  try {
    const r = await fetch(url, { cache: 'no-store' });
    if (!r.ok) throw new Error(r.status);
    return await r.json();
  } catch (e) {
    return null;
  }
}

// ─── Status / terminal ao vivo ────────────────────────────────────────────

function classificarLinha(linha) {
  if (/✔|OK\b/.test(linha)) return 'ok';
  if (/✘|ERRO|Traceback|Error/.test(linha)) return 'erro';
  if (linha.startsWith('$')) return 'destaque';
  return '';
}

let ultimaLinhaVista = null;

async function atualizarStatus() {
  const data = await getJSON('/api/status');
  const pill = document.getElementById('status-pill');
  const pillTexto = document.getElementById('status-pill-texto');
  const etapaBox = document.getElementById('etapa-atual');
  const term = document.getElementById('terminal-corpo');

  if (!data) return;

  if (data.rodando) {
    pill.classList.add('rodando');
    pillTexto.textContent = `Rodando · ${data.titulo || '...'}`;
  } else {
    pill.classList.remove('rodando');
    pillTexto.textContent = 'Ocioso';
  }

  if (data.titulo) {
    const status = data.rodando ? 'em andamento' : (data.rc === 0 ? 'concluída' : (data.rc !== null ? `erro (código ${data.rc})` : 'aguardando'));
    etapaBox.innerHTML = `Última etapa: <b>${escapeHtml(data.titulo)}</b> — ${status}` +
      (data.atualizado_em ? ` <span style="opacity:.6">(${data.atualizado_em.replace('T', ' ')})</span>` : '');
  } else {
    etapaBox.textContent = 'Nenhuma etapa rodada ainda nesta sessão.';
  }

  const linhas = data.log || [];
  const chave = linhas.join('\n');
  if (chave !== ultimaLinhaVista) {
    ultimaLinhaVista = chave;
    const estavaNoFim = term.scrollHeight - term.scrollTop - term.clientHeight < 40;
    term.innerHTML = linhas.map(l => {
      const cls = classificarLinha(l);
      return `<div class="terminal-linha ${cls}">${escapeHtml(l)}</div>`;
    }).join('') + '<span class="cursor"></span>';
    if (estavaNoFim) term.scrollTop = term.scrollHeight;
  }
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ─── Banca ─────────────────────────────────────────────────────────────────

async function atualizarBanca() {
  const b = await getJSON('/api/banca');
  if (!b) return;

  const lucro = b.lucro_total ?? 0;
  document.getElementById('banca-atual').textContent = fmtMoeda(b.banca_atual);
  document.getElementById('banca-inicial-detalhe').textContent =
    `inicial ${fmtMoeda(b.banca_inicial)} · pico ${fmtMoeda(b.banca_pico)}`;

  const elLucro = document.getElementById('lucro-total');
  elLucro.textContent = `${lucro >= 0 ? '+' : ''}${fmtMoeda(lucro)}`;
  elLucro.className = 'valor ' + (lucro >= 0 ? 'positivo' : 'negativo');
  document.getElementById('roi-detalhe').textContent = `ROI ${fmtPct(b.roi_total)}`;

  document.getElementById('taxa-acerto').textContent = b.taxa_acerto != null ? fmtPct(b.taxa_acerto) : '—';
  document.getElementById('taxa-detalhe').textContent =
    `${b.total_ganhos ?? 0} ganhas / ${b.total_perdas ?? 0} perdidas`;

  document.getElementById('drawdown').textContent = b.drawdown_atual != null ? fmtPct(b.drawdown_atual) : '—';
  document.getElementById('drawdown-detalhe').textContent = b.ultima_atualizacao
    ? `atualizado ${b.ultima_atualizacao}` : '';
}

// ─── Apostas de hoje ───────────────────────────────────────────────────────

async function atualizarApostasHoje() {
  const d = await getJSON('/api/apostas-hoje');
  const corpo = document.getElementById('apostas-hoje-corpo');
  const contagem = document.getElementById('apostas-hoje-contagem');
  if (!d) return;

  contagem.textContent = `${d.liberadas} liberadas · ${d.bloqueadas} bloqueadas`;

  if (!d.arquivo_encontrado || d.apostas.length === 0) {
    corpo.innerHTML = `<div class="vazio"><span class="icone">⚽</span>Nenhuma previsão gerada para hoje ainda.<br>Rode "Previsões → Prever hoje" no terminal.</div>`;
    return;
  }

  const linhas = d.apostas.map(a => `
    <tr>
      <td>${escapeHtml(a.jogo)}</td>
      <td>${escapeHtml(a.liga)}</td>
      <td>${escapeHtml(a.mercado)}</td>
      <td>${fmtNum(a.prob * 100, 1)}%</td>
      <td>${fmtNum(a.odd)}</td>
      <td>${fmtNum(a.ev * 100, 1)}%</td>
      <td>${a.apostar
        ? '<span class="badge liberada">● liberada</span>'
        : `<span class="badge bloqueada" title="${escapeHtml(a.motivo_nao_apostar)}">○ bloqueada</span>`}</td>
    </tr>`).join('');

  corpo.innerHTML = `
    <table>
      <thead><tr>
        <th>Jogo</th><th>Liga</th><th>Mercado</th><th>Prob.</th><th>Odd</th><th>EV</th><th>Status</th>
      </tr></thead>
      <tbody>${linhas}</tbody>
    </table>`;
}

// ─── Histórico recente ─────────────────────────────────────────────────────

async function atualizarHistorico() {
  const d = await getJSON('/api/historico-recente');
  const corpo = document.getElementById('historico-corpo');
  if (!d) return;

  if (!d.apostas || d.apostas.length === 0) {
    corpo.innerHTML = `<div class="vazio"><span class="icone">📜</span>Nenhuma aposta liquidada ainda.</div>`;
    return;
  }

  const linhas = d.apostas.map(a => `
    <tr>
      <td>${escapeHtml(a.data)}</td>
      <td>${escapeHtml(a.jogo)}</td>
      <td>${escapeHtml(a.mercado)}</td>
      <td>${fmtNum(a.odd)}</td>
      <td>${fmtMoeda(a.valor_apostado)}</td>
      <td><span class="badge ${a.resultado}">${a.resultado === 'ganhou' ? '● ganhou' : '● perdeu'}</span></td>
      <td class="${a.lucro >= 0 ? '' : ''}" style="color:${a.lucro >= 0 ? 'var(--verde)' : 'var(--vermelho)'}">
        ${a.lucro >= 0 ? '+' : ''}${fmtMoeda(a.lucro)}
      </td>
    </tr>`).join('');

  corpo.innerHTML = `
    <table>
      <thead><tr>
        <th>Data</th><th>Jogo</th><th>Mercado</th><th>Odd</th><th>Stake</th><th>Resultado</th><th>Lucro</th>
      </tr></thead>
      <tbody>${linhas}</tbody>
    </table>`;
}

// ─── Loop de atualização ───────────────────────────────────────────────────

function iniciar() {
  atualizarStatus();
  atualizarBanca();
  atualizarApostasHoje();
  atualizarHistorico();

  setInterval(atualizarStatus, REFRESH_RAPIDO_MS);
  setInterval(atualizarBanca, REFRESH_MEDIO_MS);
  setInterval(atualizarApostasHoje, REFRESH_MEDIO_MS);
  setInterval(atualizarHistorico, REFRESH_LENTO_MS);
}

document.addEventListener('DOMContentLoaded', iniciar);
