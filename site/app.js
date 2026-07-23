"use strict";

const pipelineDetails = {
  ingest: {
    title: "Ingestão diária",
    text: "Atualiza calendário e estatísticas de forma incremental. Execuções repetidas preservam o mesmo estado e permitem recuperar dias em que o notebook ficou desligado.",
    control: "IDEMPOTÊNCIA"
  },
  validate: {
    title: "Validação",
    text: "Confere schema, tipos, duplicidades, completude e coerência temporal. Uma entrada inválida bloqueia a etapa seguinte em vez de produzir previsões silenciosamente incorretas.",
    control: "FAIL CLOSED"
  },
  features: {
    title: "Feature pipeline",
    text: "Transformações são reproduzíveis e respeitam o instante da partida. Estatísticas futuras não podem aparecer nas variáveis usadas para uma decisão passada.",
    control: "POINT-IN-TIME"
  },
  research: {
    title: "Backtest temporal",
    text: "Avalia o modelo fora da amostra usando cortes por data. O estado permanece sob revisão porque ROI histórico positivo não veio acompanhado de calibração superior ao baseline.",
    control: "TEMPORAL SPLITS"
  },
  registry: {
    title: "Model registry",
    text: "Registra versão, métricas, features e hashes dos artefatos. O paper trading consegue provar exatamente qual bundle gerou cada sinal.",
    control: "SHA-256"
  },
  capture: {
    title: "Captura pré-jogo",
    text: "O scanner roda a cada 15 minutos e aceita sinais apenas entre 30 e 90 minutos antes do kickoff. Linhas capturadas são imutáveis.",
    control: "IMMUTABLE LEDGER"
  },
  settle: {
    title: "Settlement",
    text: "Resultados são pesquisados depois do encerramento esperado. Partidas adiadas permanecem pendentes e entram numa fila de novas tentativas, sem inventar um placar.",
    control: "RETRY POLICY"
  },
  monitor: {
    title: "Observabilidade",
    text: "Acompanha volume, ROI, drawdown, intervalo de confiança e calibração. Alertas usam a origem do dado e o tamanho da amostra para impedir conclusões prematuras.",
    control: "SAMPLE-AWARE"
  }
};

let snapshot = null;

function getPath(source, path) {
  return path.split(".").reduce((value, key) => value?.[key], source);
}

function formatValue(value, kind) {
  const number = Number(value);
  if (kind === "currency") {
    return new Intl.NumberFormat("pt-BR", { style: "currency", currency: "BRL" }).format(number);
  }
  if (kind === "pct") {
    return `${number.toLocaleString("pt-BR", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}%`;
  }
  if (kind === "decimal") {
    return number.toLocaleString("pt-BR", { minimumFractionDigits: 3, maximumFractionDigits: 3 });
  }
  return value?.toLocaleString?.("pt-BR") ?? String(value);
}

function bindSnapshot() {
  document.querySelectorAll("[data-bind]").forEach((element) => {
    const value = getPath(snapshot, element.dataset.bind);
    element.textContent = formatValue(value, element.dataset.format);
  });

  const h = snapshot.historical;
  document.querySelector("#ciMetric").textContent =
    `${formatValue(h.roiCiLow, "pct")} → ${formatValue(h.roiCiHigh, "pct")}`;
  document.querySelector("#snapshotTime").textContent =
    new Intl.DateTimeFormat("pt-BR", { dateStyle: "medium", timeStyle: "short" })
      .format(new Date(snapshot.meta.snapshotAt));
}

function renderMonthlyBars() {
  const container = document.querySelector("#monthlyBars");
  const max = Math.max(...snapshot.monthly.map((item) => Math.abs(item.profit)));
  container.innerHTML = snapshot.monthly.map((item) => {
    const height = Math.max(8, Math.abs(item.profit) / max * 36);
    const className = item.profit < 0 ? "negative-bar" : "";
    return `<i class="${className}" style="height:${height}px" title="${item.month}: ${formatValue(item.profit, "currency")}"></i>`;
  }).join("");
}

function renderEquityChart() {
  const svg = document.querySelector("#equityChart");
  const values = snapshot.equity;
  const width = 800;
  const height = 280;
  const pad = { left: 42, right: 18, top: 18, bottom: 30 };
  const min = Math.floor(Math.min(...values) / 50) * 50;
  const max = Math.ceil(Math.max(...values) / 50) * 50;
  const x = (index) => pad.left + index * (width - pad.left - pad.right) / (values.length - 1);
  const y = (value) => pad.top + (max - value) * (height - pad.top - pad.bottom) / (max - min);
  const points = values.map((value, index) => `${x(index)},${y(value)}`).join(" ");
  const area = `${pad.left},${height - pad.bottom} ${points} ${x(values.length - 1)},${height - pad.bottom}`;
  const ticks = [min, (min + max) / 2, max];

  svg.innerHTML = `
    <defs>
      <linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#1d8f62" stop-opacity=".23"/>
        <stop offset="100%" stop-color="#1d8f62" stop-opacity="0"/>
      </linearGradient>
    </defs>
    ${ticks.map((tick) => `
      <line class="chart-grid-line" x1="${pad.left}" x2="${width - pad.right}" y1="${y(tick)}" y2="${y(tick)}"/>
      <text class="chart-label" x="0" y="${y(tick) + 3}">R$ ${Math.round(tick)}</text>
    `).join("")}
    <polygon class="chart-area" points="${area}"/>
    <polyline class="chart-line" points="${points}"/>
    <circle class="chart-end" cx="${x(values.length - 1)}" cy="${y(values.at(-1))}" r="5"/>
    <text class="chart-label" x="${pad.left}" y="${height - 7}">01 fev</text>
    <text class="chart-label" text-anchor="middle" x="${width / 2}" y="${height - 7}">15 abr</text>
    <text class="chart-label" text-anchor="end" x="${width - pad.right}" y="${height - 7}">23 jul</text>
  `;
}

function renderSignals() {
  document.querySelector("#signalTable").innerHTML = snapshot.demoSignals.map((signal) => `
    <div class="signal-row">
      <span class="signal-time">${signal.time}</span>
      <div class="signal-match"><strong>${signal.match}</strong><small>${signal.league}</small></div>
      <div class="signal-market"><strong>${signal.market}</strong><small>EV +${signal.ev.toLocaleString("pt-BR")}%</small></div>
      <span class="signal-number">${signal.probability.toLocaleString("pt-BR")}%</span>
      <span class="signal-quality">${signal.quality}</span>
    </div>
  `).join("");
}

function renderPipeline() {
  const track = document.querySelector("#pipelineTrack");
  track.innerHTML = snapshot.pipeline.map((node, index) => `
    <button class="pipeline-node" data-node="${node.id}">
      <span class="node-top"><span class="node-index">${String(index + 1).padStart(2, "0")}</span><i class="node-status ${node.status}"></i></span>
      <strong>${node.label}</strong>
      <small>${node.detail}</small>
    </button>
  `).join("");

  track.addEventListener("click", (event) => {
    const button = event.target.closest("[data-node]");
    if (!button) return;
    document.querySelectorAll(".pipeline-node").forEach((node) => node.classList.remove("active"));
    button.classList.add("active");
    const detail = pipelineDetails[button.dataset.node];
    document.querySelector("#pipelineDetail").innerHTML = `
      <span class="panel-kicker">COMPONENTE</span>
      <h2>${detail.title}</h2>
      <p>${detail.text}</p>
      <span class="detail-control">${detail.control}</span>
    `;
  });
}

function renderOperations() {
  document.querySelector("#opsList").innerHTML = snapshot.operations.map((operation) => `
    <div class="op-row"><strong>${operation.job}</strong><span>${operation.frequency}</span><span>${operation.guard}</span></div>
  `).join("");
}

function showView(name, updateHash = true) {
  if (!document.querySelector(`[data-page="${name}"]`)) name = "overview";
  document.querySelectorAll(".view").forEach((view) => view.classList.toggle("active", view.dataset.page === name));
  document.querySelectorAll(".nav-link").forEach((link) => link.classList.toggle("active", link.dataset.view === name));
  document.querySelector(".mobile-nav").classList.remove("open");
  document.querySelector(".menu-toggle").setAttribute("aria-expanded", "false");
  if (updateHash) history.replaceState(null, "", `#${name}`);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function bindNavigation() {
  document.addEventListener("click", (event) => {
    const target = event.target.closest("[data-view], [data-go]");
    if (target) showView(target.dataset.view || target.dataset.go);
  });
  document.querySelector(".menu-toggle").addEventListener("click", (event) => {
    const mobile = document.querySelector(".mobile-nav");
    const open = mobile.classList.toggle("open");
    event.currentTarget.setAttribute("aria-expanded", String(open));
  });
  window.addEventListener("hashchange", () => showView(location.hash.slice(1) || "overview", false));
}

function showError() {
  document.querySelector(".toast").textContent = "Não foi possível carregar o snapshot demonstrativo.";
  document.querySelector(".toast").classList.add("show");
}

async function init() {
  bindNavigation();
  showView(location.hash.slice(1) || "overview", false);
  try {
    const response = await fetch("./data/snapshot.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    snapshot = await response.json();
    bindSnapshot();
    renderMonthlyBars();
    renderEquityChart();
    renderSignals();
    renderPipeline();
    renderOperations();
  } catch (error) {
    console.error(error);
    showError();
  }
}

init();
