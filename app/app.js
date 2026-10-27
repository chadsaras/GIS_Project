// Demo website: three steps over precomputed, out-of-sample results (data/data.js from scripts/22_app_data.py).
"use strict";

const STATES = window.APP_DATA.states;
const BREAKS = [12000, 16000, 20000, 25000, 32000, 45000, 70000];  // R$ per person, 2022
const PALETTE = ["#fff5e6", "#fde3b8", "#fcc98a", "#f9a65a", "#ef7d3c", "#d9542b", "#b0331f", "#7a1f14"];
const RISK = { low: "#2e8b57", medium: "#e19a1f", high: "#c4493b" };
const RISK_TEXT = {
  low: "Low risk: the range is tight and usually right.",
  medium: "Medium risk: a wider range.",
  high: "High risk: a wide range. Mining, industry and big cities often look different from space than their GDP says.",
};

const $ = (s, el = document) => el.querySelector(s);
const fmt = (v) => "R$ " + Math.round(v).toLocaleString("en-US");
const short = (v) => v >= 1e6 ? (v / 1e6).toFixed(1) + "M" : v >= 1e3 ? Math.round(v / 1e3) + "k" : String(v);
const factor = (a, b) => Math.max(a / b, b / a);
const colorValue = (v) => PALETTE[BREAKS.findIndex((b) => v < b) === -1 ? BREAKS.length : BREAKS.findIndex((b) => v < b)];
const colorRank = (p) => PALETTE[Math.min(PALETTE.length - 1, Math.floor(p * PALETTE.length))];

function percentiles(rows, key) {
  const order = rows.map((r, i) => [r[key], i]).sort((a, b) => a[0] - b[0]);
  const out = new Array(rows.length);
  order.forEach(([, i], k) => { out[i] = k / (rows.length - 1); });
  return out;
}

function legendValue(el) {
  el.innerHTML = PALETTE.map((c, i) => `<i style="background:${c}"></i>` +
    (i < BREAKS.length ? `<span class="lab">${short(BREAKS[i])}</span>` : "")).join("") +
    '<span class="lab">R$ per person</span>';
}
function legendRisk(el) {
  el.innerHTML = Object.entries(RISK).map(([k, c]) => `<i style="background:${c}"></i><span class="lab">${k} risk</span>`).join("");
}
function legendRank(el) {
  el.innerHTML = PALETTE.map((c) => `<i style="background:${c}"></i>`).join("") +
    '<span class="lab">poorest → richest within the state (rank)</span>';
}

// A choropleth of one state; fill(row) gives the colour, onPick(row) handles clicks.
function stateMap(el, state, fill, onPick, onHover) {
  const byCode = new Map(state.rows.map((r) => [r.code, r]));
  const map = L.map(el, { attributionControl: false, zoomSnap: 0.25, scrollWheelZoom: false });
  let selected = null;
  const style = (f) => {
    const r = byCode.get(f.properties.muni_code);
    const sel = selected === f.properties.muni_code;
    return { fillColor: r ? fill(r) : "#ccc", fillOpacity: 0.92, color: sel ? "#1e2433" : "#ffffff",
             weight: sel ? 3 : 0.4 };
  };
  const layer = L.geoJSON(state.shapes, {
    style,
    onEachFeature: (f, lyr) => {
      const r = byCode.get(f.properties.muni_code);
      if (!r) return;
      lyr.bindTooltip(r.name, { sticky: true, direction: "top", opacity: 0.9 });
      lyr.on("click", () => onPick && onPick(r));
      lyr.on("mouseover", () => onHover && onHover(r.code));
      lyr.on("mouseout", () => onHover && onHover(null));
    },
  }).addTo(map);
  map.fitBounds(layer.getBounds(), { padding: [6, 6] });
  const layers = new Map();
  layer.eachLayer((l) => layers.set(l.feature.properties.muni_code, l));
  return {
    map, byCode,
    restyle() { layer.setStyle(style); },
    select(code, zoom) {
      selected = code;
      layer.setStyle(style);
      const l = layers.get(code);
      if (l) { l.bringToFront(); if (zoom) map.fitBounds(l.getBounds(), { maxZoom: 8.5, padding: [60, 60] }); }
    },
    highlight(code) {
      layer.setStyle(style);
      const l = layers.get(code);
      if (l) { l.setStyle({ color: "#1e2433", weight: 2.5 }); l.bringToFront(); }
    },
  };
}

/* ---------------- Step 1: guess the town ---------------- */
const MG = STATES.MG;
let mode = "est";
const revealed = new Map();  // code -> {inside, satWins}
const fill1 = (r) => mode === "risk" ? RISK[r.risk] : colorValue(mode === "official" ? r.official : r.est);
const m1 = stateMap($("#map-mg"), MG, fill1, (r) => pick(r));
legendValue($("#legend-mg"));

document.querySelectorAll("#mg-mode button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll("#mg-mode button").forEach((x) => x.classList.toggle("on", x === b));
  mode = b.dataset.mode;
  (mode === "risk" ? legendRisk : legendValue)($("#legend-mg"));
  m1.restyle();
}));
document.querySelectorAll(".presets button").forEach((b) => b.addEventListener("click", () => {
  const r = m1.byCode.get(Number(b.dataset.code));
  if (r) pick(r, true);
}));

function axis(r) {
  const vals = [r.lo, r.hi, r.est, r.ntl, r.official];
  const lo = Math.log(Math.min(...vals) / 1.35), hi = Math.log(Math.max(...vals) * 1.35);
  const x = (v) => (100 * (Math.log(v) - lo) / (hi - lo)).toFixed(2) + "%";
  const ticks = [5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6].filter((t) => Math.log(t) > lo && Math.log(t) < hi);
  return { x, ticks };
}

function pick(r, zoom = false) {
  m1.select(r.code, zoom);
  const { x, ticks } = axis(r);
  const done = revealed.has(r.code);
  $("#card").innerHTML = `
    <p class="eyebrow">Minas Gerais · ${r.pop.toLocaleString("en-US")} people</p>
    <h3>${r.name}</h3>
    <span class="badge ${r.risk}">${r.risk} risk</span>
    <p class="big">${fmt(r.est)} <small>per person, satellite estimate</small></p>
    <div class="bar">
      <div class="axis"></div>
      <div class="range" style="left:${x(r.lo)};width:calc(${x(r.hi)} - ${x(r.lo)})"></div>
      <div class="pt ntl" style="left:${x(r.ntl)}" title="Night lights only: ${fmt(r.ntl)}"></div>
      <div class="pt est" style="left:${x(r.est)}" title="Satellite estimate: ${fmt(r.est)}"></div>
      <div class="pt off" id="off" style="left:${done ? x(r.official) : x(r.est)};opacity:${done ? 1 : 0}"></div>
      ${ticks.map((t) => `<span class="tick" style="left:${x(t)}">${short(t)}</span>`).join("")}
    </div>
    <div class="keys">
      <span><i style="background:var(--teal)"></i>satellite estimate</span>
      <span><i style="border:2px solid var(--teal);background:rgba(19,138,138,.18)"></i>90% range ${fmt(r.lo)}–${fmt(r.hi)}</span>
      <span><i style="background:#9aa1ad"></i>night lights only ${fmt(r.ntl)}</span>
    </div>
    <p class="muted" style="font-size:14px;margin:10px 0 0">${RISK_TEXT[r.risk]}</p>
    <button class="reveal" id="reveal" ${done ? "disabled" : ""}>${done ? "Revealed" : "Reveal the official value"}</button>
    <div id="verdict">${done ? verdict(r) : ""}</div>`;
  if (!done) $("#reveal").addEventListener("click", () => reveal(r, x));
}

function verdict(r) {
  const inside = r.official >= r.lo && r.official <= r.hi;
  const fs = factor(r.est, r.official), fn = factor(r.ntl, r.official);
  return `<div class="verdict">
      Official: <b>${fmt(r.official)}</b> per person ·
      <b class="${inside ? "in" : "out"}">${inside ? "inside the range" : "outside the range"}</b><br>
      Satellite off by ×${fs.toFixed(2)}, night lights alone off by ×${fn.toFixed(2)}
      ${fs < fn ? " · satellite closer" : " · night lights closer"}
    </div>${r.why ? `<p class="why"><b>Likely reason:</b> ${r.why}</p>` : ""}`;
}

function reveal(r, x) {
  const off = $("#off");
  off.style.opacity = 1;
  requestAnimationFrame(() => { off.style.left = x(r.official); });
  const inside = r.official >= r.lo && r.official <= r.hi;
  revealed.set(r.code, { inside, satWins: factor(r.est, r.official) < factor(r.ntl, r.official) });
  setTimeout(() => { $("#verdict").innerHTML = verdict(r); }, 650);
  const b = $("#reveal");
  b.disabled = true; b.textContent = "Revealed";
  const n = revealed.size, vals = [...revealed.values()];
  $("#s-n").textContent = n;
  $("#s-in").textContent = `${vals.filter((v) => v.inside).length} / ${n}`;
  $("#s-win").textContent = `${vals.filter((v) => v.satWins).length} / ${n}`;
}

/* ---------------- Step 2: transfer ---------------- */
function stats(rows, key) {
  const y = rows.map((r) => Math.log(r.official)), p = rows.map((r) => Math.log(r[key]));
  const my = y.reduce((a, b) => a + b, 0) / y.length;
  const r2 = 1 - y.reduce((s, v, i) => s + (v - p[i]) ** 2, 0) / y.reduce((s, v) => s + (v - my) ** 2, 0);
  const ry = percentiles(rows, "official"), rp = percentiles(rows, key);
  const mr = 0.5, num = ry.reduce((s, v, i) => s + (v - mr) * (rp[i] - mr), 0);
  const den = Math.sqrt(ry.reduce((s, v) => s + (v - mr) ** 2, 0) * rp.reduce((s, v) => s + (v - mr) ** 2, 0));
  const within = rows.filter((r) => factor(r[key], r.official) <= 1.5).length / rows.length;
  const inside = rows.filter((r) => r.official >= r.lo && r.official <= r.hi).length / rows.length;
  return { r2, rho: num / den, within, inside };
}

function transfer(code) {
  const S = STATES[code];
  const body = $("#transfer-body");
  if (!S) {
    body.innerHTML = `<div class="pending"><p><b>Coming next.</b> The São Paulo transfer test (645 municipalities)
      runs as soon as its satellite data is downloaded. The models trained on Minas Gerais are applied unchanged and the
      result is compared with São Paulo's official GDP, including whether the error ranges and risk labels still hold.</p></div>`;
    return;
  }
  let view = "value", model = "est";
  body.innerHTML = `
    <div class="panel">
      <div class="toolbar">
        <div class="seg" id="t-view"><button data-v="value" class="on">Values (R$)</button><button data-v="rank">Ranks within state</button></div>
        <div class="seg" id="t-model"><button data-m="est" class="on">Satellite model</button><button data-m="ntl">Night lights only</button></div>
      </div>
      <div class="two">
        <div><p class="eyebrow">Predicted · trained only on Minas Gerais</p><div id="map-t1" class="map small"></div></div>
        <div><p class="eyebrow">Official · IBGE 2022</p><div id="map-t2" class="map small"></div></div>
      </div>
      <div class="legend" id="legend-t"></div>
      <div class="metrics" id="t-metrics"></div>
    </div>
    <div class="grid" style="margin-top:16px">
      <div class="panel scatter"><p class="eyebrow">Each dot is a municipality · hover to find it on the maps</p><div id="t-scatter"></div></div>
      <div class="panel" id="t-card"><p class="muted">Hover or click a municipality.</p></div>
    </div>`;
  const pred = { est: percentiles(S.rows, "est"), ntl: percentiles(S.rows, "ntl") }, offp = percentiles(S.rows, "official");
  const idx = new Map(S.rows.map((r, i) => [r.code, i]));
  const f1 = (r) => view === "rank" ? colorRank(pred[model][idx.get(r.code)]) : colorValue(r[model]);
  const f2 = (r) => view === "rank" ? colorRank(offp[idx.get(r.code)]) : colorValue(r.official);
  const card = (r) => {
    if (!r) return;
    $("#t-card").innerHTML = `<p class="eyebrow">${S.name} · ${r.pop.toLocaleString("en-US")} people</p><h3>${r.name}</h3>
      <span class="badge ${r.risk}">${r.risk} risk</span>
      <p style="margin:10px 0 0">Satellite model <b>${fmt(r.est)}</b> · night lights <b>${fmt(r.ntl)}</b><br>
      Official <b>${fmt(r.official)}</b> · satellite off by ×${factor(r.est, r.official).toFixed(2)}</p>`;
  };
  const hover = (c) => { [a, b].forEach((m) => m.highlight(c)); dots(c); if (c) card(a.byCode.get(c)); };
  const a = stateMap($("#map-t1"), S, f1, (r) => card(r), hover);
  const b = stateMap($("#map-t2"), S, f2, (r) => card(r), hover);
  let syncing = false;
  const sync = (from, to) => from.map.on("move", () => {
    if (syncing) return; syncing = true; to.map.setView(from.map.getCenter(), from.map.getZoom(), { animate: false }); syncing = false;
  });
  sync(a, b); sync(b, a);

  const W = 520, H = 380, P = 46;
  const all = S.rows.flatMap((r) => [r.est, r.ntl, r.official]);
  const lo = Math.log(Math.min(...all) / 1.1), hi = Math.log(Math.max(...all) * 1.1);
  const sx = (v) => P + (W - P - 10) * (Math.log(v) - lo) / (hi - lo), sy = (v) => H - P - (H - P - 10) * (Math.log(v) - lo) / (hi - lo);
  const ticks = [5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5].filter((t) => Math.log(t) > lo && Math.log(t) < hi);
  function drawScatter() {
    $("#t-scatter").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="predicted vs official">
      <line x1="${sx(Math.exp(lo))}" y1="${sy(Math.exp(lo))}" x2="${sx(Math.exp(hi))}" y2="${sy(Math.exp(hi))}" stroke="#1e2433" stroke-dasharray="4 4"/>
      ${ticks.map((t) => `<text x="${sx(t)}" y="${H - P + 16}" font-size="11" fill="#687082" text-anchor="middle">${short(t)}</text>
        <text x="${P - 6}" y="${sy(t) + 4}" font-size="11" fill="#687082" text-anchor="end">${short(t)}</text>`).join("")}
      <text x="${(W + P) / 2}" y="${H - 8}" font-size="12" fill="#687082" text-anchor="middle">predicted, R$ per person (${model === "est" ? "satellite model" : "night lights"})</text>
      <text x="12" y="${(H - P) / 2}" font-size="12" fill="#687082" text-anchor="middle" transform="rotate(-90 12 ${(H - P) / 2})">official</text>
      ${S.rows.map((r) => `<circle data-c="${r.code}" cx="${sx(r[model]).toFixed(1)}" cy="${sy(r.official).toFixed(1)}" r="3.6"/>`).join("")}
    </svg>`;
    $("#t-scatter").querySelectorAll("circle").forEach((c) => {
      c.addEventListener("mouseenter", () => hover(Number(c.dataset.c)));
      c.addEventListener("mouseleave", () => hover(null));
    });
  }
  function dots(code) {
    $("#t-scatter").querySelectorAll("circle").forEach((c) => c.classList.toggle("hl", Number(c.dataset.c) === code));
  }
  function refresh() {
    a.restyle(); b.restyle(); drawScatter();
    (view === "rank" ? legendRank : legendValue)($("#legend-t"));
    const s = stats(S.rows, model);
    $("#t-metrics").innerHTML = `
      <div><b>${s.r2.toFixed(2)}</b><span>R² against official (log)</span></div>
      <div><b>${s.rho.toFixed(2)}</b><span>rank agreement (Spearman)</span></div>
      <div><b>${Math.round(100 * s.within)}%</b><span>within ×1.5 of official</span></div>
      ${model === "est" ? `<div><b>${Math.round(100 * s.inside)}%</b><span>inside the Minas Gerais 90% range</span></div>` : ""}`;
  }
  body.querySelectorAll("#t-view button").forEach((x) => x.addEventListener("click", () => {
    body.querySelectorAll("#t-view button").forEach((y) => y.classList.toggle("on", y === x)); view = x.dataset.v; refresh();
  }));
  body.querySelectorAll("#t-model button").forEach((x) => x.addEventListener("click", () => {
    body.querySelectorAll("#t-model button").forEach((y) => y.classList.toggle("on", y === x)); model = x.dataset.m; refresh();
  }));
  refresh();
}
transfer("SP");

/* ---------------- Step 3: the largest misses ---------------- */
const misses = MG.rows.filter((r) => r.why)
  .sort((p, q) => Math.abs(Math.log(q.est / q.official)) - Math.abs(Math.log(p.est / p.official)));
$("#misses").innerHTML = misses.map((r) => {
  const under = r.est < r.official;
  return `<li data-c="${r.code}"><b>${r.name}</b> · <span class="g ${under ? "under" : "over"}">${under ? "under" : "over"}-estimated ×${factor(r.est, r.official).toFixed(1)}</span>
    <span class="badge ${r.risk}" style="margin-left:4px">${r.risk}</span><small>${r.why}</small></li>`;
}).join("");
$("#misses").querySelectorAll("li").forEach((li) => li.addEventListener("click", () => {
  $("#guess").scrollIntoView({ behavior: "smooth" });
  pick(m1.byCode.get(Number(li.dataset.c)), true);
}));

// ?town=<IBGE code> opens that town in step 1 (and &reveal shows the answer), e.g. for links on a slide
const params = new URLSearchParams(location.search);
const start = m1.byCode.get(Number(params.get("town")));
if (start) {
  pick(start, true);
  if (params.has("reveal")) $("#reveal").click();
}
