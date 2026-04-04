from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from .branding import ACCELERATOR_NAME

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__ACCELERATOR__ Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<style>
body { font-family: Inter, Arial, sans-serif; margin: 0; background: #f6f8fb; color: #12202f; }
header { padding: 28px 36px; background: linear-gradient(135deg, #0f766e, #1f7a8c); color: white; }
main { padding: 24px 28px 40px 28px; display: grid; gap: 20px; }
.panel { background: white; border-radius: 18px; padding: 18px 20px; box-shadow: 0 10px 28px rgba(15, 32, 58, 0.08); }
.cards { display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 14px; }
.card { background: linear-gradient(180deg, #fbfcff, #f2f6fb); border-radius: 16px; padding: 16px; }
.card .label { font-size: 13px; color: #556273; }
.card .value { font-size: 28px; font-weight: 700; margin-top: 6px; }
.row { display: grid; grid-template-columns: 1.1fr 1fr; gap: 18px; }
.controls { display: flex; gap: 12px; margin-bottom: 12px; align-items: center; flex-wrap: wrap; }
select { border: 1px solid #d7dce4; border-radius: 10px; padding: 8px 12px; background: white; }
svg { width: 100%; height: 360px; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { padding: 10px 8px; border-bottom: 1px solid #eef2f7; text-align: left; vertical-align: top; }
th { color: #475569; font-weight: 700; }
.badge { display: inline-block; padding: 4px 8px; border-radius: 999px; background: #e7f7f4; color: #0f766e; font-size: 12px; font-weight: 700; }
.footer { font-size: 12px; color: #667085; }
</style>
</head>
<body>
<header>
  <h1 style="margin:0 0 8px 0;">__ACCELERATOR__ Dashboard</h1>
  <div>Interactive experiment dashboard for end-to-end case studies, secure modes, load sweeps, extended comparisons, and summary tables.</div>
</header>
<main>
  <section class="panel">
    <div class="cards" id="cards"></div>
  </section>
  <section class="panel">
    <div class="controls">
      <label>Case</label>
      <select id="caseSelect"></select>
    </div>
    <div class="row">
      <div>
        <h3 style="margin:4px 0 8px 0;">End-to-End P99</h3>
        <svg id="caseChart"></svg>
      </div>
      <div>
        <h3 style="margin:4px 0 8px 0;">Load Sweep</h3>
        <svg id="loadChart"></svg>
      </div>
    </div>
  </section>
  <section class="panel">
    <div class="row">
      <div>
        <h3 style="margin:4px 0 8px 0;">Stage Breakdown</h3>
        <svg id="stageChart"></svg>
      </div>
      <div>
        <h3 style="margin:4px 0 8px 0;">Case Notes</h3>
        <div class="table-wrap"><table id="manifestTable"></table></div>
      </div>
    </div>
  </section>
  <section class="panel">
    <h3 style="margin:4px 0 8px 0;">Cross-Case Comparison</h3>
    <div class="table-wrap"><table id="compareTable"></table></div>
  </section>
  <section class="footer">Charts are derived from fig16, fig18, fig19, fig20, case_manifest, and table1.</section>
</main>
<script>
const data = __PAYLOAD__;
const palette = new Map([
  ["__ACCELERATOR__", "#D94841"],
  ["Secure-Elastic", "#0F766E"],
  ["Secure-CT", "#2A9D8F"],
  ["Secure-Rand", "#7C6EE6"],
  ["Multi-core", "#495057"],
  ["GPU-Tensor", "#C79500"],
  ["Transformer-ASIC", "#1F7A8C"]
]);
function color(mode) { return palette.get(mode) || "#4F7CAC"; }
const cases = Array.from(new Set(data.fig16.map(d => d.case)));
const caseSelect = d3.select("#caseSelect");
caseSelect.selectAll("option").data(cases).enter().append("option").attr("value", d => d).text(d => d);
function summaryCards() {
  const summary = data.table1;
  const getValue = (category, metric, fallback) => {
    const row = summary.find(d => d.category === category && d.metric === metric);
    if (!row) return fallback;
    if (typeof row.value === "number") return row.value.toFixed(2);
    return row.value;
  };
  const cards = [
    { label: "Best Security Score", value: d3.max(data.fig20, d => +d.security_score || 0).toFixed(2) },
    { label: "Secure-Elastic Mean Latency Gain", value: getValue("compare", "mean_latency_gain_vs_multicore", "-") },
    { label: "Case Count", value: String(cases.length) },
    { label: "Load Sweep Points", value: String(data.fig19.length) }
  ];
  const root = d3.select("#cards");
  root.selectAll("div.card").data(cards).enter().append("div").attr("class", "card").html(d => `<div class="label">${d.label}</div><div class="value">${d.value}</div>`);
}
function renderCaseChart(caseName) {
  const rows = data.fig16.filter(d => d.case === caseName);
  const svg = d3.select("#caseChart");
  svg.selectAll("*").remove();
  const width = svg.node().clientWidth;
  const height = svg.node().clientHeight;
  const margin = {top: 22, right: 20, bottom: 70, left: 54};
  const x = d3.scaleBand().domain(rows.map(d => d.mode)).range([margin.left, width - margin.right]).padding(0.22);
  const y = d3.scaleLinear().domain([0, d3.max(rows, d => +d.p99_ms) * 1.15]).range([height - margin.bottom, margin.top]);
  svg.append("g").attr("transform", `translate(0,${height - margin.bottom})`).call(d3.axisBottom(x)).selectAll("text").attr("transform", "rotate(-20)").style("text-anchor", "end");
  svg.append("g").attr("transform", `translate(${margin.left},0)`).call(d3.axisLeft(y));
  svg.selectAll("rect.bar").data(rows).enter().append("rect").attr("class", "bar").attr("x", d => x(d.mode)).attr("y", d => y(+d.p99_ms)).attr("width", x.bandwidth()).attr("height", d => y(0) - y(+d.p99_ms)).attr("rx", 8).attr("fill", d => color(d.mode));
  svg.selectAll("text.value").data(rows).enter().append("text").attr("x", d => x(d.mode) + x.bandwidth() / 2).attr("y", d => y(+d.p99_ms) - 6).attr("text-anchor", "middle").style("font-size", "11px").text(d => (+d.p99_ms).toFixed(1));
}
function renderLoadChart(caseName) {
  const rows = data.fig19.filter(d => d.case === caseName);
  const modes = Array.from(new Set(rows.map(d => d.mode)));
  const svg = d3.select("#loadChart");
  svg.selectAll("*").remove();
  const width = svg.node().clientWidth;
  const height = svg.node().clientHeight;
  const margin = {top: 20, right: 24, bottom: 48, left: 52};
  const x = d3.scaleLinear().domain(d3.extent(rows, d => +d.load_factor)).range([margin.left, width - margin.right]);
  const y = d3.scaleLinear().domain([0, d3.max(rows, d => +d.p99_ms) * 1.12]).range([height - margin.bottom, margin.top]);
  const line = d3.line().x(d => x(+d.load_factor)).y(d => y(+d.p99_ms));
  svg.append("g").attr("transform", `translate(0,${height - margin.bottom})`).call(d3.axisBottom(x).ticks(4));
  svg.append("g").attr("transform", `translate(${margin.left},0)`).call(d3.axisLeft(y));
  modes.forEach(mode => {
    const part = rows.filter(d => d.mode === mode).sort((a, b) => +a.load_factor - +b.load_factor);
    svg.append("path").datum(part).attr("fill", "none").attr("stroke", color(mode)).attr("stroke-width", 2.4).attr("d", line);
    svg.selectAll(`circle.${mode.replace(/[^a-zA-Z0-9]/g, '')}`).data(part).enter().append("circle").attr("cx", d => x(+d.load_factor)).attr("cy", d => y(+d.p99_ms)).attr("r", 4).attr("fill", color(mode));
  });
}
function renderStageChart(caseName) {
  const rows = data.fig18.filter(d => d.case === caseName && ["__ACCELERATOR__", "Secure-Elastic", "Multi-core"].includes(d.mode));
  const modes = Array.from(new Set(rows.map(d => d.mode)));
  const stages = Array.from(new Set(rows.map(d => d.stage)));
  const stackData = modes.map(mode => {
    const part = rows.filter(d => d.mode === mode);
    const record = { mode };
    stages.forEach(stage => {
      const row = part.find(d => d.stage === stage);
      record[stage] = row ? +row.share : 0;
    });
    return record;
  });
  const stack = d3.stack().keys(stages);
  const series = stack(stackData);
  const svg = d3.select("#stageChart");
  svg.selectAll("*").remove();
  const width = svg.node().clientWidth;
  const height = svg.node().clientHeight;
  const margin = {top: 20, right: 20, bottom: 42, left: 94};
  const x = d3.scaleLinear().domain([0, 1]).range([margin.left, width - margin.right]);
  const y = d3.scaleBand().domain(modes).range([margin.top, height - margin.bottom]).padding(0.22);
  svg.append("g").attr("transform", `translate(0,${height - margin.bottom})`).call(d3.axisBottom(x).ticks(5, ".0%"));
  svg.append("g").attr("transform", `translate(${margin.left},0)`).call(d3.axisLeft(y));
  series.forEach((layer, idx) => {
    svg.selectAll(`rect.layer${idx}`).data(layer).enter().append("rect").attr("x", d => x(d[0])).attr("y", d => y(d.data.mode)).attr("width", d => x(d[1]) - x(d[0])).attr("height", y.bandwidth()).attr("fill", d3.interpolateTurbo(idx / Math.max(1, stages.length - 1)));
  });
  const legend = svg.append("g").attr("transform", `translate(${margin.left},${margin.top - 6})`);
  stages.forEach((stage, idx) => {
    const g = legend.append("g").attr("transform", `translate(${idx * 105},0)`);
    g.append("rect").attr("width", 12).attr("height", 12).attr("rx", 3).attr("fill", d3.interpolateTurbo(idx / Math.max(1, stages.length - 1)));
    g.append("text").attr("x", 18).attr("y", 10).style("font-size", "12px").text(stage);
  });
}
function renderManifest(caseName) {
  const row = data.manifest.find(d => d.case === caseName);
  const table = d3.select("#manifestTable");
  table.selectAll("*").remove();
  const entries = [
    ["Case", row.case],
    ["Stage Mix", row.stages],
    ["SLO", row.slo_ms + " ms"],
    ["Focus", row.focus],
    ["Storyline", row.storyline],
    ["Prompt", row.prompt]
  ];
  table.append("tbody").selectAll("tr").data(entries).enter().append("tr").html(d => `<th>${d[0]}</th><td>${d[1]}</td>`);
}
function renderCompareTable() {
  const grouped = d3.rollups(data.fig20, v => ({
    latency: d3.mean(v, d => +d.latency_gain_vs_multicore),
    energy: d3.mean(v, d => +d.energy_gain_vs_multicore),
    throughput: d3.mean(v, d => +d.throughput_gain_vs_multicore),
    security: d3.mean(v, d => +d.security_score)
  }), d => d.mode).map(([mode, stats]) => ({ mode, ...stats })).sort((a, b) => d3.descending(a.latency, b.latency));
  const table = d3.select("#compareTable");
  table.selectAll("*").remove();
  table.append("thead").html("<tr><th>Mode</th><th>Latency Gain</th><th>Energy Gain</th><th>Throughput Gain</th><th>Security</th></tr>");
  table.append("tbody").selectAll("tr").data(grouped).enter().append("tr").html(d => `
    <td><span class="badge">${d.mode}</span></td>
    <td>${d.latency.toFixed(2)}</td>
    <td>${d.energy.toFixed(2)}</td>
    <td>${d.throughput.toFixed(2)}</td>
    <td>${d.security.toFixed(2)}</td>
  `);
}
function renderAll(caseName) {
  renderCaseChart(caseName);
  renderLoadChart(caseName);
  renderStageChart(caseName);
  renderManifest(caseName);
}
summaryCards();
renderCompareTable();
renderAll(cases[0]);
caseSelect.on("change", event => renderAll(event.target.value));
</script>
</body>
</html>
"""


def export_dashboard(results_dir: Path, output_html: Path) -> Path:
    results_dir = Path(results_dir)
    output_html = Path(output_html)
    data = {
        'fig16': pd.read_csv(results_dir / 'fig16_case_studies.csv').to_dict(orient='records'),
        'fig18': pd.read_csv(results_dir / 'fig18_service_breakdown.csv').to_dict(orient='records'),
        'fig19': pd.read_csv(results_dir / 'fig19_load_sweep.csv').to_dict(orient='records'),
        'fig20': pd.read_csv(results_dir / 'fig20_extended_compare.csv').to_dict(orient='records'),
        'manifest': pd.read_csv(results_dir / 'case_manifest.csv').to_dict(orient='records'),
        'table1': pd.read_csv(results_dir / 'table1_sosp_summary.csv').to_dict(orient='records'),
    }
    payload = json.dumps(data, ensure_ascii=False)
    html = HTML_TEMPLATE.replace('__PAYLOAD__', payload).replace('__ACCELERATOR__', ACCELERATOR_NAME)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding='utf-8')
    return output_html
