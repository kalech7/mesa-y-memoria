import {renderComparison, renderContext, phases, phaseIcons} from "./comparison.js";
import {icon} from "./icons.js";
import {renderChatMemoryStudy, renderMeals, renderThesis} from "./study.js?v=8";
import {renderInteractive} from "./interactive.js";
import {connectChat} from "./chat.js?v=8";
import {connectSharedBudget} from "./shared-budget.js";
import {connectProfiles} from "./profiles.js";

for (const slot of document.querySelectorAll("[data-icon]")) slot.append(icon(slot.dataset.icon));

const byId = (id) => document.getElementById(id);
const form = byId("run-form");
const scenario = byId("scenario");
const run = byId("run");
const status = byId("status");
const error = byId("error");
const live = byId("live");
const question = byId("question");
const verifyWrites = byId("verify-writes");
const results = byId("results");
const timeline = byId("timeline");
const download = byId("export");
const guardForm = byId("guard-form");
const guardCase = byId("guard-case");
const guardRun = byId("guard-run");
const tabs = document.querySelector(".workspace-tabs");
let catalog;
let result;
let phase;
let running = false;
let workspace = "models";
let openBudget;
let openProfiles;
const savedResults = new Map();
const savedPhases = new Map();
const completedRequests = new Map();

async function request(path, options) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.error || body.message || `Error HTTP ${response.status}`);
  }
  return body;
}

function mode() {
  return form.elements.mode.value;
}

function selection() {
  if (workspace === "guard") return {scenario_id: guardCase.value};
  const payload = {scenario_id: scenario.value, mode: mode(),
    model_ids: ["sft", "grpo"]};
  if (payload.mode === "live" && question.value.trim()) payload.question = question.value.trim();
  if (payload.mode === "live" && verifyWrites.checked) payload.verify_writes = true;
  return payload;
}

function notePendingChange() {
  if (!result) return;
  const pending = completedRequests.get(workspace) !== JSON.stringify(selection());
  status.textContent = pending ? "Configuración cambiada. Ejecuta la prueba para actualizar el resultado." :
    result.custom ? "Consulta editada · sin puntuación" : "";
}

function syncControls() {
  const selected = catalog?.scenarios.find((item) => item.id === scenario.value);
  if (selected?.question) question.placeholder = selected.question;
  const replay = form.querySelector('[value="replay"]');
  const available = catalog?.live_available && selected?.live_available !== false;
  if (live.checked && !available) replay.checked = true;
  byId("custom").hidden = mode() !== "live";
  byId("live-note").textContent = selected?.live_available === false
    ? "Este caso usa una traza guardada."
    : catalog && !catalog.live_available
      ? "Inferencia local requiere el entorno de investigación."
      : mode() === "live" ? `Dispositivo: ${catalog.device}. Puede tardar unos minutos.` : "";
  scenario.disabled = running || !catalog;
  guardCase.disabled = running || !catalog?.guard_scenarios?.length;
  guardRun.disabled = guardCase.disabled;
  for (const button of tabs.querySelectorAll("button")) button.disabled = running;
  live.disabled = running || !available;
  replay.disabled = running || !catalog;
  question.disabled = running;
  verifyWrites.disabled = running || mode() !== "live";
  run.disabled = running || !selected;
  byId("run-label").textContent = running ? "Comparando…" : mode() === "live"
    ? `Ejecutar en ${catalog?.device?.toUpperCase() || "local"}` : "Ver traza";
  if (catalog && !catalog.live_available && selected?.live_available !== false) {
    byId("live-note").textContent = catalog.live_reason || "La inferencia local requiere el entorno de investigación.";
  }
}

function showEvidence(data) {
  const directed = data.models.some((model) => model.checkpoint === "RL-credit");
  const scores = directed ? [["SFT de partida", "22 %"], ["GRPO dirigido", "34 %"]]
    : [["SFT de partida", "52 %"], ["GRPO terminal", "60 %"], ["SFT adicional", "68 %"]];
  const terms = scores.map(([label, value]) => {
    const entry = document.createElement("div");
    const name = document.createElement("dt");
    const score = document.createElement("dd");
    name.textContent = label;
    score.textContent = value;
    entry.append(name, score);
    return entry;
  });
  byId("evidence-scores").replaceChildren(...terms);
  byId("evidence-note").textContent = directed
    ? "Éxito conjunto estricto en 25 historias. GRPO dirigido, paso 8 seleccionado en validación."
    : "Éxito conjunto en 25 historias. El GRPO revisado seleccionó el paso inicial en validación.";
}

function showPhase(next, focus = false) {
  phase = next;
  savedPhases.set(workspace, phase);
  if (!result) return;
  for (const button of timeline.querySelectorAll("button")) {
    const active = button.dataset.phase === phase;
    button.setAttribute("aria-pressed", String(active));
    if (active && focus) button.focus();
  }
  renderContext(byId("context"), result, phase);
  renderComparison(byId("comparison"), result, phase);
}

function renderStudy(data) {
  const container = byId("study-summary");
  container.hidden = !data.study;
  container.replaceChildren();
  if (!data.study) return;
  const title = document.createElement("h3");
  title.textContent = "Acierto futuro en el estudio";
  const summary = document.createElement("p");
  summary.textContent = `${data.study.source} · ${data.study.roots} historias · ${data.study.episodes} ejecuciones en ${data.study.seeds.length} ${data.study.seeds.length === 1 ? "semilla" : "semillas"}.`;
  const metrics = document.createElement("dl");
  metrics.className = "study-metrics";
  const percent = new Intl.NumberFormat("es-EC", {style: "percent", maximumFractionDigits: 1});
  for (const model of data.models) {
    const entry = document.createElement("div");
    entry.dataset.arm = model.id;
    const label = document.createElement("dt");
    label.textContent = model.label;
    const value = document.createElement("dd");
    value.textContent = percent.format(data.study.metrics[model.id].future_supported.macro);
    entry.append(label, value);
    metrics.append(entry);
  }
  container.append(title, summary, metrics);
}

function display(data) {
  result = data;
  savedResults.set(workspace, data);
  if (workspace === "models") showEvidence(data);
  renderStudy(data);
  byId("scope").dataset.source = data.mode === "live" ? "live" : "archive";
  byId("scope").dataset.mode = data.mode;
  byId("scope").textContent = data.mode === "guard-replay" ? "Reproducción archivada · misma conversación" :
    data.mode === "live" ? ["Inferencia local", data.custom && "consulta editada",
      data.verified_writes && "escrituras verificadas"].filter(Boolean).join(" · ") : "Traza archivada";
  byId("experiment-title").textContent = data.scenario.title;
  byId("case-description").textContent = data.scenario.description || "";
  byId("case-description").hidden = !data.scenario.description;
  timeline.replaceChildren();
  for (const [key, label] of Object.entries(phases)) {
    if (!data.models.some((model) => model.steps.some((step) => step.phase === key))) continue;
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.phase = key;
    button.append(icon(phaseIcons[key]), label);
    timeline.append(button);
  }
  status.textContent = data.custom
    ? "Consulta editada · sin puntuación"
    : "";
  results.hidden = false;
  download.hidden = false;
  showPhase(savedPhases.get(workspace) || (workspace === "guard" ? "reuse" : "select"));
}

function clearResult() {
  result = undefined;
  byId("scope").textContent = workspace === "guard" ? "Verificación de escritura" : "Comparación de checkpoints";
  byId("experiment-title").textContent = "Explora un caso";
  byId("case-description").hidden = true;
  results.hidden = true;
  download.hidden = true;
}

async function compare() {
  if (running || (workspace === "guard" ? !guardCase.value : !scenario.value)) return;
  const inGuard = workspace === "guard";
  const inference = !inGuard && mode() === "live";
  const payload = selection();
  running = true;
  status.dataset.state = "running";
  error.hidden = true;
  byId("results").setAttribute("aria-busy", "true");
  download.disabled = true;
  syncControls();
  status.textContent = inference
    ? "Ejecutando inferencia y abriendo una nueva sesión para cada checkpoint…"
    : "Abriendo la traza archivada…";
  try {
    const data = inGuard ? await request(`/api/guard/${encodeURIComponent(guardCase.value)}`) : await request("/api/compare", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    completedRequests.set(workspace, JSON.stringify(payload));
    display(data);
  } catch (problem) {
    status.textContent = result ? "Se sigue mostrando la última comparación completada." : "";
    error.textContent = problem.message;
    error.hidden = false;
  } finally {
    running = false;
    delete status.dataset.state;
    byId("results").setAttribute("aria-busy", "false");
    download.disabled = false;
    syncControls();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  compare();
});
guardForm.addEventListener("submit", (event) => {
  event.preventDefault();
  compare();
});
guardCase.addEventListener("change", notePendingChange);
tabs.addEventListener("click", (event) => {
  const button = event.target.closest("[data-workspace]");
  if (!button || running || workspace === button.dataset.workspace) return;
  workspace = button.dataset.workspace;
  const inGuard = workspace === "guard";
  const inStudy = workspace === "study";
  const inChat = workspace === "chat";
  const inBudget = workspace === "budget";
  const inProfiles = workspace === "profiles";
  const hash = {profiles: "#perfiles", budget: "#presupuesto", chat: "#conversar", models: "#comparar", study: "#resultados", guard: "#verificar"}[workspace];
  history.replaceState(null, "", hash);
  for (const item of tabs.querySelectorAll("button")) item.setAttribute("aria-pressed", String(item === button));
  document.body.dataset.workspace = workspace;
  form.hidden = workspace !== "models";
  guardForm.hidden = !inGuard;
  byId("configure-title").hidden = inStudy || inChat || inBudget || inProfiles;
  byId("pilot-reference").hidden = workspace !== "models";
  byId("thesis-study").hidden = !inStudy;
  byId("chat").hidden = !inChat;
  byId("shared-budget").hidden = !inBudget;
  byId("profiles").hidden = !inProfiles;
  error.hidden = true;
  status.textContent = "";
  if (inProfiles) {
    clearResult();
    byId("scope").textContent = "Preferencias y vigencia entre visitas";
    byId("scope").dataset.mode = "live";
    byId("experiment-title").textContent = "Conversa y comprueba qué recuerda";
    openProfiles?.();
    return;
  }
  if (inBudget) {
    clearResult();
    byId("scope").textContent = "Historias de desarrollo";
    byId("scope").dataset.mode = "live";
    byId("experiment-title").textContent = "Una pregunta entre dos visitas";
    openBudget?.();
    return;
  }
  if (inChat) {
    clearResult();
    byId("scope").textContent = "Qwen2.5 · Inferencia local";
    byId("scope").dataset.mode = "live";
    byId("experiment-title").textContent = "Conversa con el modelo";
    return;
  }
  if (inStudy) {
    clearResult();
    byId("scope").textContent = "Evaluación exploratoria";
    byId("scope").dataset.mode = "replay";
    byId("experiment-title").textContent = "Resultados del banco reservado";
    const panel = byId("thesis-study");
    renderThesis(panel, catalog.study_results, () => {
      const guardTab = tabs.querySelector('[data-workspace="guard"]');
      guardTab.click();
      guardTab.focus();
    });
    if (catalog.chat_memory_study) renderChatMemoryStudy(panel, catalog.chat_memory_study);
    if (catalog.recommendation_study) renderMeals(panel, catalog.recommendation_study);
    if (catalog.interactive_study) renderInteractive(panel, catalog.interactive_study, request);
    if (catalog.profiles_v5) {
      const profilesResults = document.createElement("button");
      profilesResults.type = "button";
      profilesResults.className = "secondary";
      profilesResults.textContent = "Ver resultados de la corrida de perfiles";
      profilesResults.addEventListener("click", () => {
        tabs.querySelector('[data-workspace="profiles"]').click();
        openProfiles?.(true);
      });
      panel.prepend(profilesResults);
    }
    return;
  }
  const previous = savedResults.get(workspace);
  if (previous) {
    display(previous);
    notePendingChange();
  } else {
    clearResult();
    compare();
  }
});
form.addEventListener("change", (event) => {
  if (event.target === scenario) question.value = "";
  syncControls();
  notePendingChange();
});
question.addEventListener("input", notePendingChange);
timeline.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-phase]");
  if (button) showPhase(button.dataset.phase);
});
timeline.addEventListener("keydown", (event) => {
  const shift = {ArrowLeft: -1, ArrowRight: 1}[event.key];
  if (!shift) return;
  const choices = [...timeline.querySelectorAll("button")];
  const current = choices.indexOf(document.activeElement);
  if (current === -1) return;
  event.preventDefault();
  const next = choices[(current + shift + choices.length) % choices.length];
  showPhase(next.dataset.phase, true);
});
download.addEventListener("click", () => {
  if (!result) return;
  const trace = new Blob([JSON.stringify(result, null, 2)], {type: "application/json"});
  const url = URL.createObjectURL(trace);
  const link = document.createElement("a");
  link.href = url;
  link.download = `memoria-${result.scenario.id}-${result.mode}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

try {
  catalog = await request("/api/catalog");
  for (const item of catalog.scenarios) {
    scenario.add(new Option(item.title, item.id));
  }
  const groups = new Map();
  for (const item of catalog.guard_scenarios || []) {
    if (!groups.has(item.dataset)) {
      const group = document.createElement("optgroup");
      group.label = item.dataset;
      guardCase.append(group);
      groups.set(item.dataset, group);
    }
    groups.get(item.dataset).append(new Option(item.title, item.id));
  }
  tabs.querySelector('[data-workspace="guard"]').hidden = !guardCase.options.length;
  tabs.querySelector('[data-workspace="study"]').hidden = !catalog.study_results;
  syncControls();
  connectChat(catalog, request);
  openBudget = connectSharedBudget(byId("shared-budget"), catalog.shared_budget, request);
  tabs.querySelector('[data-workspace="profiles"]').hidden = !catalog.profiles_v5;
  if (catalog.profiles_v5) openProfiles = connectProfiles(byId("profiles"), request);
  const routes = {"#perfiles": catalog.profiles_v5 ? "profiles" : "chat", "#presupuesto": "budget", "#comparar": "models"};
  const initial = routes[location.hash] || "chat";
  if (workspace === initial) workspace = "";
  tabs.querySelector(`[data-workspace="${initial}"]`).click();
  window.addEventListener("hashchange", () => {
    const next = routes[location.hash];
    if (next) tabs.querySelector(`[data-workspace="${next}"]`).click();
  });
} catch (problem) {
  status.textContent = "";
  error.textContent = `No se pudo cargar el laboratorio. ${problem.message}`;
  error.hidden = false;
}
