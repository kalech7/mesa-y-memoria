"use strict";

const byId = (id) => document.getElementById(id);
const fields = ["agenda", "route", "arm", "scope"];
const publicArms = new Set(["SFT-ramas", "GRPO-continuacion"]);
const meals = { dinner_preference: "Cena", breakfast_preference: "Desayuno", drink_preference: "Bebida" };
const actionLabels = { ASK: "Preguntar", WAIT: "Continuar sin preguntar", READ: "Consultar la memoria" };
const sources = { qwen: "Decisión de Qwen", manual: "Decisión manual", forced: "Presupuesto agotado", reader: "Lector de memoria" };
const formatter = new Intl.DateTimeFormat("es-EC", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
const percent = new Intl.NumberFormat("es-EC", { style: "percent", maximumFractionDigits: 2 });
let catalog;
let current;

function element(tag, value, className) {
  const node = document.createElement(tag);
  if (value != null) node.textContent = value;
  if (className) node.className = className;
  return node;
}

function date(value) {
  return formatter.format(new Date(value + "T12:00:00Z"));
}

function setBusy(busy) {
  byId("controls").disabled = busy || !catalog;
  byId("run").disabled = busy;
  byId("run-form").setAttribute("aria-busy", String(busy));
  const manual = byId("route").value !== "model";
  byId("arm").disabled = manual;
}

function showError(error) {
  byId("problem").textContent = error.message;
  byId("problem").hidden = false;
  byId("request-status").textContent = "Puedes volver a intentarlo.";
}

async function api(path, body) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 180000);
  try {
    const options = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
    const response = await fetch("/api/timing/" + path, { ...options, signal: controller.signal });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "El servicio no pudo completar la solicitud.");
    return data;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("La ejecución superó el tiempo de espera.");
    if (error instanceof TypeError || error instanceof SyntaxError) throw new Error("No se pudo conectar con el laboratorio.");
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

function quotesDetails(quotes, label, open) {
  const details = element("details");
  details.open = open;
  details.append(element("summary", label));
  for (const quote of quotes) {
    const block = element("blockquote");
    block.append(element("p", "«" + quote.text + "»"), element("footer", "Turno " + quote.turn_id, "small"));
    details.append(block);
  }
  return details;
}

function renderRecord(record, visit) {
  const card = element("article", null, "record");
  const head = element("div", null, "meta");
  head.append(element("h4", (meals[record.attribute] || record.attribute) + " · " + record.value),
    element("span", record.valid ? "Vigente" : "Vencido", record.valid ? "current" : "expired"));
  const scope = record.scope;
  const scopeText = scope.kind === "session" ? "Solo durante la visita " + String(scope.session_id).replace(/^S/, "") : "Preferencia habitual";
  const added = visit.operations.some(op => op.record?.memory_id === record.memory_id);
  const meta = [added ? "Guardado en esta visita" : "Guardado en una visita anterior", scopeText];
  if (scope.valid_from) meta.push("desde el " + date(scope.valid_from));
  card.append(head, element("p", meta.join(" · "), "small"), quotesDetails(record.evidence_quotes, "Citas que autorizan el recuerdo", true));
  return card;
}

function renderFuture(rows) {
  byId("future-summary").textContent = rows.filter(row => row.correct).length + " de " + rows.length + " aciertos";
  byId("future").replaceChildren(...rows.map(row => {
    const node = element("article", null, "probe");
    const head = element("div", null, "probe-head");
    head.append(element("h4", (meals[row.attribute] || row.attribute) + " · " + date(row.date)),
      element("span", row.correct ? "Acierto" : "Fallo", row.correct ? "current" : "expired"));
    node.append(head);
    if (row.mode === "ANSWER") {
      node.append(element("p", "Respuesta del lector · " + row.value));
      if (row.quotes.length) node.append(quotesDetails(row.quotes, "Ver evidencia recuperada", false));
    } else {
      const text = row.correct_abstention ? "El lector se abstuvo. No había una preferencia aplicable." :
        "El lector se abstuvo. Faltaba un recuerdo para responder.";
      node.append(element("p", text));
    }
    return node;
  }));
}

function showVisit(index, focus) {
  const visit = current.visits[index];
  const buttons = Array.from(byId("tabs").children);
  for (const [i, button] of buttons.entries()) {
    button.setAttribute("aria-selected", String(i === index));
    button.tabIndex = i === index ? 0 : -1;
  }
  if (focus) buttons[index].focus();
  byId("visit-panel").setAttribute("aria-labelledby", buttons[index].id);
  byId("visit-date").textContent = date(visit.date);
  byId("budget").textContent = "Preguntas disponibles al inicio · " + visit.remaining_before + " / Al cierre · " + visit.remaining_after;
  byId("messages").replaceChildren(...visit.messages.map(message => {
    const node = element("div", null, "message " + message.role);
    node.append(element("span", message.role === "user" ? "Persona" : "Asistente", "role"), element("p", message.text));
    return node;
  }));
  byId("decision").textContent = (sources[visit.source] || visit.source) + " · " + (actionLabels[visit.action] || visit.action);
  byId("record-count").textContent = visit.memory.length === 1 ? "1 recuerdo" : visit.memory.length + " recuerdos";
  byId("records").replaceChildren(...visit.memory.map(record => renderRecord(record, visit)));
  if (!visit.memory.length) byId("records").append(element("p", "Aún no se ha guardado ningún recuerdo en esta historia.", "empty"));
  byId("future-section").hidden = visit.action !== "READ";
  if (visit.action === "READ") renderFuture(current.future);
}

function renderRun(result) {
  current = result;
  const settings = current.settings;
  const agenda = catalog.agendas.find(entry => entry.id === settings.agenda);
  const source = current.model.used ? current.model.label : "Recorrido manual";
  byId("run-info").textContent = agenda.label + " · " + source;
  byId("tabs").replaceChildren(...current.visits.map((visit, i) => {
    const button = element("button", "Visita " + visit.visit);
    button.type = "button";
    button.id = "visit-tab-" + visit.visit;
    button.setAttribute("role", "tab");
    button.setAttribute("aria-controls", "visit-panel");
    button.addEventListener("click", () => showVisit(i, false));
    button.addEventListener("keydown", event => {
      const count = current.visits.length;
      const next = { ArrowRight: (i + 1) % count, ArrowLeft: (i + count - 1) % count, Home: 0, End: count - 1 };
      if (Object.hasOwn(next, event.key)) {
        event.preventDefault();
        showVisit(next[event.key], true);
      }
    });
    return button;
  }));
  showVisit(0, false);
  byId("episode").hidden = false;
}

function renderStudy(results) {
  if (!results?.summary) return;
  const names = Object.fromEntries(catalog.arms.map(entry => [entry.id, entry.label]));
  const table = element("table");
  const header = table.createTHead().insertRow();
  for (const label of ["Política", "Acierto futuro", "Preguntas por historia"]) {
    const cell = element("th", label);
    cell.scope = "col";
    header.append(cell);
  }
  const body = table.createTBody();
  for (const [arm, score] of Object.entries(results.summary)) {
    if (!publicArms.has(arm)) continue;
    const row = body.insertRow();
    const name = element("th", names[arm] || arm);
    name.scope = "row";
    row.append(name, element("td", percent.format(score.future_supported)), element("td", score.questions.toLocaleString("es-EC")));
  }
  byId("study-data").replaceChildren(element("p", "Evaluación reservada de " + results.roots + " historias en " + results.pairs + " pares. El acierto incluye respuestas con respaldo y abstenciones correctas.", "small"), table);
  byId("results-section").hidden = false;
}

async function run() {
  byId("problem").hidden = true;
  byId("episode").hidden = true;
  setBusy(true);
  byId("request-status").textContent = byId("route").value === "model" ? "Ejecutando Qwen en el servidor…" : "Reproduciendo las decisiones seleccionadas…";
  try {
    const body = Object.fromEntries(fields.map(key => [key, byId(key).value]));
    renderRun(await api("run", body));
    byId("request-status").textContent = "Elige una visita para ver qué se guardó. La tercera muestra qué se recuperó.";
    byId("run").textContent = "Volver a ejecutar";
  } catch (error) {
    byId("run").textContent = "Ejecutar prueba";
    showError(error);
  } finally {
    setBusy(false);
  }
}

async function loadLab() {
  byId("problem").hidden = true;
  setBusy(true);
  byId("request-status").textContent = "Conectando con el laboratorio…";
  try {
    catalog = await api("catalog");
    catalog.arms = catalog.arms.filter(entry => publicArms.has(entry.id));
    for (const field of fields) {
      const select = byId(field);
      select.replaceChildren(...catalog[field + "s"].map(item => new Option(item.label ?? String(item), item.id ?? item)));
      select.value = catalog.defaults[field];
    }
    renderStudy(catalog.results);
    await run();
  } catch (error) {
    catalog = null;
    byId("run").textContent = "Reintentar conexión";
    setBusy(false);
    showError(error);
  }
}

byId("run-form").addEventListener("submit", event => {
  event.preventDefault();
  if (!byId("run").disabled) (catalog ? run : loadLab)();
});
byId("controls").addEventListener("change", () => {
  setBusy(false);
  byId("episode").hidden = true;
  byId("problem").hidden = true;
  byId("run").textContent = "Ejecutar prueba";
  byId("request-status").textContent = "Ejecuta la prueba para ver esta configuración.";
});
if (window.location.hash === "#perfiles") {
  window.location.replace("/chat");
} else {
  loadLab();
}
