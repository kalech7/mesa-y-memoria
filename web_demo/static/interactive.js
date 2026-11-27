import {element, table} from "./study.js";

const percent = new Intl.NumberFormat("es-EC", {style: "percent", maximumFractionDigits: 1});
const decimal = new Intl.NumberFormat("es-EC", {maximumFractionDigits: 2});
const names = {"GRPO-longitudinal": "GRPO longitudinal", "GRPO-inmediato": "GRPO inmediato",
  SFT: "SFT", "SFT-adicional": "SFT adicional", "Reglas-observables": "Reglas observables",
  "GRPO-ramificado": "GRPO por ramas", "SFT-por-retorno": "SFT con las mejores ramas",
  "Continuar-siempre": "Continuar siempre", "GRPO-longitudinal-terminal": "GRPO longitudinal terminal",
  "GRPO-inmediato-terminal": "GRPO inmediato terminal"};
const scopes = {ongoing: "Habitual", interval: "Temporal", session: "De esta conversación",
  contextual: "Por contexto", unknown: "Alcance pendiente"};
const actions = {PROCEED: "Responder y continuar", TEMPORAL_SCOPE: "Preguntar si el cambio durará",
  CURRENT_CONTEXT: "Preguntar por el lugar", VALID_UNTIL: "Preguntar hasta cuándo", GENERIC: "Pedir más detalles"};
const queries = {reuse: "Reutilización del recuerdo", unrelated_preference: "Otra preferencia",
  last_day_of_interval: "Último día de la excepción", first_day_after_interval: "Después del vencimiento"};
const replies = {partial: "Respuesta parcial", unavailable: "Sin aclaración", irrelevant: "Respuesta sin el dato", redundant: "Pregunta redundante"};
const points = value => `${value > 0 ? "+" : ""}${decimal.format(100 * value)} pp`;

function branchComparison(report) {
  const section = element("section", null, "study-summary");
  const heading = element("h3", "GRPO por ramas en historias nuevas");
  const result = table("Políticas seleccionadas para el banco nuevo",
    ["Método", "Actual", "Futuro", "Preguntas por episodio"], Object.entries(report.means).map(([method, m]) =>
      [names[method], percent.format(m.current_supported), percent.format(m.future_supported), decimal.format(m.questions)]));
  const difference = report.contrast;
  const description = element("p",
    `${difference.independent_roots} historias reservadas · ${report.seeds.length} semillas. Diferencia de acierto futuro frente a SFT adicional de ${points(difference.estimate)} · IC 95 % de ${points(difference.interval[0])} a ${points(difference.interval[1])}.`,
    "study-meta");
  const conclusion = element("p", report.primary_success ? "La diferencia cumplió el criterio de superioridad."
    : "La comparación no estableció superioridad en acierto futuro.", "study-meta");
  section.append(heading, result, description, conclusion);
  return section;
}

function messagesList(messages) {
  const list = element("ul", null, "dialogue");
  for (const message of messages) {
    list.append(element("li", `${message.role === "assistant" ? "Asistente" : "Usuario"}. ${message.text}`));
  }
  return list;
}

function memoryDetail(records, label = "Memoria después de la conversación") {
  const node = element("details");
  const list = element("ul", null, "memory");
  for (const record of records) {
    const scope = record.scope || {};
    const context = typeof scope.context === "object" && scope.context ? JSON.stringify(scope.context) : scope.context;
    const note = [scopes[scope.kind], context, scope.valid_from && `desde ${scope.valid_from}`,
      scope.valid_until_exclusive && `vence el ${scope.valid_until_exclusive}`].filter(Boolean).join(" · ");
    const row = element("li");
    row.append(element("strong", record.value));
    if (note) row.append(element("span", note, "record-scope"));
    for (const evidence of record.evidence_quotes || []) {
      row.append(element("span", `«${evidence.text}»`, "record-scope"));
    }
    list.append(row);
  }
  node.append(element("summary", `${label} (${records.length})`), list);
  return node;
}

function traceCard(method, trace) {
  const card = element("article", null, "model interaction-card");
  const head = element("header", null, "model-head");
  head.append(element("h3", names[method] || method));
  const body = element("div", null, "model-content");
  trace.visits.forEach((visit, index) => {
    const section = element("section", null, "visit");
    section.append(element("h4", [`Conversación ${index + 1}`, visit.query?.date].filter(Boolean).join(" · ")));
    const dialogue = messagesList(visit.opening_messages || []);
    for (const event of visit.events || []) {
      const question = typeof event.question === "string" ? event.question : event.question?.text;
      const reply = typeof event.reply === "string" ? event.reply : event.reply?.text;
      if (question) dialogue.append(element("li", `Asistente. ${question}`));
      if (reply) {
        const item = element("li", `Usuario. ${reply}`);
        if (replies[event.delivery]) item.append(element("span", replies[event.delivery], "record-scope"));
        dialogue.append(item);
      }
    }
    section.append(dialogue);
    const response = visit.outcome?.response || visit.response || visit.outcome;
    if (response?.value) section.append(element("p", `Respuesta: ${response.value}`));
    section.append(element("p", visit.current_supported ? "Respuesta sustentada" : "Sin acierto sustentado"));
    if (Array.isArray(visit.memory_after)) section.append(memoryDetail(visit.memory_after));
    body.append(section);
  });
  const future = element("section", null, "visit");
  future.append(element("h4", "Consultas posteriores"));
  const checks = element("ul", null, "memory");
  for (const item of trace.future) {
    const value = item.response?.value || "sin respuesta única";
    checks.append(element("li", `${queries[item.kind] || "Consulta"} · ${item.query.date}: ${value} · ${item.supported ? "acierto sustentado" : "sin acierto sustentado"}`));
  }
  future.append(checks);
  body.append(future);
  card.append(head, body);
  return card;
}

function simulator(select, request) {
  const box = element("section", null, "interactive-controls");
  const start = element("button", "Tomar las decisiones en este caso", "secondary");
  start.type = "button";
  const content = element("div");
  const status = element("p", null, "status");
  status.setAttribute("role", "status");
  let history = [];
  let requestId = 0;

  async function advance(next) {
    const id = ++requestId;
    const pending = next ? [...history, next] : [];
    const buttons = [...box.querySelectorAll("button")];
    buttons.forEach(button => { button.disabled = true; });
    status.textContent = "Ejecutando la conversación…";
    try {
      const result = await request("/api/interactive", {method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({case_id: select.value, actions: pending})});
      if (id !== requestId) return;
      history = pending;
      start.textContent = "Reiniciar mi conversación";
      if (result.done) {
        content.replaceChildren(traceCard("Mi recorrido", result));
      } else {
        const obs = result.observation;
        const section = element("div", null, "action-options");
        for (const action of result.legal_actions) {
          const button = element("button", actions[action], "secondary");
          button.type = "button";
          button.addEventListener("click", () => advance(action));
          section.append(button);
        }
        content.replaceChildren(element("h4", `Conversación ${obs.visit} · ${obs.query.date}`),
          messagesList(obs.user_messages), memoryDetail(obs.memory, "Memoria disponible"), section);
        section.querySelector("button")?.focus();
      }
      status.textContent = result.done ? "Se completó la consulta posterior con memoria persistida." : "";
    } catch (error) {
      if (id === requestId) status.textContent = error.message;
    } finally {
      if (id === requestId) buttons.forEach(button => { button.disabled = false; });
    }
  }

  start.addEventListener("click", () => advance());
  select.addEventListener("change", () => {
    requestId += 1;
    history = [];
    start.disabled = false;
    start.textContent = "Tomar las decisiones en este caso";
    status.textContent = "";
    content.replaceChildren();
  });
  box.append(start, status, content);
  return box;
}

export function renderInteractive(container, report, request) {
  const heading = element("h3", "Comparación en conversaciones de varios turnos");
  const scope = element("p", `${report.roots} historias reservadas · ${report.episodes} episodios por semilla · ${report.seeds.length} semillas. Se entrenó un selector compacto sobre evidencia pública, con el mismo entorno y memoria para los métodos.`, "study-meta");
  const summaryRows = Object.entries(report.means).map(([method, m]) => [method, names[method], percent.format(m.current_supported),
    percent.format(m.future_supported), decimal.format(m.questions)]);
  const columns = ["Método", "Actual", "Futuro", "Preguntas por episodio"];
  const scores = table("Políticas seleccionadas en validación", columns,
    summaryRows.filter(([method]) => !method.endsWith("-terminal")).map(row => row.slice(1)));
  const terminal = table("Diagnóstico de checkpoints terminales", columns,
    summaryRows.filter(([method]) => method.endsWith("-terminal")).map(row => row.slice(1)));
  const primary = report.contrasts["GRPO-longitudinal menos GRPO-inmediato"].future_supported;
  const estimate = element("p", `GRPO longitudinal frente a inmediato · ${points(primary.estimate)} de acierto futuro · IC 95 % de ${points(primary.interval[0])} a ${points(primary.interval[1])}.`, "study-meta");
  const criterion = element("p", report.primary_criteria.all_met
    ? "La mejora alcanzó el umbral y las condiciones de seguridad fijadas antes de evaluar."
    : "La mejora no alcanzó el criterio fijado antes de evaluar.", "study-meta");
  const end = report.terminal_contrasts.future_supported;
  const terminalEstimate = element("p", `Diferencia terminal · ${points(end.estimate)} de acierto futuro · IC 95 % de ${points(end.interval[0])} a ${points(end.interval[1])}.`, "study-meta");
  const episode = element("section", null, "study-summary interaction-panel");
  const label = element("label", "Historia reservada");
  const select = element("select");
  select.id = "interaction-case";
  label.htmlFor = select.id;
  const priority = item => item.id.startsWith("aclaracion-") ? 0 : item.family === "F4" ? 1 : 2;
  const order = [...report.cases].sort((a, b) => priority(a) - priority(b) || a.seed - b.seed);
  for (const item of order) select.add(new Option(`${item.title} · semilla ${item.seed}`, item.id));
  const status = element("p", null, "status");
  status.setAttribute("role", "status");
  const comparison = element("div", null, "comparison interaction-comparison");
  async function show() {
    const selected = select.value;
    status.textContent = "Abriendo la historia…";
    try {
      const data = await request(`/api/interactive/${encodeURIComponent(selected)}`);
      if (select.value !== selected) return;
      comparison.replaceChildren(...["GRPO-longitudinal-terminal", "GRPO-inmediato-terminal", "SFT-adicional"].map(
        method => traceCard(method, data.methods[method])));
      status.textContent = "";
    } catch (error) {
      if (select.value === selected) status.textContent = error.message;
    }
  }
  select.addEventListener("change", show);
  episode.append(label, select, simulator(select, request), status, comparison);
  if (report.branch_comparison) container.append(branchComparison(report.branch_comparison));
  container.append(heading, scope, scores, estimate, criterion, terminal, terminalEstimate);
  if (report.sensitivity) {
    const sensitivity = table("Banco independiente con una actualización SFT inicial",
      ["Método seleccionado", "Acierto futuro"], Object.entries(report.sensitivity).map(
        ([method, value]) => [names[method], percent.format(value)]));
    container.append(sensitivity);
  }
  if (report.guard) {
    const g = report.guard;
    const audit = table(`Verificador en ${g.visits} conversaciones reservadas`, ["Comprobación", "Operaciones"],
      [["Conservó escrituras respaldadas", `${g.confirmed_operations_retained} de ${g.confirmed_operations}`],
        ["Recuperó escrituras omitidas en la prueba", `${g.confirmed_operations_recovered_from_noop} de ${g.confirmed_operations}`],
        ["Bloqueó valores inventados en la prueba", `${g.fabricated_value_rejected} de ${g.fabricated_value}`],
        ["Bloqueó ampliaciones de vigencia en la prueba", `${g.expanded_scope_rejected} de ${g.expanded_scope}`]]);
    container.append(audit);
  }
  container.append(episode);
  show();
}
