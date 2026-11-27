import {icon} from "./icons.js";

export const phases = {
  select: "Aclaración",
  respond: "Respuesta actual",
  write: "Escritura",
  reuse: "Sesión futura",
};

export const phaseIcons = {select: "help", respond: "chat", write: "database", reuse: "time"};

const decisions = {
  ASK: "ASK · Pidió aclaración",
  PROCEED: "PROCEED · Continuó",
  ANSWER: "ANSWER · Respondió",
  CONDITIONALIZE: "CONDITIONALIZE · Respuesta condicional",
  DEFER: "DEFER · Dejó la respuesta pendiente",
  NOOP: "NOOP · Sin cambios",
};

const slots = {
  TEMPORAL_SCOPE: "Alcance temporal",
  CURRENT_CONTEXT: "Contexto de la consulta",
  VALID_UNTIL: "Fecha de vencimiento",
  GENERIC: "Más detalles",
  CURRENT_PREFERENCE: "Preferencia actual",
  CONTEXT: "Contexto de la consulta",
  CONTEXT_SCOPE: "Alcance por contexto",
};

const operations = {NOOP: "Conservó la memoria", ADD: "Guardó una preferencia",
  SUPERSEDE: "Actualizó una preferencia", MARK_UNRESOLVED: "Marcó el alcance como pendiente"};
const guardStatuses = {repaired: "Recuperó una actualización respaldada", vetoed: "Bloqueó la escritura",
  partially_kept: "Conservó parte de la propuesta", kept: "Aceptó la propuesta", noop: "Conservó la memoria"};

const scopes = {
  stable: "Estable",
  permanent: "Estable",
  ongoing: "Estable",
  persistent: "Estable",
  contextual: "Contextual",
  temporary: "Temporal",
  interval: "Temporal",
  session: "Solo esta sesión",
  session_only: "Solo esta sesión",
};

const roles = {user: "Usuario", assistant: "Asistente", system: "Sistema", tool: "Simulador"};
const number = (value) => new Intl.NumberFormat("es-EC", {maximumFractionDigits: 3}).format(value);

function element(tag, value, className) {
  const node = document.createElement(tag);
  if (value != null) node.textContent = value;
  if (className) node.className = className;
  return node;
}

function pre(value) {
  return element("pre", typeof value === "string" ? value : JSON.stringify(value, null, 2));
}

function detail(label, content, expanded = false) {
  const block = element("details");
  block.open = expanded;
  const summary = element("summary");
  summary.append(icon("chevron"), label);
  block.append(summary, content);
  return block;
}

function dialogue(messages) {
  const list = element("ul", null, "dialogue");
  for (const message of messages) {
    const line = element("li");
    line.append(element("strong", `${roles[message.role] || message.role}. `), message.text);
    list.append(line);
  }
  return list;
}

const scopeName = (kind) => scopes[kind?.toLowerCase()] || kind;

function memoryList(records = [], query, usedIds = []) {
  if (!records.length) return element("p", "Memoria vacía.", "score-legend");
  const list = element("ul", null, "memory");
  for (const record of records) {
    const scope = record.scope || {};
    const used = usedIds.includes(record.memory_id);
    const expired = query?.date && scope.valid_until_exclusive && query.date >= scope.valid_until_exclusive;
    const flags = [
      scopeName(scope.kind),
      typeof scope.context === "object" && scope.context ? JSON.stringify(scope.context) : scope.context,
      scope.valid_from && `desde ${scope.valid_from}`,
      scope.valid_until_exclusive && `hasta ${scope.valid_until_exclusive} (excl.)`,
    ].filter(Boolean);
    const row = element("li");
    row.append(element("strong", record.value || record.memory_id));
    if (used) row.append(element("span", "Citado", "memory-tag"));
    if (expired) row.append(element("span", "Vencido", "memory-tag expired"));
    if (flags.length) row.append(element("span", flags.join(" · "), "record-scope"));
    list.append(row);
  }
  return list;
}

function memoryView(step, attribute) {
  const wrapper = element("div");
  const query = step.observation?.query;
  const usedIds = step.phase === "reuse" ? step.action?.evidence_ids : [];
  const records = step.memory || [];
  const relevant = records.filter((record) => record.attribute === attribute);
  const other = records.filter((record) => record.attribute !== attribute);
  wrapper.append(relevant.length ? memoryList(relevant, query, usedIds) :
    element("p", "Sin recuerdos de esta preferencia.", "score-legend"));
  if (other.length) wrapper.append(detail(`Otros recuerdos (${other.length})`, memoryList(other, query, usedIds)));
  return wrapper;
}

function actionLabel(step) {
  const action = step.action || {};
  if (step.error) return "Salida inválida";
  if (step.phase === "write") {
    const ops = (action.operations || []).map((op) => operations[op.op] || op.op).filter(Boolean);
    return ops.length ? ops.join(" · ") : decisions.NOOP;
  }
  return decisions[action.action || action.mode] || action.action || action.mode || "Sin decisión registrada";
}

function branchList(branches) {
  const list = element("ul", null, "memory");
  for (const branch of branches) {
    const item = element("li");
    const condition = typeof branch.condition === "object" && branch.condition
      ? Object.entries(branch.condition).map(([key, value]) =>
          key === "compatible_value" ? `si aplica la preferencia ${value}` : `${key.replaceAll("_", " ")} = ${String(value)}`
        ).join(" · ") : branch.condition;
    item.append(element("strong", branch.value), condition ? ` · ${condition}` : "");
    list.append(item);
  }
  return list;
}

function displayScore(value) {
  if (value === 1 || value === true) return "Sí";
  if (value === 0 || value === false) return "No";
  return value == null ? "—" : number(value);
}

function scoreList(metrics) {
  const scores = element("dl", null, "metrics");
  for (const [key, label] of Object.entries({
    C2: "Acierto actual",
    FJS: "Acierto futuro",
    J_primary: "Ambos aciertos",
    questions: "Preguntas",
    R_long: "Retorno del episodio",
  })) {
    const value = ["questions", "R_long"].includes(key)
      ? (metrics[key] == null ? "—" : number(metrics[key]))
      : displayScore(metrics[key]);
    const row = element("div");
    row.append(element("dt", label), element("dd", value));
    scores.append(row);
  }
  return scores;
}

export function renderContext(container, result, phase) {
  container.replaceChildren();
  const item = result.scenario;
  const selected = result.models[0]?.steps.find((step) => step.phase === phase);
  const initial = result.models[0]?.steps.find((step) => step.phase === "select");
  const future = phase === "reuse";
  const query = selected?.observation?.query;
  const queryText = typeof query === "string" ? query : query?.text || query?.question;
  const question = future
    ? queryText || item.future_question || selected?.messages?.find((message) => message.role === "user")?.text
    : item.question || queryText;
  const date = future ? item.future_date : item.current_date;
  const title = element("h3", null, "context-title");
  title.append(icon(future ? "time" : "chat"),
    [future ? "Consulta futura" : "Consulta actual", date].filter(Boolean).join(" · "));
  container.append(title);
  if (question) container.append(element("p", question, "query"));
  if (result.expected) {
    const expected = result.expected[future ? "future" : "current"];
    if (expected != null) {
      const value = typeof expected === "string" ? expected : JSON.stringify(expected);
      container.append(element("p", `Referencia del simulador · ${value}`, "score-legend"));
    }
  }
  const originalTurns = initial?.observation?.user_messages || initial?.messages;
  if (!future && originalTurns?.length) {
    container.append(detail("Ver conversación inicial", dialogue(
      originalTurns.map((turn) => ({role: "user", ...turn})),
    )));
  }
  if (initial?.memory?.length) {
    container.append(detail("Ver memoria inicial", memoryList(initial.memory)));
  }
}

function modelCard(model, phase, attribute) {
  const card = element("article", null, "model");
  card.dataset.arm = model.id;
  const heading = element("header", null, "model-head");
  const mark = element("span", null, "model-mark");
  mark.append(icon(model.id === "grpo" ? "bolt" : "layers"));
  const label = element("div");
  label.append(element("h3", model.label));
  if (model.checkpoint) label.append(element("p", model.checkpoint));
  heading.append(mark, label);
  card.append(heading);
  const body = element("div", null, "model-content");
  if (model.metrics) {
    const verdicts = element("div", null, "verdicts");
    for (const [key, title] of [["C2", "Actual"], ["FJS", "Futuro"]]) {
      const success = Boolean(model.metrics[key]);
      const chip = element("span", `${title} · ${success ? "Acierto" : "Sin acierto sustentado"}`, "verdict");
      chip.dataset.result = success ? "success" : "failure";
      verdicts.append(chip);
    }
    body.append(verdicts);
  }
  const step = model.steps.find((entry) => entry.phase === phase);
  if (!step) {
    body.append(element("p", "Sin datos para esta fase."));
    card.append(body);
    return card;
  }

  const action = element("p", null, "action");
  const actionIcon = step.error || step.action?.action === "ASK" ? "help" :
    phase === "select" ? "play" : phaseIcons[phase];
  action.append(icon(actionIcon), actionLabel(step));
  body.append(action);
  if (step.error) body.append(element("p", step.error, "invalid"));
  if (step.action?.slot) body.append(element("p", slots[step.action.slot] || step.action.slot));
  if (step.action?.value) body.append(element("p", step.action.value));
  const clarification = phase === "select" && step.action?.action === "ASK";
  if (clarification && step.messages?.length) {
    const questionIndex = step.messages.findLastIndex((message) => message.role === "assistant");
    if (questionIndex !== -1) body.append(dialogue(step.messages.slice(questionIndex)));
  }
  if (step.action?.branches?.length) body.append(branchList(step.action.branches));
  if (phase === "write") {
    const count = step.operations?.filter((operation) => operation.op !== "NOOP").length || 0;
    body.append(element("p", count ? `${count} ${count === 1 ? "operación aplicada" : "operaciones aplicadas"}` : "0 operaciones aplicadas"));
    if (model.verification) {
      const status = element("p", guardStatuses[model.verification.status] || model.verification.status, "guard-status");
      status.dataset.status = model.verification.status;
      body.append(status, detail("Evidencia de la verificación", pre(model.verification)));
    }
  }
  if (phase === "reuse" && model.metrics) body.append(scoreList(model.metrics));

  const memoryTitle = phase === "reuse" ? "Memoria persistida" : phase === "write" ? "Memoria después de escribir" : "Memoria disponible";
  body.append(detail(memoryTitle, memoryView(step, attribute), ["write", "reuse"].includes(phase)));
  if (step.operations?.length) body.append(detail("Operaciones aplicadas", pre(step.operations)));
  if (phase === "respond" && step.messages?.length) body.append(detail("Ver diálogo de este modelo", dialogue(step.messages)));
  if (step.raw) body.append(detail("Salida original", pre(step.raw)));
  if (step.observation) body.append(detail("Entrada de la fase", pre(step.observation)));
  if (phase === "reuse" && model.provenance) body.append(detail("Origen y huella del checkpoint", pre(model.provenance)));
  if (phase === "reuse" && model.artifact) body.append(detail("Origen de la trayectoria", pre(model.artifact)));
  card.append(body);
  return card;
}

export function renderComparison(container, result, phase) {
  container.replaceChildren(...result.models.map((model) => modelCard(model, phase, result.scenario.attribute)));
}
