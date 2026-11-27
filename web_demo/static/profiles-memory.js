import {element} from "./study.js";

const attributes = {dinner_preference: "Cena", breakfast_preference: "Desayuno", drink_preference: "Bebida"};
const scopes = {ongoing: "Habitual", session: "Solo esta visita", interval: "Temporal", contextual: "Por lugar", unknown: "Pendiente de aclarar"};
const actions = {PROCEED: "Respondió con la evidencia disponible", PROVISIONAL: "Respondió para esta visita sin guardar una preferencia", DEFER: "Dejó la respuesta pendiente",
  TEMPORAL_SCOPE: "Preguntó cuánto durará el cambio", VALID_UNTIL: "Preguntó la fecha de finalización", CURRENT_CONTEXT: "Preguntó dónde se aplica la preferencia", GENERIC: "Pidió una aclaración"};
const dateFormat = new Intl.DateTimeFormat("es-EC", {dateStyle: "medium", timeZone: "UTC"});
const formatDate = value => dateFormat.format(new Date(`${value}T00:00:00Z`));
const previousDate = value => new Date(Date.parse(`${value}T00:00:00Z`) - 86400000).toISOString().slice(0, 10);

export function memoryRecord(record, state) {
  const scope = record.scope;
  const end = scope.valid_until_exclusive && previousDate(scope.valid_until_exclusive);
  const expired = (end && end < state.date) || (scope.kind === "session" && scope.session_id !== `S${state.visit}`);
  const future = scope.valid_from && scope.valid_from > state.date;
  const elsewhere = scope.context && state.context && scope.context !== state.context;
  const unknownPlace = scope.context && !state.context;
  const superseded = state.memory?.some(other => other.applicable === true && other.status === "confirmed" && other.supersedes?.includes(record.memory_id));
  const retired = record.status && !["confirmed", "unresolved"].includes(record.status);
  const row = element("article", null, "profiles-record");
  row.append(element("h4", record.value));
  const meaning = [attributes[record.attribute] || record.attribute, scopes[scope.kind] || scope.kind,
    scope.context, scope.kind === "session" && scope.session_id,
    scope.valid_from && `Desde ${formatDate(scope.valid_from)}`, end && `Hasta ${formatDate(end)} inclusive`].filter(Boolean);
  row.append(element("p", meaning.join(" · "), "help"));
  const statusText = retired ? `Estado · ${record.status}` : superseded ? "Sustituido en este contexto" : expired ? "Vencido"
    : future ? "Empieza más adelante" : unknownPlace ? "Requiere indicar el lugar" : elsewhere ? "Aplica en otro lugar"
      : record.applicable === null || record.status === "unresolved" ? "Requiere aclarar su alcance" : "Vigente";
  const status = element("span", statusText, "memory-tag");
  if (expired) status.classList.add("expired");
  row.append(status);
  const quotes = record.evidence_quotes || [];
  if (quotes.length) {
    const details = element("details");
    details.append(element("summary", `Ver evidencia (${quotes.length})`));
    for (const quote of quotes) {
      const citation = element("blockquote", quote.text);
      citation.append(element("cite", quote.turn_id));
      details.append(citation);
    }
    row.append(details);
  }
  return row;
}

function provenance(message) {
  const panel = element("details", null, "profiles-provenance");
  panel.append(element("summary", "Ver decisión y evidencia"));
  const trace = message.trace;
  if (trace?.selected_action) panel.append(element("p", actions[trace.selected_action] || trace.selected_action));
  if (["TEMPORAL_SCOPE", "VALID_UNTIL", "CURRENT_CONTEXT", "GENERIC"].includes(trace?.selected_action)) {
    panel.append(element("p", "Esta aclaración consumió una pregunta del presupuesto compartido.", "help"));
  }
  const writes = (message.operations || []).filter(operation => operation.record);
  panel.append(element("p", writes.length
    ? `Guardó ${writes.map(operation => `${operation.record.value} · ${scopes[operation.record.scope.kind] || operation.record.scope.kind}`).join(", ")}.`
    : "La memoria de preferencias se conservó sin escrituras nuevas.", "help"));
  if (message.citations?.length) {
    const list = element("ul");
    for (const citation of message.citations) {
      const text = typeof citation === "string" ? citation : citation.text
        || citation.quotes?.map(quote => `«${quote.text}»`).join(" · ") || citation.value || citation.memory_id;
      list.append(element("li", text));
    }
    panel.append(list);
  }
  if (trace?.model?.fallback) panel.append(element("p", trace.model.fallback, "help"));
  const technical = element("details");
  technical.append(element("summary", "Detalles de ejecución"), element("pre", JSON.stringify({decision: trace, verification: message.verification, operations: message.operations}, null, 2)));
  panel.append(technical);
  return panel;
}

export function messageBubble(message, label) {
  const row = element("li", null, `chat-bubble chat-${message.role}`);
  row.append(element("strong", message.role === "user" ? "Tú" : label || "Asistente"), element("p", message.text));
  if (message.role === "assistant" && message.trace?.selected_action) {
    const text = actions[message.trace.selected_action] || message.trace.selected_action;
    row.append(element("small", text, "chat-write-status"));
  }
  if (message.trace || message.citations?.length || message.operations?.length || message.verification) row.append(provenance(message));
  return row;
}

export function pastVisits(visits) {
  const archive = element("details", null, "profiles-archive");
  archive.append(element("summary", `Visitas anteriores (${visits.length})`));
  for (const visit of [...visits].reverse()) {
    const section = element("section");
    section.append(element("h4", `Visita ${visit.visit ?? visit.session}${visit.date ? ` · ${formatDate(visit.date)}` : ""}`));
    for (const message of visit.messages || []) {
      const line = element("p");
      line.append(element("strong", message.role === "user" ? "Tú" : "Asistente"), ` · ${message.text}`);
      section.append(line);
    }
    archive.append(section);
  }
  return archive;
}
