"use strict";

const byId = id => document.getElementById(id);
const node = (tag, text, css) => {
  const item = document.createElement(tag);
  if (text != null) item.textContent = text;
  if (css) item.className = css;
  return item;
};
const model = byId("model"), picker = byId("chats"), input = byId("mensaje");
const examples = [
  ["Una cena sin pescado", "¿Qué puedo cenar hoy sin pescado?"],
  ["Una idea de desayuno", "¿Qué puedo desayunar sin huevo?"],
  ["Una bebida", "¿Qué bebida me recomiendas sin café?"],
  ["Plan de cenas", "Hoy quiero sopa para cenar. Estoy armando el menú de cenas de toda la semana."],
  ["Plan de desayunos", "Hoy quiero sopa para cenar. La cena de hoy ya está resuelta. Mañana elegimos los desayunos de toda la semana."],
  ["Guardar un hábito", "Mi cena habitual es locro de papa."],
  ["Consultar un recuerdo", "¿Qué tengo de cena habitual?"],
];
const mealNames = {dinner_preference: "Cena", breakfast_preference: "Desayuno", drink_preference: "Bebida"};
const scopeNames = {session: "Solo esta visita", interval: "Por fechas", contextual: "Por lugar", ongoing: "Habitual"};
const storagePrefix = "tesis-food-v6.session.";
const saved = new Map();
let state = null, active = "", busy = true, retry = null, durable = true;

function restoreKeys() {
  try {
    for (const key of Object.keys(localStorage)) {
      if (!key.startsWith(storagePrefix)) continue;
      let entry;
      try { entry = JSON.parse(localStorage.getItem(key)); } catch { continue; }
      if (entry?.id && entry?.owner_token && entry?.arm) saved.set(entry.id, entry);
    }
    return localStorage.getItem("tesis-food-v6.active");
  } catch { durable = false; return ""; }
}

function remember(entry) {
  saved.set(entry.id, entry);
  try { localStorage.setItem(storagePrefix + entry.id, JSON.stringify(entry)); }
  catch { durable = false; }
}

function markActive(id) {
  active = id;
  try { localStorage.setItem("tesis-food-v6.active", id || ""); }
  catch { durable = false; }
}

function choices() {
  const options = [...saved.values()].filter(entry => entry.arm === model.value)
    .sort((a, b) => b.touched - a.touched);
  picker.replaceChildren(new Option("Conversación nueva", ""),
    ...options.map(entry => new Option(entry.title || "Conversación sin mensajes", entry.id)));
  picker.value = active;
}

async function request(path, payload) {
  const response = await fetch(path, payload === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "El servidor no pudo completar el turno.");
  return data;
}

function showControls() {
  const ready = model.options.length > 0;
  for (const el of [model, picker, byId("new"), input, byId("send"), ...byId("examples").children]) {
    el.disabled = busy || !ready;
  }
  for (const el of [byId("next"), byId("date"), byId("place")]) el.disabled = busy || !state?.messages?.length;
}

async function run(work, notice) {
  if (busy) return;
  busy = true;
  showControls();
  byId("error").hidden = true;
  byId("status").textContent = notice;
  try {
    await work();
    byId("status").textContent = !durable ? "Este navegador solo podrá retomar el chat mientras esta página siga abierta."
      : state?.pending ? "Puedes responder la pregunta del asistente." : "";
  } catch (error) {
    byId("error").textContent = error.message;
    byId("error").hidden = false;
    byId("status").textContent = "";
  } finally {
    busy = false;
    showControls();
  }
}

function details(title) {
  const box = node("details");
  box.append(node("summary", title));
  return box;
}

function starterButton(title, text) {
  const button = node("button", title);
  button.type = "button";
  button.addEventListener("click", () => { input.value = text; input.focus(); });
  return button;
}

function recordCard(record, all) {
  const box = node("article", null, "record"), meta = node("div", null, "meta");
  const scope = record.scope;
  const superseded = all.some(other => other.applicable === true && other.supersedes?.includes(record.memory_id));
  const label = superseded ? "Sustituido" : record.applicable === true ? "Vigente"
    : record.applicable == null ? "Falta precisar el alcance" : "Fuera de vigencia en esta visita";
  meta.append(node("h3", record.value), node("span", label, label === "Vigente" ? "current" : "expired"));
  box.append(meta, node("p", [mealNames[record.attribute], scopeNames[scope.kind] || scope.kind, scope.context,
    scope.valid_from && "Desde " + scope.valid_from, scope.valid_until_exclusive && "Hasta antes del " + scope.valid_until_exclusive].filter(Boolean).join(" · "), "small"));
  if (record.evidence_quotes?.length) {
    const evidence = details("Ver frase de origen");
    for (const quote of record.evidence_quotes) evidence.append(node("blockquote", quote.text));
    box.append(evidence);
  }
  return box;
}

function bubble(message) {
  const li = node("li"), box = node("div", null, "food-bubble " + message.role);
  box.append(node("strong", message.role === "user" ? "Tú" : "Asistente"), node("p", message.text));
  if (message.trace) {
    const panel = details("Cómo respondió");
    const trace = message.trace;
    const action = trace.action || trace.selected_action;
    const source = trace.source === "qwen" ? (action === "ASK" || action === "TEMPORAL_SCOPE" ? "Qwen decidió preguntar ahora." : "Qwen decidió responder y reservar la pregunta.")
      : trace.source === "catalog" ? "Idea de comida del catálogo, filtrada por ingredientes."
      : trace.source === "recipe" ? "Preparación del recetario para el plato solicitado."
      : trace.reason === "outside_two_decision_visits" ? "Las decisiones de aclaración del experimento corresponden a las dos primeras visitas."
      : trace.source === "forced" ? "El presupuesto o las acciones disponibles determinaron esta respuesta."
      : trace.source === "conversation" ? "Orientación del asistente de alimentación."
      : "El asistente consultó la evidencia disponible sobre tus preferencias.";
    panel.append(node("p", source, "small"));
    if (trace.source === "qwen" && trace.model?.label && trace.device)
      panel.append(node("p", trace.model.label + " · " + trace.device, "small"));
    if (message.recommendation?.excluded?.length)
      panel.append(node("p", "Ingredientes excluidos del menú · " + message.recommendation.excluded.join(", "), "small"));
    for (const cite of message.citations || []) {
      for (const quote of cite.quotes || (cite.text ? [cite] : [])) panel.append(node("blockquote", quote.text));
    }
    box.append(panel);
  }
  li.append(box);
  return li;
}

function render() {
  const messages = state?.messages || [], memory = state?.memory || [];
  const last = messages.at(-1);
  let suggestions = examples;
  if (state?.pending) suggestions = [
    ["Solo esta visita", "Solo por hoy."], ["Es un hábito", "Es mi preferencia habitual."], ...examples,
  ];
  else if (last?.recommendation?.chosen || last?.trace?.source === "recipe") suggestions = [
    ["Ver cómo se prepara", "¿Cómo lo preparo?"], ["Pedir otra opción", "Dame otra opción."], ...examples,
  ];
  byId("examples").replaceChildren(...suggestions.map(([title, text]) => starterButton(title, text)));
  byId("visit").textContent = "Visita " + (state?.visit || 1) + (state?.date ? " · " + state.date : "");
  byId("budget").textContent = state?.visit > 2 ? "Visita de consulta"
    : (state?.questions_remaining ?? 1) === 1 ? "1 pregunta disponible" : "Pregunta de aclaración utilizada";
  byId("dialog").replaceChildren(...messages.map(bubble));
  byId("count").textContent = String(memory.length);
  byId("records").replaceChildren(...(memory.length ? memory.map(record => recordCard(record, memory))
    : [node("p", "Aún no hay preferencias guardadas en esta conversación.", "empty")]));
  byId("previous").hidden = !state?.past_visits?.length;
  byId("archive").replaceChildren(...(state?.past_visits || []).map(visit => {
    const row = node("article");
    row.append(node("h3", "Visita " + visit.visit + " · " + visit.date));
    for (const msg of visit.messages) row.append(node("p", (msg.role === "user" ? "Tú · " : "Asistente · ") + msg.text));
    return row;
  }));
  if (state) {
    const nextDate = new Date(state.date + "T12:00:00Z");
    nextDate.setUTCDate(nextDate.getUTCDate() + 1);
    byId("date").value = nextDate.toISOString().slice(0, 10);
  } else { byId("date").value = ""; }
  byId("place").value = state?.context || "";
  choices();
}

async function selectChat(id) {
  state = null;
  retry = null;
  markActive(id);
  input.value = "";
  byId("next-panel").open = false;
  render();
  if (!id) return;
  const entry = saved.get(id);
  state = await request("/api/food/chat", {action: "load", id, owner_token: entry.owner_token});
  render();
}

async function submit(action) {
  const text = input.value.trim();
  if (action === "send" && !text) return;
  await run(async () => {
    if (!state) {
      if (active) throw new Error("Vuelve a elegir la conversación guardada o abre una nueva.");
      state = await request("/api/food/chat", {action: "start", model: model.value});
      remember({id: state.id, arm: model.value, owner_token: state.owner_token, touched: Date.now()});
      markActive(state.id);
      choices();
    }
    const entry = saved.get(state.id);
    const fields = action === "send" ? {text} : {date: byId("date").value, context: byId("place").value || null};
    const signature = JSON.stringify([action, state.id, state.revision, fields]);
    if (retry?.signature !== signature) retry = {signature, payload: {
      id: state.id, owner_token: entry.owner_token, action, revision: state.revision,
      request_id: crypto.randomUUID(), ...fields,
    }};
    state = await request("/api/food/chat", retry.payload);
    retry = null;
    remember({...entry, title: entry.title || text.slice(0, 56), touched: Date.now()});
    if (action === "send") input.value = "";
    byId("next-panel").open = false;
    render();
  }, action === "send" ? "Procesando tu mensaje…" : "Abriendo visita…");
  input.focus({preventScroll: true});
}

byId("compose").addEventListener("submit", event => { event.preventDefault(); submit("send"); });
byId("next").addEventListener("click", () => submit("new_visit"));
byId("new").addEventListener("click", () => run(() => selectChat(""), ""));
picker.addEventListener("change", () => run(() => selectChat(picker.value), "Abriendo conversación…"));
model.addEventListener("change", () => run(() => {
  const entry = [...saved.values()].filter(item => item.arm === model.value).sort((a, b) => b.touched - a.touched)[0];
  return selectChat(entry?.id || "");
}, "Abriendo el modelo…"));

const initialId = restoreKeys();
busy = false;
run(async () => {
  const catalog = await request("/api/timing/catalog");
  model.replaceChildren(...catalog.arms.map(arm => new Option(arm.label, arm.id)));
  const last = saved.get(initialId);
  const available = last && catalog.arms.some(arm => arm.id === last.arm);
  model.value = available ? last.arm : catalog.defaults.arm;
  await selectChat(available ? last.id : "");
}, "Conectando con la USFQ…");
