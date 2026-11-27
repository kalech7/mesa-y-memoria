import {element} from "./study.js";

const scopeNames = {ongoing: "Habitual", session: "Solo una sesión", interval: "Temporal",
  contextual: "Por contexto", unknown: "Pendiente"};
const meals = {dinner_preference: "Cena", breakfast_preference: "Desayuno", drink_preference: "Bebida"};
const phases = {select: "Decisión de aclarar", respond: "Respuesta", write: "Escritura", recommendation: "Elección de una sugerencia"};
const verificationLabels = {kept: "Escritura conservada", repaired: "Escritura reparada", noop: "Sin cambios",
  vetoed: "Escritura bloqueada", partially_kept: "Escritura conservada en parte"};
const identity = "tesis-chat-local";

function detail(message) {
  const panel = element("details", null, "chat-trace");
  panel.append(element("summary", "Ver decisiones del modelo"));
  for (const step of message.trace) {
    panel.append(element("h4", phases[step.phase]), element("pre", step.raw || step.error || "Sin salida"));
    if (step.error && step.raw) panel.append(element("p", step.error, "help"));
  }
  if (message.write && !message.write.valid) panel.append(element("p", `SQLite rechazó la escritura · ${message.write.error}`, "help"));
  if (message.context_limited) panel.append(element("p", "El modelo recibió el intercambio reciente y los recuerdos que cabían en su contexto.", "help"));
  return panel;
}

const isoDate = new Intl.DateTimeFormat("es-EC", {dateStyle: "medium", timeZone: "UTC"});
const displayDate = value => isoDate.format(new Date(`${value}T00:00:00Z`));
const priorDay = value => new Date(Date.parse(`${value}T00:00:00Z`) - 86400000).toISOString().slice(0, 10);

function memoryRecord(record, {session, date}) {
  const row = element("article", null, "chat-record");
  const scope = record.scope;
  const last = scope.valid_until_exclusive && priorDay(scope.valid_until_exclusive);
  const dateLabel = last === scope.valid_from && last === date ? "Solo hoy" : scopeNames[scope.kind];
  const info = [meals[record.attribute] || record.attribute, dateLabel,
    typeof scope.context === "string" ? scope.context : null,
    scope.kind === "session" && scope.session_id !== `S${session}` ? "De una sesión anterior" : null,
    scope.valid_from > date ? `Desde el ${displayDate(scope.valid_from)}` : null,
    last && last !== date ? `${last < date ? "Venció el" : "Hasta el"} ${displayDate(last)}` : null,
    record.supersedes?.length ? `Actualiza ${record.supersedes.join(", ")}` : null].filter(Boolean);
  row.append(element("strong", record.value), element("p", info.join(" · "), "help"));
  if (record.evidence_quotes.length) {
    const source = element("details");
    source.append(element("summary", "Evidencia"));
    for (const quote of record.evidence_quotes) source.append(element("blockquote", quote.text));
    row.append(source);
  }
  return row;
}

function archivedSession(session) {
  const group = element("section", null, "chat-archive-session");
  const date = session.messages.find(message => message.date)?.date;
  group.append(element("h4", `Sesión ${session.session}${session.simulated ? " · Simulada" : ""}`));
  if (date) group.append(element("p", displayDate(date), "help"));
  for (const message of session.messages) {
    const line = element("p");
    line.append(element("strong", message.role === "user" ? "Tú" : "Asistente"), document.createTextNode(` · ${message.text}`));
    group.append(line);
  }
  return group;
}

export function connectChat(catalog, request) {
  const byId = id => document.getElementById(id);
  const model = byId("chat-model");
  const form = byId("chat-form");
  const input = byId("chat-message");
  const verify = byId("chat-verify");
  const send = byId("chat-send");
  const next = byId("chat-next");
  const sample = byId("chat-sample");
  const archive = byId("chat-archive");
  const example = byId("chat-example");
  const conversation = byId("chat-messages");
  const memory = byId("chat-records");
  const status = byId("chat-status");
  const error = byId("chat-error");
  let chat;
  let busy = false;
  let simulated = false;
  let retry = null;
  const byModel = new Map();
  const chatKey = (id, demo) => demo ? `${id}/simulado` : id;
  const activeKey = () => chatKey(model.value, simulated);
  const post = payload => request("/api/chat", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)});
  const choices = catalog.chat_choices ?? [{id: "sft", label: "SFT"}, {id: "grpo", label: "RL (GRPO)"}];
  const options = new Map(choices.map(item => [item.id, item.label]));
  const checkpoints = new Map(catalog.chat_models.map(item => [item.id, item.checkpoint]));
  for (const [id, label] of options) model.add(new Option(label, id));
  if (!catalog.live_available) status.textContent = catalog.live_reason;

  function controls() {
    const disabled = busy || !catalog.live_available;
    model.disabled = input.disabled = verify.disabled = send.disabled = example.disabled = sample.disabled = disabled;
    next.disabled = disabled || !(chat?.messages.length || chat?.memory.length);
    input.placeholder = chat?.pending ? "Responde a la aclaración del modelo…" : "¿Qué prefiero para cenar?";
    byId("chat-session").textContent = `Sesión ${chat?.session || 1}`;
    byId("chat-empty").hidden = !!chat?.messages.length;
    byId("chat-sample-label").textContent = simulated ? "Volver a mi chat" : "Abrir historial simulado";
    byId("chat-memory-note").textContent = simulated
      ? "Los recuerdos iniciales y las sesiones marcadas como simuladas son ficticios."
      : "Los recuerdos se conservan al abrir una nueva sesión.";
    byId("chat-checkpoint").textContent = checkpoints.get(model.value) || "";
    const hasMemory = !!chat?.memory.length;
    byId("chat-empty-title").textContent = hasMemory ? "Pon a prueba un recuerdo" : "Empieza con una preferencia";
    byId("chat-empty-note").textContent = hasMemory
      ? "Pregúntale qué prefieres cenar hoy o mañana y compara los recuerdos que recupera."
      : "Escribe como lo harías en un chat. Si el modelo pregunta, respóndele con tus palabras.";
  }

  function show() {
    conversation.replaceChildren();
    for (const message of chat?.messages || []) {
      const bubble = element("li", null, `chat-bubble chat-${message.role}`);
      bubble.append(element("strong", message.role === "user" ? "Tú" : options.get(chat.model.id)), element("p", message.text));
      if (message.fallback) {
        bubble.append(element("small", "Respuesta de respaldo basada en la evidencia. La salida del modelo fue inválida.", "help"));
      }
      if (message.recommendation?.source === "catalog") {
        const reason = message.recommendation.reason === "remembered_preference"
          ? "El verificador priorizó un recuerdo vigente compatible."
          : message.recommendation.reason === "unnecessary_clarification"
          ? "El modelo pidió aclarar y el verificador encontró un plato compatible."
          : message.recommendation.model_choice && message.text.startsWith("¿")
          ? "El verificador bloqueó la sugerencia y pidió más información."
          : "La sugerencia del modelo fue sustituida por una opción comprobada.";
        bubble.append(element("small", message.trace?.length ? reason
          : "Comprobación de restricciones de la comida.", "help"));
      }
      if (message.recommendation?.raw) {
        bubble.append(element("small", "Elección del modelo sin corrección.", "help"));
      } else if (message.recommendation?.model_choice && message.recommendation.source === "model") {
        bubble.append(element("small", "Elección del modelo conservada.", "help"));
      }
      if (message.verification && message.verification.status !== "noop") {
        bubble.append(element("small", verificationLabels[message.verification.status], "chat-write-status"));
      }
      if (message.selection_guard) {
        const note = message.selection_guard === "explicit_scope" ? "La vigencia ya estaba indicada en tu mensaje."
          : "Falta confirmar la vigencia del cambio.";
        bubble.append(element("small", note, "help"));
      }
      if (message.memory_read?.source === "sqlite" && message.memory_read.evidence_ids?.length) {
        bubble.append(element("small", "Consulta de recuerdos en SQLite.", "help"));
      }
      const usedIds = message.recommendation?.memory_used?.map(item => item.memory_id) || message.memory_read?.evidence_ids;
      if (usedIds?.length) {
        const values = (chat.memory || []).filter(item => usedIds.includes(item.memory_id)).map(item => item.value);
        if (values.length) bubble.append(element("p", `Recuerdo consultado · ${[...new Set(values)].join(", ")}`, "chat-memory-used"));
      }
      if (message.trace?.length) bubble.append(detail(message));
      conversation.append(bubble);
    }
    conversation.scrollTop = conversation.scrollHeight;
    const records = (chat?.memory || []).map(record => memoryRecord(record, chat));
    memory.replaceChildren(...(records.length ? records : [element("p", "Aún no hay recuerdos guardados.", "help")]));
    const sessions = chat?.past_sessions || [];
    archive.hidden = !sessions.length;
    byId("chat-archive-sessions").replaceChildren(...[...sessions].reverse().map(archivedSession));
    controls();
  }

  function remember() {
    if (chat) byModel.set(chatKey(chat.model.id, chat.simulated), chat);
    try {
      const stored = Object.fromEntries([...byModel].map(([key, value]) => [key, value.id]));
      sessionStorage.setItem(identity, JSON.stringify({model: model.value, simulated, ids: stored}));
    } catch {}
  }

  async function perform(action) {
    if (busy || !catalog.live_available) return;
    const text = input.value.trim();
    if (action === "send" && !text) return;
    busy = true;
    controls();
    error.hidden = true;
    status.textContent = action === "send" ? "Ejecutando el modelo local…"
      : action === "open_sample" ? "Cargando el historial simulado…" : "Abriendo otra sesión…";
    try {
      if (!chat) {
        chat = await post({action: "start", model: model.value, simulated});
        remember();
      }
      if (action === "open_sample") {
        status.textContent = "";
        show();
        return;
      }
      const signature = JSON.stringify([chat.id, action, text, verify.checked]);
      const payload = retry?.signature === signature ? retry.payload :
        {action, id: chat.id, revision: chat.revision, text, verify: verify.checked,
          request_id: crypto.randomUUID()};
      retry = {signature, payload};
      chat = await post(payload);
      retry = null;
      remember();
      if (action === "send") input.value = "";
      status.textContent = chat.pending ? "El modelo espera tu aclaración." : "";
      show();
    } catch (problem) {
      if (action === "open_sample") {
        simulated = false;
        chat = byModel.get(activeKey());
        remember();
        show();
      }
      error.textContent = problem.message;
      error.hidden = false;
      status.textContent = "";
    } finally {
      busy = false;
      controls();
      input.focus({preventScroll: true});
    }
  }

  form.addEventListener("submit", event => { event.preventDefault(); perform("send"); });
  input.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      form.requestSubmit();
    }
  });
  next.addEventListener("click", () => perform("new_session"));
  function switchChat() {
    chat = byModel.get(activeKey());
    retry = null;
    error.hidden = true;
    status.textContent = chat?.pending ? "El modelo espera tu aclaración." : "";
    show();
    remember();
    if (simulated && !chat) perform("open_sample");
  }
  model.addEventListener("change", switchChat);
  sample.addEventListener("click", () => {
    simulated = !simulated;
    switchChat();
  });
  example.addEventListener("click", () => {
    const recall = {dinner_preference: "¿Qué prefiero para cenar?", breakfast_preference: "¿Qué prefiero desayunar?",
      drink_preference: "¿Qué bebida prefiero?"};
    input.value = chat?.memory.length ? recall[chat.attribute] : model.value.startsWith("comidas-")
      ? "¿Qué puedo cenar? Hoy no quiero comer huevo ni arroz." : "Para la cena, prefiero sopa de verduras de forma habitual.";
    input.focus();
  });

  show();
  try {
    const saved = JSON.parse(sessionStorage.getItem(identity));
    if (saved && catalog.live_available) {
      if (options.has(saved.model)) model.value = saved.model;
      simulated = saved.simulated === true;
      const ids = Object.entries(saved.ids || {}).filter(([key]) => options.has(key.split("/")[0]));
      busy = true;
      controls();
      Promise.allSettled(ids.map(([, id]) => post({action: "resume", id}))).then(results => {
        for (const result of results) if (result.status === "fulfilled") {
          byModel.set(chatKey(result.value.model.id, result.value.simulated), result.value);
        }
        chat = byModel.get(activeKey());
        busy = false;
        if (!chat) simulated = false;
        if (!chat && ids.length) status.textContent = "El servidor se reinició. Puedes empezar otra conversación.";
        else if (chat?.pending) status.textContent = "El modelo espera tu aclaración.";
        show();
      });
    }
  } catch {}
}
