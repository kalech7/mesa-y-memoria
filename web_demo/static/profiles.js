import {createProfileView} from "./profiles-view.js";
import {renderProfileStudy} from "./profiles-study.js";

const storageKey = "tesis-profiles-v5";

function stored() {
  try {
    const parsed = JSON.parse(localStorage.getItem(storageKey));
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? {...parsed, chats: parsed.chats && typeof parsed.chats === "object" ? parsed.chats : {}}
      : {identity: "Mi prueba", chats: {}};
  }
  catch { return {identity: "Mi prueba", chats: {}}; }
}

export function connectProfiles(container, request) {
  const view = createProfileView(container);
  const saved = stored();
  view.identity.value = saved.identity || "Mi prueba";
  let catalog, chat, busy = false, opened = false, retry;
  const key = () => JSON.stringify([view.identity.value.trim() || "Mi prueba", view.model.value]);
  const selected = () => catalog?.models.find(model => model.id === view.model.value);
  const post = payload => request("/api/profiles/chat", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
  const token = () => saved.chats?.[key()];

  function listSavedChats() {
    const options = [new Option("Elegir una conversación…", "")];
    for (const [key, access] of Object.entries(saved.chats || {})) {
      if (!access?.id || !access?.owner_token) continue;
      let pair;
      try { pair = JSON.parse(key); } catch { continue; }
      if (!Array.isArray(pair) || pair.length !== 2 || typeof pair[0] !== "string") continue;
      const model = catalog?.models.find(item => item.id === pair[1]);
      if (!model) continue;
      const stage = model.panel === "last_epoch" ? " · última época" : "";
      options.push(new Option(`${pair[0]} · ${model.method} · semilla ${model.seed}${stage}`, key));
    }
    view.savedChat.replaceChildren(...options);
    view.savedChat.value = token() ? key() : "";
    view.savedField.hidden = options.length === 1;
  }

  function remember(state) {
    const previous = token();
    saved.chats ||= {};
    saved.chats[key()] = {id: state.id, owner_token: state.owner_token || previous?.owner_token};
    saved.identity = view.identity.value.trim() || "Mi prueba";
    saved.model = view.model.value;
    listSavedChats();
    try { localStorage.setItem(storageKey, JSON.stringify(saved)); }
    catch { view.status.textContent = "El navegador no pudo conservar el acceso. Mantén esta página abierta para seguir la prueba."; }
  }

  function controls() {
    const available = !!selected()?.available;
    for (const input of [view.identity, view.model, view.refresh, view.savedChat]) input.disabled = busy;
    for (const input of [view.input, view.send, view.example]) input.disabled = busy || !available;
    for (const input of [view.next, view.date, view.context]) input.disabled = busy || !chat || !available;
    view.recover.hidden = !token();
    view.recover.disabled = busy;
    container.setAttribute("aria-busy", String(busy));
    const model = selected();
    view.modelNote.textContent = model ? [model.method, model.seed && `Semilla ${model.seed}`,
      model.panel === "last_epoch" ? "Última época" : "Seleccionado en validación",
      model.epoch !== undefined && `Época ${model.epoch}`,
      model.eligible === false && "Checkpoint fuera del criterio de selección",
      model.epoch === 0 && "Se conserva el checkpoint inicial", model.reason].filter(Boolean).join(" · ")
      : catalog ? "Los checkpoints estarán disponibles al completar su verificación." : "Cargando modelos…";
  }

  async function run(message, action) {
    if (busy) return;
    busy = true;
    view.error.hidden = true;
    view.status.textContent = message;
    controls();
    try { await action(); }
    catch (problem) {
      view.errorText.textContent = problem.message;
      view.error.hidden = false;
      view.status.textContent = "";
    } finally { busy = false; controls(); }
  }

  async function restore() {
    chat = undefined;
    retry = undefined;
    view.render(chat, selected());
    listSavedChats();
    if (token()) {
      chat = await post({action: "load", ...token()});
      remember(chat);
    }
    view.render(chat, selected());
    view.status.textContent = chat?.pending ? "Hay una aclaración pendiente de tu respuesta."
      : !selected()?.available ? "Todavía no hay un modelo disponible para conversar." : "";
  }

  async function refresh() {
    const previous = view.model.value || saved.model;
    catalog = await request("/api/profiles/catalog");
    view.model.replaceChildren();
    for (const model of catalog.models) {
      const panel = model.panel === "last_epoch" ? " · última época" : "";
      const label = model.label.replace(/ terminal diagnóstico$/u, "");
      const identity = [label, `semilla ${model.seed}`, `época ${model.epoch}`].join(" · ");
      const option = new Option(`${identity}${panel}${model.available ? "" : " · no disponible"}`, model.id);
      option.disabled = !model.available;
      view.model.add(option);
    }
    const choice = catalog.models.find(model => model.id === previous && model.available)
      || catalog.models.find(model => model.id === catalog.default_model && model.available)
      || catalog.models.find(model => model.available);
    view.model.value = choice?.id || "";
    renderProfileStudy(view.study, catalog.results);
    await restore();
  }

  async function perform(action) {
    const text = view.input.value.trim();
    if (busy || !selected()?.available || action === "send" && !text) return;
    await run(action === "send" ? "El modelo está leyendo tu mensaje…" : "Abriendo la siguiente visita…", async () => {
      if (!chat) {
        chat = await post({action: "start", model: view.model.value});
        remember(chat);
      }
      const context = view.context.value || null;
      const signature = JSON.stringify([chat.id, action, text, view.date.value, context]);
      const payload = retry?.signature === signature ? retry.payload : {
        action, ...token(), revision: chat.revision, request_id: crypto.randomUUID(),
        ...(action === "send" ? {text} : {date: view.date.value, context}),
      };
      retry = {signature, payload};
      chat = await post(payload);
      retry = undefined;
      remember(chat);
      if (action === "send") view.input.value = "";
      view.render(chat, selected());
      view.status.textContent = chat.pending ? "El asistente espera tu aclaración." : action === "new_visit" ? "Visita abierta. Puedes consultar lo que recuerda." : "";
    });
    if (!container.hidden) view.input.focus({preventScroll: true});
  }

  view.form.addEventListener("submit", event => { event.preventDefault(); perform("send"); });
  view.next.addEventListener("click", () => perform("new_visit"));
  view.refresh.addEventListener("click", () => run("Comprobando modelos y resultados…", refresh));
  view.recover.addEventListener("click", () => run("Recuperando la conversación guardada…", restore));
  view.model.addEventListener("change", () => run("Abriendo la prueba de este modelo…", restore));
  view.savedChat.addEventListener("change", () => {
    if (!view.savedChat.value || !saved.chats[view.savedChat.value]) return;
    const [identity, model] = JSON.parse(view.savedChat.value);
    view.identity.value = identity;
    view.model.value = model;
    run("Recuperando recuerdos y conversación…", restore);
  });
  view.identity.addEventListener("change", () => run("Recuperando tu prueba…", restore));
  view.example.addEventListener("click", () => {
    view.input.value = "Suelo cenar locro de papa. Solo por hoy prefiero sopa de zapallo.";
    view.input.focus();
  });
  view.render();
  controls();
  return (showResults = false) => {
    const focusResults = () => {
      if (showResults) {
        view.study.focus({preventScroll: true});
        view.study.scrollIntoView({block: "start"});
      }
    };
    if (!opened) {
      opened = true;
      return run("Cargando el laboratorio de perfiles…", refresh).then(focusResults);
    }
    focusResults();
  };
}
