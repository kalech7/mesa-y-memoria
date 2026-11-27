import {element} from "./study.js";
import {memoryRecord, messageBubble, pastVisits} from "./profiles-memory.js";

function field(labelText, tag, id, type) {
  const wrapper = element("div", null, "profiles-field");
  const label = element("label", labelText);
  const input = element(tag);
  input.id = id;
  if (type) input.type = type;
  label.htmlFor = id;
  wrapper.append(label, input);
  return [wrapper, input];
}

function button(text, className = "secondary") {
  const node = element("button", text, className);
  node.type = "button";
  return node;
}

export function createProfileView(container) {
  const config = element("div", null, "profiles-config");
  const [identityField, identity] = field("Nombre de tu prueba", "input", "profiles-identity", "text");
  identity.maxLength = 48;
  identity.autocomplete = "off";
  identity.setAttribute("aria-describedby", "profiles-identity-help");
  const identityHelp = element("small", "Este navegador recuerda cada prueba y su modelo.", "help");
  identityHelp.id = "profiles-identity-help";
  identityField.append(identityHelp);
  const [modelField, model] = field("Quién decide cuándo preguntar", "select", "profiles-model");
  const [savedField, savedChat] = field("Conversaciones guardadas en este navegador", "select", "profiles-saved");
  savedField.classList.add("profiles-saved-field");
  savedField.hidden = true;
  const modelNote = element("p", null, "help");
  modelNote.id = "profiles-model-note";
  model.setAttribute("aria-describedby", modelNote.id);
  modelField.append(modelNote);
  const refresh = button("Actualizar modelos");
  config.append(identityField, modelField, refresh, savedField);

  const status = element("p", null, "status");
  status.setAttribute("role", "status");
  status.setAttribute("aria-live", "polite");
  const error = element("div", null, "error");
  error.setAttribute("role", "alert");
  error.hidden = true;
  const errorText = element("p");
  const recover = button("Recuperar conversación");
  error.append(errorText, recover);
  const shortcuts = element("nav", null, "profiles-shortcuts");
  shortcuts.setAttribute("aria-label", "Secciones de esta prueba");
  const resultsLink = element("a", "Ver resultados de esta corrida");
  resultsLink.href = "#perfiles-resultados";
  const memoryLink = element("a", "Recuerdos y evidencia");
  memoryLink.href = "#perfiles-recuerdos";
  shortcuts.append(resultsLink, memoryLink);
  const layout = element("div", null, "profiles-layout");
  const main = element("div", null, "profiles-conversation");
  const visit = element("div", null, "profiles-visit");
  const visitTitle = element("h3", "Visita 1");
  const budget = element("span", "2 preguntas disponibles", "chat-session");
  budget.setAttribute("aria-live", "polite");
  visit.append(visitTitle, budget);
  const horizonNote = element("p", "Esta visita amplía las tres visitas del experimento.", "help profiles-horizon");
  horizonNote.hidden = true;
  const empty = element("div", null, "profiles-empty");
  empty.append(element("p", "Cuéntale qué te gusta y hasta cuándo aplica. Cambia de visita para comprobar qué recuerda."));
  const example = button("Escribir un ejemplo");
  empty.append(example);
  const messages = element("ol", null, "chat-messages");
  messages.setAttribute("aria-label", "Mensajes de la visita actual");
  messages.setAttribute("role", "log");
  messages.setAttribute("aria-live", "polite");
  messages.setAttribute("aria-relevant", "additions");
  const form = element("form", null, "profiles-form");
  const [inputField, input] = field("Tu mensaje", "textarea", "profiles-message");
  input.rows = 3;
  input.maxLength = 800;
  input.required = true;
  input.placeholder = "Suelo cenar locro de papa, pero solo hoy prefiero sopa de zapallo.";
  const send = button("Enviar mensaje", "primary");
  send.type = "submit";
  form.append(inputField, send);
  const nextPanel = element("details", null, "profiles-next");
  nextPanel.append(element("summary", "Cambiar de visita o fecha"));
  const nextFields = element("div", null, "profiles-next-fields");
  const [dateField, date] = field("Fecha de la siguiente visita", "input", "profiles-date", "date");
  const [contextField, context] = field("Lugar de la siguiente visita", "select", "profiles-context");
  for (const [value, label] of [["", "Sin indicar"], ["casa", "Casa"], ["oficina", "Oficina"]]) context.add(new Option(label, value));
  const next = button("Saltar a la siguiente visita");
  nextFields.append(dateField, contextField, next);
  nextPanel.append(element("p", "La nueva visita conserva los recuerdos y el presupuesto de preguntas.", "help"), nextFields);
  const archive = element("div");
  main.append(visit, horizonNote, empty, messages, form, nextPanel, archive);

  const memory = element("aside", null, "profiles-memory");
  memory.id = "perfiles-recuerdos";
  memory.tabIndex = -1;
  memory.setAttribute("aria-labelledby", "profiles-memory-title");
  const memoryTitle = element("h3", "Recuerdos y evidencia");
  memoryTitle.id = "profiles-memory-title";
  const memoryDate = element("p", null, "help");
  const records = element("div");
  memory.append(memoryTitle, memoryDate, records);
  layout.append(main, memory);
  const study = element("section", null, "profiles-study");
  study.id = "perfiles-resultados";
  study.tabIndex = -1;
  study.setAttribute("aria-label", "Resultados verificados del experimento");
  for (const [link, target] of [[resultsLink, study], [memoryLink, memory]]) {
    link.addEventListener("click", event => {
      event.preventDefault();
      target.focus({preventScroll: true});
      target.scrollIntoView({block: "start"});
    });
  }
  container.append(shortcuts, config, status, error, layout, study);

  function render(state, chosen) {
    visitTitle.textContent = `Visita ${state?.visit || 1}${state?.date ? ` · ${state.date}` : ""}`;
    horizonNote.hidden = !state?.extrapolation;
    const remaining = state?.questions_remaining ?? 2;
    budget.textContent = `${remaining} ${remaining === 1 ? "pregunta disponible" : "preguntas disponibles"}`;
    empty.hidden = !!state?.messages?.length;
    messages.replaceChildren(...(state?.messages || []).map(message => messageBubble(message, chosen?.label)));
    messages.scrollTop = messages.scrollHeight;
    input.placeholder = state?.pending ? "Responde a la aclaración del asistente…" : "Cuéntale qué prefieres o consulta un recuerdo…";
    memoryDate.textContent = state ? `Vigencia en ${state.date}${state.context ? ` · ${state.context}` : ""}` : "Los recuerdos aparecerán al conversar.";
    records.replaceChildren(...(state?.memory?.length ? state.memory.map(record => memoryRecord(record, state)) : [element("p", "Aún no hay preferencias guardadas.", "help")]));
    archive.replaceChildren(...(state?.past_visits?.length ? [pastVisits(state.past_visits)] : []));
    if (state?.date) date.value = state.date;
    context.value = state?.context || "";
  }
  return {identity, model, savedField, savedChat, modelNote, refresh, status, error, errorText, recover, input, form, send, date, context, next, example, study, render};
}
