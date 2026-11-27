import {element, table} from "./study.js";

const labels = {
  PROCEED: "Responder y continuar", PROVISIONAL: "Responder sin guardar", DEFER: "Aplazar la respuesta",
  TEMPORAL_SCOPE: "Preguntar si el cambio durará", CURRENT_CONTEXT: "Preguntar por el lugar",
  VALID_UNTIL: "Preguntar hasta cuándo", GENERIC: "Pedir más detalles",
};
const probes = {
  first_preference_reuse: "Preferencia de la primera visita",
  second_preference_reuse: "Preferencia de la segunda visita",
  unrelated_preference: "Otra preferencia",
};
const percent = new Intl.NumberFormat("es-EC", {style: "percent", maximumFractionDigits: 1});

function button(text, handler, className = "secondary") {
  const node = element("button", text, className);
  node.type = "button";
  node.addEventListener("click", handler);
  return node;
}

function selectField(title, id) {
  const field = element("div", null, "budget-field");
  const label = element("label", title);
  const select = element("select");
  select.id = id;
  label.htmlFor = id;
  field.append(label, select);
  return [field, select];
}

function dialogue(messages) {
  const list = element("ul", null, "dialogue");
  for (const message of messages) {
    list.append(element("li", `${message.role === "assistant" ? "Asistente" : "Usuario"}. ${message.text}`));
  }
  return list;
}

function memories(records, title) {
  const details = element("details", null, "budget-memory");
  const list = element("ul", null, "memory");
  for (const record of records) {
    const row = element("li");
    const scope = record.scope;
    const kind = {ongoing: "Habitual", interval: "Temporal", contextual: "Por contexto", session: "De esta sesión"}[scope.kind];
    row.append(element("strong", record.value),
      element("span", [kind, scope.valid_from && `desde ${scope.valid_from}`,
        scope.valid_until_exclusive && `vence el ${scope.valid_until_exclusive}`].filter(Boolean).join(" · "), "record-scope"));
    list.append(row);
  }
  details.append(element("summary", `${title} (${records.length})`), list);
  return details;
}

function closedVisit(visit, index) {
  const details = element("details", null, "budget-closed");
  const result = visit.outcome.value || (visit.outcome.mode === "DEFER" ? "Respuesta aplazada" : "Respuesta condicional");
  const questions = visit.events.flatMap(event => [event.question, event.reply]).filter(Boolean);
  details.append(
    element("summary", `Visita ${index + 1} · ${result} · ${visit.questions} ${visit.questions === 1 ? "pregunta" : "preguntas"}`),
    dialogue([...visit.opening_messages, ...questions]),
    element("p", labels[visit.closing_action], "help"),
    memories(visit.memory_after, "Memoria al cerrar la visita"),
  );
  return details;
}

function future(result) {
  return table("Recuperación posterior", ["Consulta", "Valor recuperado", "Evidencia suficiente"],
    result.future.map(item => [probes[item.kind] || item.kind, item.response.value || "Sin respuesta única",
      item.supported ? "Sí" : "No"]));
}

function comparisonCard(result) {
  const card = element("article", null, "model");
  const heading = element("header", null, "model-head");
  const content = element("div", null, "model-content");
  heading.append(element("h4", result.label));
  if (result.epoch !== null && result.epoch !== undefined) {
    heading.append(element("p", result.epoch === 0 ? "Checkpoint inicial seleccionado" : `Época ${result.epoch} seleccionada`));
  }
  const ask = result.visits.findIndex(visit => visit.questions);
  const metric = result.metrics;
  content.append(element("p", ask < 0 ? "Continuó sin preguntar" : `Usó la pregunta en la visita ${ask + 1}`),
    element("p", `Acierto futuro sustentado · ${percent.format(metric.future_supported)}`, "help"),
    ...result.visits.map(closedVisit), future(result));
  card.append(heading, content);
  return card;
}

export function connectSharedBudget(container, initial, request) {
  const [caseField, select] = selectField("Historia", "budget-case");
  const [policyField, model] = selectField("Quién decide el siguiente paso", "budget-policy");
  const controls = element("div", null, "budget-toolbar");
  const budget = element("div", null, "budget-remaining");
  const availability = element("p", null, "help");
  const status = element("p", null, "status");
  const error = element("p", null, "error");
  const current = element("div", null, "budget-current");
  const comparisons = element("section", null, "budget-comparisons");
  status.setAttribute("role", "status");
  budget.setAttribute("aria-live", "polite");
  error.setAttribute("role", "alert");
  model.setAttribute("aria-describedby", "budget-availability");
  availability.id = "budget-availability";
  comparisons.setAttribute("aria-label", "Comparación de políticas");
  error.hidden = true;
  let catalog = initial, history = [], busy = false, state, opened = false;

  for (const item of catalog.cases) select.add(new Option(item.title, item.id));
  const apply = button("Ejecutar un paso", () => step({step_with: model.value}), "primary");
  const restart = button("Reiniciar", () => step({actions: []}));
  const refresh = button("Actualizar modelos", () => run("Comprobando los adaptadores…", async () => {
    catalog = await request("/api/shared-budget");
    showPolicies();
  }));
  const compare = button("Comparar desde el inicio", () => run("Ejecutando las políticas en la misma historia…", async () => {
    const compare = catalog.policies.filter(item => item.available).map(item => item.id);
    const result = await post({compare});
    const grid = element("div", null, "comparison budget-grid");
    const title = element("h3", "Recorridos comparados");
    title.tabIndex = -1;
    grid.append(...result.comparisons.map(comparisonCard));
    comparisons.replaceChildren(title, grid);
    if (!container.hidden) title.focus();
  }));
  const buttons = element("div", null, "action-options budget-toolbar-actions");
  buttons.append(apply, restart, compare, refresh);
  controls.append(caseField, policyField, buttons, availability);
  container.append(controls, budget, status, error, current, comparisons);

  function showPolicies() {
    const previous = model.value;
    model.replaceChildren();
    for (const item of catalog.policies) {
      const option = new Option(item.epoch === 0 ? `${item.label} · checkpoint inicial` : item.label, item.id);
      option.disabled = !item.available;
      model.add(option);
    }
    const selection = catalog.policies.find(item => item.id === previous && item.available)
      || catalog.policies.find(item => item.id === "sft" && item.available) || catalog.policies.find(item => item.available);
    model.value = selection.id;
    const pending = catalog.policies.filter(item => ["sft", "grpo"].includes(item.id) && !item.available);
    const device = catalog.base_model && `${catalog.base_model} · ${catalog.device.toUpperCase()}`;
    availability.textContent = [device, ...new Set(pending.map(item => item.reason))].filter(Boolean).join(" · ");
  }

  function sync() {
    select.disabled = model.disabled = restart.disabled = refresh.disabled = busy;
    apply.disabled = busy || !state || state.done;
    compare.disabled = busy || catalog.policies.filter(item => item.available).length < 2;
    for (const action of current.querySelectorAll("button")) action.disabled = busy;
    container.setAttribute("aria-busy", String(busy));
  }

  function post(fields) {
    return request("/api/shared-budget", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({case_id: select.value, ...fields})});
  }

  async function run(message, action) {
    if (busy) return;
    busy = true;
    error.hidden = true;
    status.textContent = message;
    sync();
    try {
      await action();
      status.textContent = "";
    } catch (problem) {
      status.textContent = "";
      error.textContent = problem.message;
      error.hidden = false;
    } finally {
      busy = false;
      sync();
    }
  }

  function step(fields, focus = true) {
    return run("Avanzando la historia…", async () => {
      const result = await post({actions: history, ...fields});
      state = result;
      history = result.actions;
      const heading = element("h3", result.done ? "Consultas futuras" : `Visita ${result.observation.visit} de 2`);
      heading.tabIndex = -1;
      current.replaceChildren(...result.visits.map(closedVisit), heading);
      budget.replaceChildren(element("strong", String(result.questions_remaining)),
        element("span", result.questions_remaining === 1 ? "pregunta disponible" : "preguntas disponibles"));
      if (result.done) {
        current.append(future(result));
      } else {
        const actions = element("div", null, "action-options");
        for (const action of result.legal_actions) {
          actions.append(button(labels[action], () => step({actions: [...history, action]})));
        }
        current.append(dialogue(result.observation.user_messages), memories(result.observation.memory, "Memoria disponible"),
          element("p", "Elige una acción o delega el paso en el modelo.", "help"), actions);
      }
      if (focus && !container.hidden) heading.focus();
    });
  }

  select.addEventListener("change", () => {
    history = [];
    state = undefined;
    budget.replaceChildren();
    current.replaceChildren();
    comparisons.replaceChildren();
    step({actions: []}, false);
  });
  showPolicies();
  sync();
  return () => {
    if (!opened) {
      opened = true;
      step({actions: []}, false);
    }
  };
}
