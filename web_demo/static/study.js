const percent = new Intl.NumberFormat("es-EC", {style: "percent", minimumFractionDigits: 1, maximumFractionDigits: 1});

const labels = {SFT: "SFT congelado", "SFT-adicional-SELECT": "SFT adicional", "GRPO-SELECT": "GRPO"};

export function element(tag, content, className) {
  const node = document.createElement(tag);
  if (content) node.textContent = content;
  if (className) node.className = className;
  return node;
}

export function table(caption, headers, rows) {
  const wrapper = element("div", null, "study-table");
  const grid = element("table");
  grid.createCaption().textContent = caption;
  const header = grid.createTHead().insertRow();
  for (const name of headers) {
    const cell = element("th", name);
    cell.scope = "col";
    header.append(cell);
  }
  const body = grid.createTBody();
  for (const [name, ...values] of rows) {
    const row = body.insertRow();
    const label = element("th", name);
    label.scope = "row";
    row.append(label);
    for (const value of values) row.insertCell().textContent = value;
  }
  wrapper.append(grid);
  return wrapper;
}

export function renderChatMemoryStudy(container, report) {
  const labels = {"sft": "SFT", "grpo_sampled": "RL (GRPO)"};
  const rows = [];
  for (const verify of [false, true]) {
    for (const [method, label] of Object.entries(labels)) {
      const group = report.models.filter(row => row.verify === verify && row.model.includes(`-${method}-`));
      const count = key => group.reduce((sum, row) => sum + (row[key] || 0), 0);
      const fraction = (key, total) => `${count(key)}/${count(total)}`;
      rows.push([label, verify ? "Activado" : "Apagado", fraction("dialogue_passes", "dialogues"),
        fraction("recall_passed", "recall_total"), fraction("meal_passed", "meal_total")]);
    }
  }
  const panel = element("section", null, "study-summary");
  panel.append(element("h3", "Memoria entre sesiones"),
    element("p", "12 historias simuladas de regresión en las semillas 101, 202 y 303. Una historia cuenta como resuelta si acierta la conversación y conserva los recuerdos respaldados para después."),
    table("Evaluación del chat en CUDA", ["Modelo", "Verificador", "Historias completas", "Recuerdos acertados",
      "Pedidos de comida acertados"], rows),
    element("p", "Los perfiles comparten el controlador SFT para las operaciones de memoria. El adaptador SFT o GRPO elige el plato o la aclaración.", "help"));
  container.append(panel);
}

export function renderMeals(container, report) {
  const labels = {inicial: "SFT original", "sft-comun": "SFT inicial con exclusiones",
    sft: "SFT adicional", grpo: "SFT + GRPO", "sft-anterior": "SFT del banco anterior",
    "grpo-anterior": "GRPO del banco anterior"};
  const difference = report.difference;
  const signed = new Intl.NumberFormat("es-EC", {signDisplay: "exceptZero", maximumFractionDigits: 1});
  const points = value => signed.format(value * 100);
  const panel = element("section", null, "study-summary");
  if (report.exploration) {
    const {metrics, device, selection} = report.exploration;
    const names = {sft: "SFT", grpo: "RL (GRPO) · seis acciones", grpo_sampled: "GRPO · muestreo",
      sft_previo: "SFT anterior", grpo_previo: "GRPO anterior"};
    const cases = Object.values(metrics)[0].conversations;
    panel.append(element("h3", "Exploración de las seis acciones del menú"),
      element("p", `${cases} conversaciones reservadas · tres semillas · elecciones directas en ${device.toUpperCase()} con el verificador apagado.`),
      table("Aclaraciones y recomendaciones sin verificador",
        ["Modelo", "Semilla", "Aclaró sin plato", "Acertó con plato", "Platos incompatibles", "Época elegida"],
        Object.entries(metrics).map(([key, result]) => {
          const [seed, method] = key.split("-");
          const {exhausted, single, multiple} = result.states;
          return [names[method], seed, `${exhausted.correct}/${exhausted.n}`,
            `${single.correct + multiple.correct}/${single.n + multiple.n}`, `${result.conflicts}/${result.cases}`,
            selection[key] === 0 ? "Inicial" : selection[key] ?? "—"];
        })));
  }
  if (report.clarification) {
    const {metrics, selection, device} = report.clarification;
    const fraction = (n, total) => `${n}/${total}`;
    const names = {sft_previo: "SFT anterior", grpo_previo: "GRPO anterior",
      sft: "SFT", grpo: "SFT + GRPO", grpo_control: "GRPO · recompensa anterior"};
    panel.append(element("h3", "Aclarar cuando se agota el menú"),
      element("p", `Conversaciones reservadas · tres semillas · inferencia en ${device.toUpperCase()}. Se puntúa la elección directa del modelo.`),
      table("Aclaraciones y platos compatibles", ["Modelo", "Semilla", "Aclaró sin plato", "Acertó con plato",
        "Aclaró de más", "Conversaciones completas"], Object.entries(metrics).map(([key, stats]) => {
        const [seed, method] = key.includes("-") ? key.split("-") : ["—", key];
        const {exhausted, single, multiple} = stats.states;
        const viable = single.n + multiple.n;
        const label = selection[key] === 0 ? `${names[method]} · pesos iniciales` : names[method];
        return [label, seed, fraction(exhausted.correct, exhausted.n),
          fraction(single.correct + multiple.correct, viable), fraction(single.asked + multiple.asked, viable),
          fraction(stats.fully_correct, stats.conversations)];
      })));
  }
  if (report.qa) {
    const {panels, selection, device} = report.qa;
    const names = {baseline: "SFT previo", sft: "SFT adicional", grpo: "SFT + GRPO"};
    const {conversations, cases} = panels.baseline.reserved.live_order;
    const fraction = (n, total) => `${n}/${total}`;
    panel.append(element("h3", "Erratas y cambios de idea"),
      element("p", `${conversations} conversaciones simuladas reservadas · ${cases} turnos · una semilla · evaluación en ${device.toUpperCase()}. Se puntúa la elección directa del modelo con el verificador apagado.`),
      table("Resultados en el orden del menú del chat",
        ["Modelo", "Turnos correctos", "Platos incompatibles", "Conversaciones completas", "Época elegida"],
        Object.entries(panels).map(([key, value]) => {
          const stats = value.reserved.live_order;
          return [names[key], fraction(stats.correct, stats.cases), fraction(stats.conflicts, stats.cases),
            fraction(stats.fully_correct, stats.conversations), selection[key] ?? "—"];
        })),
      table("Acierto con seis órdenes del mismo menú", ["Modelo", "Historias nuevas", "Banco previo"],
        Object.entries(panels).map(([key, value]) => [names[key],
          percent.format(value.reserved.all_orders.correct / value.reserved.all_orders.cases),
          percent.format(value.archived.all_orders.correct / value.archived.all_orders.cases)])));
  }
  if (report.varied) {
    const varied = report.varied;
    panel.append(element("h3", "Qwen LoRA · generalización con el menú reordenado"),
      element("p", `${varied.dialogues} diálogos distintos · ${varied.cases} casos · seis órdenes por caso. Las elecciones se puntúan antes de la verificación.`),
      table("Elección directa en los seis órdenes",
        ["Modelo", "Acierto", "Conflicto", "Acierto en los seis", "Pares resueltos", "Época elegida"],
        Object.entries(varied.metrics).map(([key, value]) => [labels[key], percent.format(value.all.success),
          percent.format(value.all.conflict), percent.format(value.all_orders_success), percent.format(value.paired_flips),
          varied.selection[key] ?? "—"])),
      element("p", `Azar uniforme ${percent.format(varied.uniform)}. Diferencia GRPO frente a SFT adicional de ${points(varied.difference.estimate)} puntos, IC 95 % de ${points(varied.difference.interval[0])} a ${points(varied.difference.interval[1])}.`));
    if (varied.shuffle) {
      panel.append(table("Permutaciones nuevas y control sin diálogo", ["Modelo", "Con diálogo", "Sin diálogo"],
        Object.entries(varied.shuffle).map(([key, value]) => [labels[key], percent.format(value.success),
          percent.format(value.blinded_success)])));
    }
  }
  panel.append(element("h3", "Qwen LoRA · recomendaciones con exclusiones"),
    element("p", `${report.cases} casos reservados. El modelo elige entre seis opciones y la evaluación conserva esa elección.`),
    table("Selección del modelo antes de verificar", ["Modelo", "Acierto", "Conflicto", "Paso elegido"],
      Object.entries(report.metrics).map(([key, value]) => [labels[key], percent.format(value.success),
        percent.format(value.conflict), report.selection_steps[key] ?? "—"])),
    element("p", `GRPO frente a SFT adicional · ${points(difference.estimate)} puntos · IC 95 % de ${points(difference.interval[0])} a ${points(difference.interval[1])}.`));
  const detail = element("details");
  detail.append(element("summary", "Comprobación con todo el vocabulario"),
    table("Acierto de la primera salida sin limitar el vocabulario", ["Modelo", "Acierto"],
      Object.entries(report.metrics).map(([key, value]) => [labels[key], percent.format(value.raw_action_success)])));
  panel.append(detail);
  if (report.position) {
    panel.append(table("Acierto al cambiar el orden de las opciones", ["Modelo", "Acierto"],
      Object.entries(report.position).map(([key, value]) => [labels[key], percent.format(value.accuracy)])));
  }
  if (report.independent) {
    const extra = report.independent;
    panel.append(element("h4", "Puntuación de ingredientes por plato"),
      element("p", `${extra.cases} casos nuevos. GRPO conservó el ajuste SFT común en la selección por validación.`),
      table("Control sin posición del menú", ["Modelo", "Acierto", "Conflicto", "Paso elegido"],
        Object.entries(extra.metrics).map(([key, value]) => [labels[key], percent.format(value.success),
          percent.format(value.conflict), extra.selection_steps[key] ?? "—"])));
  }
  container.append(panel);
}

export function renderThesis(container, report, openCases) {
  const methods = Object.entries(report.methods);
  const compare = table("Comparación de selectores", ["Método", "Acierto actual", "Acierto futuro", "Éxito conjunto estricto"],
    methods.map(([name, metrics]) => [labels[name], ...[metrics.C2, metrics.FJS, metrics.strict_joint].map(percent.format)]));
  const guard = table("Acierto futuro con el verificador", ["Método", "Escritura original", "Con veto", "Con reparación"],
    methods.map(([name, metrics]) => [labels[name], ...["original", "veto", "repair"].map(arm => percent.format(metrics.guard_fjs[arm]))]));
  const scope = report.scope;
  const details = element("p", `${scope.independent_roots} historias reservadas, ${scope.training_seeds.length} semillas y ${scope.complete_paired_cells} ejecuciones por selector. Los porcentajes promedian las familias.`, "study-meta");
  const selection = Object.values(report.grpo_selected_steps);
  const analysis = element("section", null, "study-summary study-interpretation");
  analysis.append(element("h3", "Qué se observó"),
    element("p", `Validación conservó el checkpoint SFT inicial en ${selection.filter(step => step === 0).length} de ${selection.length} semillas de GRPO.`),
    element("p", "El verificador cambió únicamente la escritura. Cada contraste conserva la aclaración y la respuesta actual."));
  const explore = element("button", "Explorar historias reservadas", "secondary");
  explore.type = "button";
  explore.addEventListener("click", openCases);
  analysis.append(explore);
  container.replaceChildren(details, compare, guard, analysis);
}
