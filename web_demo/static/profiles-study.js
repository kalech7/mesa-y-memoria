import {element, table} from "./study.js";

const percent = new Intl.NumberFormat("es-EC", {style: "percent", maximumFractionDigits: 1});
const number = new Intl.NumberFormat("es-EC", {maximumFractionDigits: 2});
const signed = new Intl.NumberFormat("es-EC", {maximumFractionDigits: 3, signDisplay: "exceptZero"});
const labels = {"Qwen-inicial": "Qwen inicial", "SFT-ramas": "SFT", "GRPO-ramas": "GRPO", "GRPO-temperatura": "GRPO con temperatura"};
const stages = {training: "Entrenamiento en curso", completed: "Evaluación completada", failed: "La ejecución requiere revisión",
  scoring_common_reference: "Calculando la referencia común", evaluating_common_parent: "Evaluando el modelo inicial",
  training_matched_qwen: "Entrenando las políticas", evaluating_checkpoint: "Evaluando un checkpoint",
  checkpoint_saved: "Checkpoint guardado", enumerating_training: "Calculando ramas de entrenamiento",
  enumerating_validation: "Calculando ramas de validación", selecting: "Seleccionando checkpoints en validación"};

function summaryRows(summary) {
  return Object.entries(summary).filter(([, row]) => row?.macro).map(([name, {macro}]) => [
    labels[name] || name, number.format(macro.longitudinal), percent.format(macro.future_supported),
    percent.format(macro.current_supported), number.format(macro.questions), percent.format(macro.future_unsupported_reuse),
  ]);
}

export function renderProfileStudy(container, results) {
  container.replaceChildren(element("h3", "Resultados del experimento"));
  if (!results || results.available === false) {
    container.append(element("p", results?.reason || "Los resultados aparecerán cuando termine la evaluación y se verifiquen sus archivos.", "help"));
    return;
  }
  const source = results.reserved || results;
  const headings = ["Modelo", "Utilidad", "Acierto futuro", "Acierto actual", "Preguntas", "Reutilización sin respaldo"];
  const verified = source.selection_verified !== false;
  if (results.reserved) container.append(element("p", verified ? "Evaluación reservada con checkpoints verificados."
    : "La evaluación reservada espera la verificación de sus checkpoints.", "help"));
  for (const [name, panel] of Object.entries(verified ? source.panels || {} : {})) {
    const rows = summaryRows(panel.qwen?.summary || panel.summary || {});
    if (rows.length) container.append(table(name === "validation_selected" ? "Reservado · checkpoints seleccionados en validación" : "Reservado · última época", headings, rows));
    if (name === "validation_selected") {
      const comparison = panel.comparisons?.["SFT-ramas"];
      const contrast = comparison?.contrasts?.longitudinal;
      if (contrast?.interval?.length === 2) container.append(element("p",
        `Contraste primario · GRPO T4 − SFT · Diferencia de utilidad ${signed.format(contrast.estimate)} · `
        + `IC 95 % [${signed.format(contrast.interval[0])}, ${signed.format(contrast.interval[1])}] · `
        + (comparison.passes ? "Cumple el criterio predefinido." : "No alcanza el criterio predefinido."), "profiles-primary-result"));
    }
  }
  if (verified && source.semantic?.parsers) {
    const rows = Object.entries(source.semantic.parsers).map(([name, parser]) => {
      const count = metric => `${parser.matches[metric]}/${parser.visits}`;
      return [name.toUpperCase(), count("response_exact"), count("new_write_exact"), count("retrieval_exact")];
    });
    container.append(table("Aciertos del analizador frente a anotaciones independientes", ["Analizador", "Respuesta actual", "Escritura", "Recuperación"], rows));
  }
  if (!source.panels) {
    const rows = summaryRows(results.validation?.summary || source.summary || {});
    if (rows.length) container.append(table("Resultados de validación", headings, rows));
    container.append(element("p", "La evaluación reservada aparecerá al finalizar y verificar sus archivos.", "help"));
  }
  const status = results.status;
  const stage = typeof status === "string" ? status : status?.stage || status?.status;
  if (stage) container.append(element("p", stages[stage] || "La corrida está en curso. Los resultados se actualizarán al verificar sus archivos.", "help"));
  const details = element("details", null, "profiles-provenance");
  details.append(element("summary", "Consultar resultados y verificación"), element("pre", JSON.stringify(results, null, 2)));
  container.append(details);
}
