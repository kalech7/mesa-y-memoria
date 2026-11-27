"""Reconstruct exploratory reports from real JSONL traces, including absences.

No outcome is inferred from checkpoint existence. Human authorship and review
remain not_executed when only AI proxy and automatic audits were performed.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

from .paired_statistics import aggregate_rows, binomial_interval, paired_bootstrap


def _flatten(row):
    flat = {**row, **row.get("metrics", {}), **row.get("rewards_by_component", {})}
    flat["experiment"] = row.get("experiment", "E3" if row.get("intervention") else "E1" if "FJS" not in flat and "C2" in flat else "E2")
    flat.setdefault("observation_condition", "O2")
    flat.setdefault("extraction_track", "REF")
    flat.setdefault("reader_id", "fixed_symbolic")
    flat.setdefault("technical_status", "executed")
    if flat["technical_status"] == "failed":
        flat.setdefault("C2", 0)
        flat.setdefault("FJS", 0)
    if flat.get("C2") is not None:
        flat["current_error"] = 1. - flat["C2"]
    opportunity = row.get("trace", {}).get("episode", {}).get("opportunity", {})
    query = row.get("trace", {}).get("episode", {}).get("current_query", {})
    flat["control_group"] = ("primary_F4" if opportunity.get("eligible") else
                             "unaffected_attribute" if query.get("attribute") == "drink_preference" else
                             "clear_case" if opportunity.get("negative_control") else "other") if flat["experiment"] == "E3" else None
    # Analysis keeps scalar outcomes and the preregistered opportunity flag.
    # Full prompts/generations/SQLite traces remain in the streamed artifact.
    compact = {key: value for key, value in flat.items() if value is None or isinstance(value, (str, int, float, bool))}
    compact["trace"] = {"episode": {"opportunity": row.get("trace", {}).get("episode", {}).get("opportunity", {})}}
    compact["attempts"] = [{"status": a.get("status")} for a in row.get("attempts", [{}])]
    compact["resource_counters"] = row.get("resource_counters", {})
    return compact


def _signature(row):
    return tuple(row.get(k) for k in ("experiment", "split", "method", "observation_condition", "intervention",
                                     "reader_id", "extraction_track", "nonresponse_rate", "control_group"))


def _descriptive_interval(root_rows, metric, resamples):
    """95% root uncertainty, with seeds averaged within each root."""
    values, families = defaultdict(list), {}
    for row in root_rows:
        values[row["root_id"]].append(row[metric])
        families[row["root_id"]] = row["scenario_family"]
    strata = defaultdict(list)
    for rid in sorted(values):
        strata[families[rid]].append(mean(values[rid]))
    combined = [value for entries in strata.values() for value in entries]
    if all(v == 0 for v in combined) or all(v == 1 for v in combined):
        result = binomial_interval(int(sum(combined)), len(combined))
        result["interpretation"] = "Conservative root-level boundary bound; avoids degenerate bootstrap"
        return result
    try:
        import numpy as np
        rng = np.random.default_rng(7331)
        boot = []
        for offset in range(0, resamples, 256):
            count = min(256, resamples - offset)
            samples = np.zeros(count)
            for entries in strata.values():
                array = np.asarray(entries)
                indices = rng.integers(0, len(array), size=(count, len(array)))
                samples += array[indices].mean(axis=1) / len(strata)
            boot.extend(samples.tolist())
        interval = np.quantile(boot, [.025, .975]).tolist()
    except ImportError:
        import random
        rng = random.Random(7331)
        boot = sorted(mean(mean(rng.choice(entries) for _ in entries) for entries in strata.values()) for _ in range(resamples))
        interval = [boot[int((resamples - 1) * .025)], boot[int((resamples - 1) * .975)]]
    return {"interval": interval, "confidence": .95, "independent_roots": len(values),
            "resamples": resamples, "method": "family_stratified_root_bootstrap_seeds_averaged"}


def summarize_rows(rows, *, resamples=20000):
    groups = defaultdict(list)
    for row in rows:
        groups[_signature(row)].append(row)
    summaries = []
    for signature, entries in sorted(groups.items(), key=lambda pair: repr(pair[0])):
        summary = dict(zip(("experiment", "split", "method", "observation_condition", "intervention", "reader_id",
                            "extraction_track", "nonresponse_rate", "control_group"), signature))
        summary.update(scheduled_episodes=len(entries), independent_roots=len({r["root_id"] for r in entries}),
                       attempts=sum(len(r.get("attempts", [{}])) for r in entries),
                       technical_failures=sum(r["technical_status"] == "failed" for r in entries),
                       training_seeds=sorted({r["seed"] for r in entries}))
        for metric in ("C2", "FJS", "current_error", "questions", "unsupported_write_episode", "unsupported_current", "invalid_outputs", "R_long", "R_short"):
            valid = [r for r in entries if r.get(metric) is not None]
            if not valid:
                continue
            root_rows = aggregate_rows(valid, metric)
            by_seed_family = defaultdict(list)
            for row in root_rows:
                by_seed_family[(row["seed"], row["scenario_family"])].append(row[metric])
            seeds = sorted({key[0] for key in by_seed_family})
            per_seed = {str(seed): mean(mean(values) for (s, _), values in by_seed_family.items() if s == seed) for seed in seeds}
            summary[metric + "_macro"] = mean(per_seed.values())
            summary[metric + "_micro"] = mean(r[metric] for r in valid)
            summary[metric + "_by_seed"] = per_seed
            families = sorted({key[1] for key in by_seed_family})
            summary[metric + "_by_family"] = {f: mean(mean(values) for (s, family), values in by_seed_family.items() if family == f)
                                               for f in families}
            # At zero/all events, supply conservative root bounds rather than
            # reporting a degenerate bootstrap interval for a rare event rate.
            root_values = defaultdict(list)
            for row in root_rows:
                root_values[row["root_id"]].append(row[metric])
            combined = [mean(v) for v in root_values.values()]
            if metric in {"C2", "FJS", "unsupported_write_episode", "unsupported_current"} and combined and (
                    all(v == 0 for v in combined) or all(v == 1 for v in combined)):
                summary[metric + "_boundary_root_interval"] = binomial_interval(int(sum(combined)), len(combined))
            if metric in {"C2", "FJS"}:
                summary[metric + "_descriptive_95_interval"] = _descriptive_interval(root_rows, metric, resamples)
        summaries.append(summary)
    return summaries


def _constant_difference_bound(result, root_values):
    """Do not claim zero uncertainty because no discordant roots were observed."""
    if root_values and max(root_values) - min(root_values) < 1e-12 and result["interval"][1] - result["interval"][0] < 1e-12:
        delta = binomial_interval(0, len(root_values), confidence=.99)["interval"][1]
        center = result["estimate"]
        result["raw_bootstrap_interval"] = result["interval"]
        result["interval"] = [max(-1., center - (center + 1.) * delta), min(1., center + (1. - center) * delta)]
        result["boundary_correction"] = {"method": "conservative_root_bound_for_unobserved_discordance",
                                         "roots": len(root_values), "discordant_roots_observed": 0,
                                         "unobserved_discordance_upper": delta}
    return result


def _fast_bootstrap(rows, left, right, metric, resamples, *, sensitivity=False):
    """Same paired family-stratified estimator, vectorized when NumPy exists."""
    try:
        import numpy as np
    except ImportError:
        result = paired_bootstrap(rows, left, right, metric, resamples=resamples, confidence=.99, resample_seeds=sensitivity)
        aggregated = aggregate_rows(rows, metric)
        maps = [{(r["seed"], r["root_id"]): r[metric] for r in aggregated
                 if (r["method"], r["condition"], r["intervention"]) == tuple(selector)} for selector in (left, right)]
        root_values = defaultdict(list)
        for seed, rid in maps[0]:
            root_values[rid].append(maps[0][(seed, rid)] - maps[1][(seed, rid)])
        return _constant_difference_bound(result, [mean(v) for v in root_values.values()])
    aggregated = aggregate_rows(rows, metric)
    maps = []
    for method, condition, intervention in (left, right):
        maps.append({(r["seed"], r["root_id"]): (r["scenario_family"], r[metric]) for r in aggregated
                     if (r["method"], r["condition"], r["intervention"]) == (method, condition, intervention)})
    if not maps[0] or maps[0].keys() != maps[1].keys():
        raise ValueError("PAIRED_ROOTS_REQUIRED")
    seeds = sorted({s for s, _ in maps[0]})
    roots = sorted({r for _, r in maps[0]})
    families = defaultdict(list)
    for root in roots:
        if any((seed, root) not in maps[0] for seed in seeds):
            raise ValueError("Complete paired seeds required")
        family = maps[0][(seeds[0], root)][0]
        vector = []
        for seed in seeds:
            a, b = maps[0][(seed, root)], maps[1][(seed, root)]
            if a[0] != b[0] or a[0] != family:
                raise ValueError("Paired family mismatch")
            vector.append(a[1] - b[1])
        families[family].append(vector)
    arrays = {f: np.asarray(values, dtype=float) for f, values in sorted(families.items())}
    estimate = float(np.mean([array.mean() for array in arrays.values()]))
    per_seed = {str(seed): float(np.mean([array[:, i].mean() for array in arrays.values()])) for i, seed in enumerate(seeds)}
    rng = np.random.default_rng(7331)
    boot = []
    for offset in range(0, resamples, 256):
        count = min(256, resamples - offset)
        seed_indices = rng.integers(0, len(seeds), size=(count, len(seeds))) if sensitivity else None
        values = np.zeros(count)
        for array in arrays.values():
            idx = rng.integers(0, len(array), size=(count, len(array)))
            sampled = array[idx]
            if seed_indices is not None:
                sampled = np.take_along_axis(sampled, seed_indices[:, None, :], axis=2)
            values += sampled.mean(axis=(1, 2)) / len(arrays)
        boot.extend(values.tolist())
    result = {"estimate": estimate, "interval": np.quantile(boot, [.005, .995]).tolist(), "confidence": .99,
            "resamples": resamples, "analysis_seed": 7331, "independent_roots": len(roots),
            "training_seeds": seeds, "per_seed": per_seed, "seed_sensitivity": sensitivity,
            "roots_by_family": {f: len(a) for f, a in arrays.items()}, "implementation": "vectorized_paired_stratified_root_bootstrap"}
    return _constant_difference_bound(result, [float(v.mean()) for array in arrays.values() for v in array])


def primary_contrasts(rows, *, resamples=20000):
    """Always return all five contrasts, with explicit missing/inconclusive states."""
    main = [r for r in rows if r.get("split") == "test_id" and r.get("observation_condition") == "O2"
            and not r.get("intervention") and r.get("reader_id") == "fixed_symbolic"
            and r.get("extraction_track") == "REF" and not r.get("nonresponse_rate") and r["experiment"] == "E2"]
    e1 = [r for r in rows if r["experiment"] == "E1" and r.get("split") == "test_id"]
    e3 = [r for r in rows if r["experiment"] == "E3" and r.get("split") == "test_id"
          and r.get("trace", {}).get("episode", {}).get("opportunity", {}).get("eligible") is True]
    definitions = [
        ("PI1_O2_minus_O1_B0", e1, ("B0", "O2", None), ("B0", "O1", None), "C2"),
        ("PI2_I2_minus_I1_B1", e3, ("B1", "O2", "I2"), ("B1", "O2", "I1"), "FJS"),
        ("PI3_B2L_minus_B3", main, ("B2-L", "O2", None), ("B3", "O2", None), "FJS"),
        ("PI3_B2L_minus_B1", main, ("B2-L", "O2", None), ("B1", "O2", None), "FJS"),
        ("PI3_B2L_minus_B2I", main, ("B2-L", "O2", None), ("B2-I", "O2", None), "FJS"),
    ]
    results = []
    for name, eligible, left, right, metric in definitions:
        item = {"contrast": name, "left": left, "right": right, "metric": metric,
                "confidence": .99, "family_correction": "Bonferroni five contrasts"}
        methods = {r["method"] for r in eligible}
        if left[0] not in methods or right[0] not in methods:
            item.update(status="not_executed", reason="Missing required paired method/eligible test roots")
        else:
            try:
                statistic = _fast_bootstrap(eligible, left, right, metric, resamples)
                item.update(status="executed", statistics=statistic,
                            seed_sensitivity=_fast_bootstrap(eligible, left, right, metric, resamples, sensitivity=True))
                low, high = statistic["interval"]
                item["conclusion"] = "inconclusive" if low <= 0 <= high else "positive_difference" if low > 0 else "negative_difference"
                item["establishes_five_percentage_points"] = low >= .05
                if name.startswith("PI3"):
                    harms = {}
                    for harm in ("current_error", "unsupported_write_episode"):
                        if all(harm in r for r in eligible):
                            harms[harm] = _fast_bootstrap(eligible, left, right, harm, resamples)
                    item["harm_guard_intervals"] = harms
                    item["clean_improvement_guards_pass"] = len(harms) == 2 and all(v["interval"][1] <= .02 for v in harms.values())
                    if len(statistic["training_seeds"]) < 3:
                        item["limitation"] = "Fewer than three training seeds; incomplete planned comparison"
            except (ValueError, KeyError) as exc:
                item.update(status="inconclusive", reason=str(exc))
        results.append(item)
    return results


def _table(summaries, experiment):
    selected = [s for s in summaries if s["experiment"] == experiment]
    if not selected:
        return "Estado: `not_executed`. No hay trazas de este experimento.\n"
    text = "Estado: `executed` (exploratorio). Medias por raíz, familia y semilla.\n\n"
    text += "| Split | Método | Condición | Brazo | Control | Raíces | Episodios | C2 macro | FJS macro | Fallos |\n|---|---|---|---|---|---:|---:|---:|---:|---:|\n"
    for row in selected:
        number = lambda key: f"{row[key]:.4f}" if key in row else "N/A"
        text += f"| {row['split']} | {row['method']} | {row['observation_condition']} | {row['intervention'] or '—'} | {row.get('control_group') or '—'} | {row['independent_roots']} | {row['scheduled_episodes']} | {number('C2_macro')} | {number('FJS_macro')} | {row['technical_failures']} |\n"
    return text


def _contrast_table(contrasts):
    text = "| Contraste predefinido | Estado | Diferencia | IC 99 % | Conclusión |\n|---|---|---:|---|---|\n"
    for contrast in contrasts:
        stat = contrast.get("statistics")
        estimate = f"{stat['estimate']:.4f}" if stat else "N/A"
        interval = f"[{stat['interval'][0]:.4f}, {stat['interval'][1]:.4f}]" if stat else "N/A"
        text += f"| {contrast['contrast']} | {contrast['status']} | {estimate} | {interval} | {contrast.get('conclusion', contrast.get('reason', ''))} |\n"
    return text


def generate_executed_reports(trace_paths, destination, *, resamples=20000, metadata=None):
    """Read real traces and write reviewable reports without altering source files."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if isinstance(trace_paths, (str, Path)):
        trace_paths = [trace_paths]
    rows, provenance = [], []
    sources = [Path(path) for path in trace_paths]
    merged = destination / "TRAZAS.jsonl"
    if any(source.resolve() == merged.resolve() for source in sources):
        raise ValueError("REPORT_OUTPUT_WOULD_OVERWRITE_SOURCE_TRACES: choose a distinct report directory")
    with merged.open("wb") as target:
        for source in sources:
            digest = hashlib.sha256()
            with source.open("rb") as handle:
                for line in handle:
                    digest.update(line)
                    if line.strip():
                        rows.append(_flatten(json.loads(line)))
                        target.write(line.rstrip(b"\r\n") + b"\n")
            provenance.append({"path": str(source), "sha256": digest.hexdigest()})
    summaries = summarize_rows(rows, resamples=resamples)
    contrasts = primary_contrasts(rows, resamples=resamples)
    summary = {"status": "executed" if rows else "not_executed", "scope": "exploratory_automatic_audit",
               "source_traces": provenance, "scheduled_rows": len(rows), "groups": summaries,
               "primary_contrasts": contrasts, "metadata": metadata or {},
               "human_audit_status": "not_executed", "human_authored_external_status": "not_executed"}
    (destination / "ANALISIS.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    fields = ["experiment", "split", "method", "seed", "root_id", "latent_world_id", "realization_id", "scenario_family",
              "observation_condition", "intervention", "control_group", "reader_id", "extraction_track", "nonresponse_rate", "technical_status", "retry_count",
              "C2", "FJS", "current_error", "questions", "unsupported_current", "unsupported_write_episode", "invalid_outputs", "R_long", "R_short"]
    with (destination / "RESULTADOS.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    common = ("Ejecución exploratoria con auditoría automática. La revisión humana y las historias externas humanas están `not_executed`. "
              "Un banco AI_AUTHORED_PROXY se informa como proxy; no constituye evidencia humana independiente. "
              "Las comparaciones ausentes no se rellenan con estimaciones.\n\n")
    names = {"INFORME_OBSERVABILIDAD.md": ("PI1: observabilidad", "E1"),
             "INFORME_VALOR_PERSISTENCIA.md": ("PI2: adquisición y persistencia", "E3"),
             "INFORME_SFT_RL.md": ("PI3: SFT y continuaciones", "E2")}
    for filename, (title, experiment) in names.items():
        (destination / filename).write_text(f"# {title}\n\n" + common + _table(summaries, experiment) + "\n" + _contrast_table(contrasts) +
            "\nLos resultados por familia/semilla, las guardas de daño y los intervalos de sensibilidad están en ANALISIS.json. "
            "Un intervalo amplio que incluye cero es inconcluso y no demuestra equivalencia. La existencia de un adaptador no prueba mejora causada por RL.\n")
    general = [s for s in summaries if s["split"] in {"test_ood", "test_external"} or s["experiment"] in {"E4", "E5", "E6"}]
    (destination / "INFORME_GENERALIZACION.md").write_text("# Generalización y robustez\n\n" + common +
        ("Hay distribuciones de desafío ejecutadas; no se agrupan con ID.\n\n" if general else "Estado: `not_executed`. No hay trazas de generalización/robustez.\n\n") +
        "```json\n" + json.dumps(general, ensure_ascii=False, indent=2) + "\n```\n")
    (destination / "RESPUESTAS_DIRECTOR.md").write_text("# Respuestas por pregunta de investigación\n\n" + common + _contrast_table(contrasts) +
        "\nPI1 se interpreta como diferencia de evidencia disponible; PI2 como disponibilidad de aclaración para escribir; PI3 como recetas bajo presupuesto declarado. "
        "Sin guardas de daño satisfechas no se afirma mejora global. Los resultados negativos se conservan.\n")
    (destination / "LIMITACIONES_Y_DESVIACIONES.md").write_text("# Limitaciones y desviaciones\n\n" + common +
        "La ejecución exploratoria no reemplaza las puertas confirmatorias de la SPEC. El banco es sintético, el horizonte es de tres sesiones y el lector principal es simbólico. "
        "La cobertura de lenguaje del evaluador limita la interpretación de historias independientes. Los contrastes usan IC 99 % y raíces completas, condicionados a las semillas disponibles. "
        "Los tres entrenamientos no equivalen a miles de usuarios independientes.\n\nMetadatos de ejecución:\n\n```json\n" +
        json.dumps(metadata or {}, ensure_ascii=False, indent=2) + "\n```\n")
    from .training_artifacts import build_training_artifacts
    summary["training_artifacts"] = build_training_artifacts(metadata or {}, destination)
    (destination / "ANALISIS.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary
