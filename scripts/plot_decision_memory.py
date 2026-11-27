"""Observed controller-RL evidence; no smoothing or fabricated pending data."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    from scripts.plot_live_rl import BLUE, ORANGE, PURPLE, Evidence, setup, pending
    from scripts.plot_budget_memory import number, series, line, legend, metrics
except ModuleNotFoundError:
    from plot_live_rl import BLUE, ORANGE, PURPLE, Evidence, setup, pending
    from plot_budget_memory import number, series, line, legend, metrics
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

GREEN, GREY = '#009E73', '#788795'
METHODS = ('Head0','RL-selected','LM-frozen','FIFO','HierRandom','Public-rules')
LABELS = ('Cabeza\ninicial','RL\nelegido','Gestor LM\ncongelado','FIFO','Azar\njerárquico','Reglas\npúblicas')
ALIASES = {'Head0': ('Head0','Initial-head','Head-initial'),
           'LM-frozen': ('LM-frozen','Instruct-frozen'),
           'HierRandom': ('Hierarchical-random','Random','HierRandom')}


def _series(rows, names):
    """First explicitly supplied key, never a computed proxy for missing data."""
    for name in names:
        points = series(rows,name,nested='values')
        if points:
            return points
    return []


def make_figure(run,output,*,visitation='pi',credit_unit='action'):
    if visitation not in {'pi','mu'} or credit_unit not in {'action','session'}:
        raise ValueError('Declare visitation pi/mu and credit unit action/session')
    run,output=Path(run).resolve(),Path(output).resolve()
    evidence=Evidence()
    numerics=evidence.json(output/'QA_STEP1_NUMERICS.json') or {}
    fixed_profiles=evidence.json(output/'FIXED_TRAIN_STATE_PROFILES.json') or {}
    plans=[evidence.json(run/name) for name in ('PILOT_PLAN.json','PLAN.json','RL_PLAN.json','FIT_PLAN.json','TRAIN_PLAN.json')]
    plan=next((value for value in reversed(plans) if value),{})
    pilot=evidence.json(run/'PILOT_RESULT.json') or {}
    closure=evidence.json(run/'DIAGNOSTIC_CLOSURE.json') or {}
    directory=run/'optimizer'
    progress=evidence.jsonl(directory/'progress.jsonl')
    rollout=evidence.jsonl(directory/'training_metrics.jsonl')
    groups=evidence.jsonl(directory/'group_diagnostics.jsonl')
    training=evidence.json(directory/'training.json') or {}
    comparison=evidence.json(run/'COMPARISON.json') or {}
    profile=evidence.json(run/'POLICY_PROFILE.json') or {}
    updates=[row for row in progress if row.get('stage')=='training']
    validation={row['step']:metrics(row.get('metrics',row.get('validation',{}))) for row in progress
                if row.get('stage')=='validation_completed' and number(row.get('step'))}
    for row in training.get('history',[]):
        if isinstance(row,dict) and number(row.get('step')) and isinstance(row.get('validation'),dict):
            validation[row['step']]=metrics(row['validation'])
    observed=max((step for step,_ in series(updates,'loss')),default=0)
    planned=plan.get('updates',plan.get('max_updates',training.get('updates')))
    seed=plan.get('seed',training.get('seed','pendiente'))
    selected=comparison.get('selected_step',training.get('best_step'))
    finals={}
    for method in METHODS:
        item=next((comparison.get('panels',{}).get(alias) for alias in ALIASES.get(method,(method,))
                   if alias in comparison.get('panels',{})),None)
        value=metrics(item)
        if number(value.get('net_return')):
            finals[method]=value
    complete=len(finals)==len(METHODS) and comparison.get('status') in {'completed','complete','executed','passed'}
    stamp=datetime.now(ZoneInfo('America/Guayaquil')).strftime('%Y-%m-%d %H:%M %Z')
    stage=(progress[-1].get('stage','pendiente') if progress else
           f"piloto {pilot['status']}; sin ajuste" if pilot.get('status') else
           'piloto TRAIN; ajuste pendiente' if plan.get('pilot_groups') else 'sin diario')
    if closure:stage=closure.get('status',stage)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','savefig.facecolor':'white'})
    fig,axes=plt.subplots(2,3,figsize=(18,11))
    fig.subplots_adjust(left=.065,right=.96,bottom=.23,top=.84,hspace=.76,wspace=.4)
    fig.suptitle('Aprender cuándo preguntar, guardar y conservar memoria',x=.065,y=.97,ha='left',fontsize=21,fontweight='bold',color='#172B42')
    counts=f'{observed}/{planned}' if number(planned) else str(observed)
    stable=profile.get('version')=='stable_controller_v1'
    architecture_note=' · LN64 / π FP64' if stable else ''
    fig.text(.065,.924,f'Controlador RL sobre LLM congelado{architecture_note} · sin SFT local · semilla {seed} · actualizaciones {counts}',fontsize=11,color='#405267')
    fig.text(.065,.894,f'{run.name} · {stage} · corte {stamp}',fontsize=9,color='#607185')

    axis=axes[0,0]
    setup(axis,'A  Objetivo y regularización','Actualización','Pérdida')
    loss,policy,kl=[series(updates,key) for key in ('loss','policy_loss','kl')]
    line(axis,loss,label='Loss total',color=BLUE)
    line(axis,policy,label='Objetivo centrado · valor ≈0',color=ORANGE,linestyle='--')
    twin=None
    if kl:
        twin=axis.twinx();line(twin,kl,label='KL conjunta exacta · eje derecho',color=GREEN,marker='.')
        twin.set_ylabel('KL frente a cabeza inicial',fontsize=9,color=GREEN);twin.spines['top'].set_visible(False)
    if not (loss or policy or kl):pending(axis,'Sin actualizaciones completas')
    grad=series(updates,'grad_norm')
    if grad:axis.text(.02,.95,f'Última norma de gradiente: {grad[-1][1]:.3g}',transform=axis.transAxes,va='top',fontsize=8,color='#405267',bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
    legend(axis,*([twin] if twin else []))

    axis=axes[0,1]
    setup(axis,'B  Probabilidad aprendida de actuar','Actualización','Probabilidad de la gate positiva')
    ask=_series(rollout,('behavior/sampled_ask_gate_probability','gate/ask_probability','sampled_ask_gate_probability','visitation/sampled_ask_gate_probability'))
    store=_series(rollout,('behavior/sampled_store_gate_probability','gate/store_probability','sampled_store_gate_probability','visitation/sampled_store_gate_probability'))
    visit_label='exploración μ' if visitation=='mu' else 'on-policy π'
    line(axis,ask,label=f'ASK · llamada π en visita {visit_label}',color=ORANGE)
    line(axis,store,label=f'STORE · llamada π en visita {visit_label}',color=GREEN)
    if not (ask or store):pending(axis,'Sin gates de visitas muestreadas')
    if ask or store:axis.axhline(.5,color=GREY,linewidth=.8,linestyle='--')
    axis.set_ylim(-.03,1.05);axis.yaxis.set_major_formatter(PercentFormatter(1));legend(axis)
    if numerics.get('run')==str(run) and numerics.get('actual_max_ASK_change_pp') is not None:
        axis.text(.02,.03,'ASK paso 1 incluye residuo FP32/Adam; ver informe.',transform=axis.transAxes,fontsize=7.5,color='#405267',bbox={'facecolor':'white','alpha':.8,'edgecolor':'none'})
    fixed_steps=sorted(int(step) for step in fixed_profiles.get('profiles',{}) if str(step).isdigit() and int(step)<=observed)
    if fixed_steps:
        fixed_step=fixed_steps[-1];fixed=fixed_profiles['profiles'][str(fixed_step)]['summary']
        if 'ask' in fixed and 'write' in fixed:
            arange=fixed['ask']['positive_gate_max']-fixed['ask']['positive_gate_min']
            wrange=fixed['write']['positive_gate_max']-fixed['write']['positive_gate_min']
            axis.text(.02,.96,f'Estados TRAIN fijos · paso {fixed_step}\nRango ASK {arange:.1e} · STORE {wrange:.1e}',transform=axis.transAxes,va='top',fontsize=7.5,color='#405267',bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
    saturation=_series(rollout,('stability/hidden_saturation_fraction_max_state',))
    lnvariance=_series(rollout,('stability/post_layernorm_variance_max',))
    if stable and saturation:
        stability_note=f"LN64 · saturación máxima por target {saturation[-1][1]:.1%}"
        if lnvariance:stability_note+=f"\nVarianza tras LN: {lnvariance[-1][1]:.5f}"
        axis.text(.02,.79 if fixed_steps else .96,stability_note,transform=axis.transAxes,va='top',fontsize=7.5,color='#405267',bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})

    axis=axes[0,2]
    setup(axis,'C  Retornos de macros ASK × WRITE' if credit_unit=='session' else 'C  Retornos de las acciones exploradas','Actualización','Retorno neto local · TRAIN con juez IA')
    ranges_by_step={}
    for row in [*updates,*groups]:
        values=row.get('returns')
        step=row.get('step',row.get('optimizer_update'))
        if number(step) and isinstance(values,list) and values and all(number(value) for value in values):
            ranges_by_step[step]=(step,min(values),max(values))
    ranges=sorted(ranges_by_step.values())
    expected=_series(rollout,('credit/expected_return',))
    if ranges:
        axis.vlines([x[0] for x in ranges],[x[1] for x in ranges],[x[2] for x in ranges],color=BLUE,alpha=.3,label='Mínimo–máximo de forks correlacionados')
    expectation_label='Expectativa local ∑ π ASK×WRITE × Q' if credit_unit=='session' else 'Expectativa local ∑ π anterior × Q observada'
    if expected:axis.scatter(*zip(*expected),label=expectation_label,color=BLUE,s=22,zorder=4)
    if not (ranges or expected):pending(axis,'Sin grupos de ajuste evaluados')
    legend(axis)

    axis=axes[1,0]
    setup(axis,'D  Utilidad y éxito en VALIDATION','Checkpoint greedy','Retorno neto')
    returns=sorted((step,row['net_return']) for step,row in validation.items() if number(row.get('net_return')))
    successes=sorted((step,row['query_success']) for step,row in validation.items() if number(row.get('query_success')))
    line(axis,returns,label='VALIDATION · retorno neto',color=PURPLE,marker='D')
    twin=None
    if successes:
        twin=axis.twinx();line(twin,successes,label='Consultas correctas con soporte · eje derecho',color=GREEN,linestyle='--',marker='D')
        twin.set_ylim(-.03,1.05);twin.yaxis.set_major_formatter(PercentFormatter(1));twin.set_ylabel('Éxito por consulta',fontsize=9,color=GREEN)
        twin.spines['top'].set_visible(False)
    if not (returns or successes):pending(axis,'Validación inicial pendiente')
    legend(axis,*([twin] if twin else []))

    axis=axes[1,1]
    setup(axis,'E  Comportamiento greedy en VALIDATION','Checkpoint','Número medio por episodio')
    actions=[]
    for name,label,color in [('ask_count','ASK',ORANGE),('write_count','STORE',GREEN),('noop_count','NOOP',BLUE),('informative_ask_count','ASK informativa',PURPLE)]:
        points=sorted((step,row[name]) for step,row in validation.items() if number(row.get(name)))
        actions.extend(points);line(axis,points,label=label,color=color,linestyle='--' if name=='informative_ask_count' else '-',marker='D')
    if not actions:pending(axis,'Sin acciones greedy medidas')
    axis.set_ylim(-.12,4.25)
    handles,labels=axis.get_legend_handles_labels()
    if handles:axis.legend(handles,labels,ncol=2,fontsize=8,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.23))

    axis=axes[1,2]
    setup(axis,'F  Controles con igual lector y presupuesto','','Retorno neto · evaluación reservada')
    if finals:
        colors=[GREY,BLUE,PURPLE,GREEN,ORANGE,'#6F604A']
        positions=[index for index,method in enumerate(METHODS) if method in finals]
        bars=axis.bar(positions,[finals[METHODS[index]]['net_return'] for index in positions],color=[colors[index] for index in positions],width=.62)
        axis.bar_label(bars,labels=[f"{finals[METHODS[index]]['net_return']:+.3f}" for index in positions],padding=5,fontsize=8.5)
        labels=list(LABELS)
        if selected is not None:labels[1]=f'RL elegido\npaso {selected}'
        axis.set_xticks(range(len(METHODS)),labels,fontsize=8)
        for index,method in enumerate(METHODS):
            if method not in finals:axis.text(index,.5,'Pendiente',transform=axis.get_xaxis_transform(),ha='center',rotation=90,color=GREY,fontsize=8)
        axis.axhline(0,color=GREY,linewidth=.7);axis.margins(y=.23)
        rl,lm=finals.get('RL-selected',{}),finals.get('LM-frozen',{})
        if (number(rl.get('query_success')) and rl.get('query_success')==lm.get('query_success')
                and number(rl.get('joint_success')) and rl.get('joint_success')==lm.get('joint_success')):
            axis.text(.5,.96,f"RL = LM: {rl['query_success']:.0%} consultas / {rl['joint_success']:.0%} conjunto",transform=axis.transAxes,ha='center',va='top',fontsize=8,color='#405267',bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
    else:
        pending(axis,'Reservado sin ejecutar\nCierre diagnóstico con validación' if closure else 'Comparación reservada pendiente')
        axis.set_xticks([])
    latest=max([observed,*validation.keys()],default=0)
    for axis in list(axes.flat)[:5]:axis.set_xlim(-.4,max(1,latest)+.4)
    note=('Visitas de exploración μ: gates de llamadas reales de π; excluyen prefijos forzados, replay y acciones intervenidas.'
          if visitation=='mu' else 'Visitas on-policy π: gates de llamadas reales; excluyen replay, acciones forzadas y evaluación greedy.')
    fig.text(.065,.12,note,fontsize=9,color='#405267')
    note=('9/3 macros por sesión y forks correlacionados; expectativa TRAIN local en estados distintos, no retorno global. VAL usa π greedy.'
          if credit_unit=='session' else 'TRAIN enumera acciones legales; sus forks están correlacionados. VALIDATION conserva el criterio de éxito original.')
    fig.text(.065,.089,note,fontsize=9,color='#405267')
    fig.text(.065,.058,'Objetivo de política ≈0 aun con gradiente útil; loss total ≈ρβKL. Head0 es débil: comparar también LM congelado y reglas públicas.',fontsize=9,color='#405267')
    comparison_status='reservado no ejecutado; cierre diagnóstico' if closure else 'comparación completa' if complete else 'comparación pendiente/parcial'
    fig.text(.065,.027,f"Selección: {selected if selected is not None else 'pendiente'}. {comparison_status}. Un seed; sin suavizado ni datos imputados.",fontsize=9,color='#405267')
    output.mkdir(parents=True,exist_ok=True)
    outputs=[]
    for suffix in ('png','svg'):
        path=output/f'decision_memory_curves.{suffix}';fig.savefig(path,dpi=180);outputs.append(str(path))
    plt.close(fig)
    if stable:
        diagnostic,panels=plt.subplots(1,3,figsize=(16,5))
        diagnostic.subplots_adjust(left=.065,right=.97,bottom=.29,top=.79,wspace=.3)
        diagnostic.suptitle('Estabilidad de la cabeza normalizada · no es una medida de éxito',fontsize=16,fontweight='bold',x=.065,ha='left')
        diagnostic.text(.065,.86,f'{run.name} · {observed}/{planned or "?"} actualizaciones · cada punto usa targets públicos codificados del grupo',fontsize=9,color='#405267')
        axis=panels[0];setup(axis,'Activaciones tanh saturadas','Actualización','Fracción con |h| > 0,999')
        mean_sat=_series(rollout,('stability/hidden_saturation_fraction',))
        line(axis,mean_sat,label='Media de activaciones',color=BLUE)
        line(axis,saturation,label='Peor target del grupo',color=ORANGE,linestyle='--')
        axis.set_ylim(-.025,1.05);axis.yaxis.set_major_formatter(PercentFormatter(1))
        if not saturation:pending(axis,'Sin targets de ajuste registrados')
        legend(axis)
        axis=panels[1];setup(axis,'Varianza después de LayerNorm','Actualización','Máxima varianza entre targets')
        line(axis,lnvariance,label='LayerNorm fija · sin parámetros afines',color=GREEN)
        axis.set_ylim(-.025,1.15)
        if not lnvariance:pending(axis,'Sin mediciones de ajuste')
        legend(axis)
        axis=panels[2];setup(axis,'Activación antes de normalizar','Actualización','Magnitud absoluta de la primera capa')
        premean=_series(rollout,('stability/firstlinear_preactivation_absmean',))
        premax=_series(rollout,('stability/firstlinear_preactivation_absmax',))
        line(axis,premean,label='Media absoluta',color=PURPLE)
        line(axis,premax,label='Máximo absoluto',color=GREY,linestyle='--')
        if not premean:pending(axis,'Sin mediciones de ajuste')
        legend(axis)
        for axis in panels:axis.set_xlim(-.4,max(1,latest)+.4)
        diagnostic.text(.065,.04,'Encoder y lector permanecen congelados. Evitar saturación permite entrenar; no demuestra mejor decisión ASK/STORE/NOOP ni mayor utilidad.',fontsize=9,color='#405267')
        for suffix in ('png','svg'):
            path=output/f'controller_stability.{suffix}';diagnostic.savefig(path,dpi=180);outputs.append(str(path))
        plt.close(diagnostic)
    if complete:
        outcome,axis=plt.subplots(figsize=(14,7.5))
        outcome.subplots_adjust(left=.08,right=.97,bottom=.25,top=.76)
        outcome.suptitle('Consultas correctas y éxito conjunto en el reservado',x=.08,y=.96,ha='left',fontsize=18,fontweight='bold',color='#172B42')
        story_count=comparison.get('panels',{}).get('RL-selected',{}).get('episodes')
        sample_note=f'{story_count} historias / {2*story_count} consultas' if number(story_count) else 'Muestra reservada documentada en COMPARISON'
        outcome.text(.08,.90,f'{run.name} · {sample_note} · lector y presupuesto iguales',fontsize=10,color='#405267')
        setup(axis,'','Comparador','Proporción de éxito sustentado')
        positions=list(range(len(METHODS)));width=.34
        query=[finals[name]['query_success'] for name in METHODS]
        joint=[finals[name]['joint_success'] for name in METHODS]
        left=axis.bar([x-width/2 for x in positions],query,width,color=BLUE,label='Consultas correctas con soporte')
        right=axis.bar([x+width/2 for x in positions],joint,width,color=ORANGE,label='Ambas consultas correctas en la misma historia')
        for bars,values in ((left,query),(right,joint)):
            axis.bar_label(bars,labels=[f'{value:.0%}' for value in values],padding=5,fontsize=10)
        labels=list(LABELS);labels[1]=f'RL elegido\npaso {selected}'
        axis.set_xticks(positions,labels,fontsize=10)
        axis.set_ylim(0,1.06);axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.legend(ncol=2,frameon=False,fontsize=9,loc='upper center',bbox_to_anchor=(.5,1.14))
        axis.set_axisbelow(True)
        rl,lm=finals['RL-selected'],finals['LM-frozen']
        delta_query=rl['query_success']-lm['query_success'];delta_joint=rl['joint_success']-lm['joint_success']
        outcome.text(.08,.12,f'RL frente a LM congelado: Δ éxito por consulta {delta_query*100:+.0f} pp; Δ éxito conjunto {delta_joint*100:+.0f} pp.',fontsize=10,color='#405267')
        outcome.text(.08,.075,'TEST usa el verificador determinista congelado; los juicios IA se usaron en TRAIN. RL-terminal predeclarado se informa aparte.',fontsize=9,color='#405267')
        for suffix in ('png','svg'):
            path=output/f'comparison_outcomes.{suffix}';outcome.savefig(path,dpi=180);outputs.append(str(path))
        plt.close(outcome)
    summary={'run':str(run),'stage':stage,'observed_optimizer_updates':observed,'planned_updates':planned,'selected_step':selected,
             'gate_ask_points':len(ask),'gate_store_points':len(store),'training_counterfactual_groups':len(ranges),'local_expected_return_points':len(expected),
             'validation':sorted(validation.items()),'final_comparison_complete':complete,'final_panels':finals,
             'sources':evidence.files,'ignored_incomplete_final_lines':evidence.partial_lines,'outputs':outputs,
             'pilot_status':pilot.get('status'),'smoothing':False,'missing_data_imputed':False,'gate_source':'explicit sampled actor-call metrics; forced/replayed/greedy excluded',
             'visitation_distribution':visitation,'credit_unit':credit_unit,'terminal_diagnostic_excluded_from_main_comparison':True}
    summary['numerical_audit_linked']=bool(numerics)
    summary['fixed_state_profile_step']=fixed_steps[-1] if fixed_steps else None
    summary['diagnostic_closure']=closure or None
    summary['controller_policy_version']=profile.get('version')
    summary['stability_points']=len(saturation)
    (output/'decision_memory_plot_sources.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--visitation',choices=('pi','mu'),default='pi')
    parser.add_argument('--credit-unit',choices=('action','session'),default='action')
    options=parser.parse_args()
    print(json.dumps(make_figure(options.run,options.output,visitation=options.visitation,credit_unit=options.credit_unit),ensure_ascii=False,indent=2))
