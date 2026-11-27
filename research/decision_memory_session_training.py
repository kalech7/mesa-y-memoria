"""Joint ASK/WRITE Monte Carlo policy gradient over a sampled session.

Enumerating the two decisions together exposes their complementarity: useful
information must also be retained. Q is an observed terminal outcome, detached;
the reference KL is joint and local to the visited macro-state.
"""
import math
from research.src.artifacts import digest
from research.budget_memory_env import ASK_ACTIONS, WRITE_ACTIONS, fingerprint


def validate_binding(actor, group, rewards):
    """Fail closed before assigning scalar outcomes to conditional actions."""
    import torch
    d=group['diagnostics']; t=group['session_index']; ask=group['ask_target']
    prefix=group['prefix']; writes=group['write_targets']
    if (type(t) is not int or not 0<=t<4 or len(prefix)!=2*t
            or ask['step_id']!=2*t or ask['session_index']!=t
            or ask['observation']['decision_type']!='ask'
            or ask['observation']['session_index']!=t):
        raise ValueError('Session target identity or prefix length mismatch')
    asked=any(x['phase']=='ask' and x['action']!='PROCEED' for x in prefix)
    expected_ask=['PROCEED'] if asked else list(ASK_ACTIONS)
    if ask['candidates']!=expected_ask or len(writes)!=len(expected_ask):
        raise ValueError('ASK budget or conditional WRITE catalogue mismatch')
    expected=[(i,j) for i in range(len(expected_ask)) for j in range(3)]
    if (len(group['action_pairs'])!=len(expected) or len(rewards)!=len(expected)
            or len(group['branches'])!=len(expected)):
        raise ValueError('Full unique Cartesian action coverage is required')
    for k,(pair,(i,j),branch) in enumerate(zip(group['action_pairs'],expected,group['branches'])):
        write=writes[i]
        binding=dict(ask_index=i,write_index=j,branch_index=k,
                     ask_action=expected_ask[i],write_action=list(WRITE_ACTIONS)[j])
        if any(pair.get(key)!=value for key,value in binding.items()):
            raise ValueError('Macro pairs must uniquely cover the canonical Cartesian order')
        cf=branch['counterfactual']
        if (any(cf.get(key)!=value for key,value in binding.items())
                or cf.get('session_index')!=t or cf.get('sampling_contract')!=group['sampling_contract']
                or cf.get('actor_fingerprint')!=d['actor_after']):
            raise ValueError('Outcome branch is bound to a different action or actor')
        if (write['candidates']!=list(WRITE_ACTIONS) or write['step_id']!=2*t+1
                or write['session_index']!=t or len(branch['decisions'])!=8):
            raise ValueError('Conditional WRITE target identity mismatch')
        for row,target,action in ((branch['decisions'][2*t],ask,expected_ask[i]),
                                  (branch['decisions'][2*t+1],write,list(WRITE_ACTIONS)[j])):
            if (row['action']!=action or row['candidates']!=target['candidates']
                    or fingerprint(row['observation'])!=fingerprint(target['observation'])
                    or target['observation_sha256']!=fingerprint(target['observation'])):
                raise ValueError('Real branch action/observation does not match conditional target')
        for row,original in zip(branch['decisions'][:2*t],prefix):
            if row['action']!=original['action'] or fingerprint(row['observation'])!=fingerprint(original['observation']):
                raise ValueError('Real outcome has a different exploration prefix')
    if (d['actor_before']!=d['actor_after'] or d['asked_before']!=asked
            or d['session_index']!=t or d['session_selection_probability']!=.25
            or d['mixture_probability']!=.5 or not d['selected_session_before_rollin']
            or d['mixture_component'] not in ('on_policy','defer_ask')
            or (asked and d['mixture_component']=='defer_ask')
            or d['importance_detached'] is not True or d['importance_clipped'] is not False
            or d['importance_batch_renormalized'] is not False):
        raise ValueError('Incorrect exogenous prefix-mixture sampling contract')
    logs=[]
    if not asked:
        for row in prefix:
            if row['phase']!='ask':continue
            features=actor.feature(row['observation'])['feature_vector']
            with torch.no_grad():lp=actor.distribution(features,row['candidates'])
            logs.append(float(lp[row['candidates'].index('PROCEED')].cpu()))
        if (len(logs)!=t or len(d['proceed_prefix_logprobs'])!=t
                or any(abs(a-b)>2e-6 for a,b in zip(logs,d['proceed_prefix_logprobs']))):
            raise ValueError('Prefix density is not the actual fixed actor likelihood')
        r=math.exp(math.fsum(logs)); ratio=2*r/(1+r)
        if (abs(d['no_ask_prefix_logprob']-math.fsum(logs))>2e-6
                or abs(d['no_ask_prefix_probability']-r)>2e-6):
            raise ValueError('Incorrect unasked-prefix density')
    else:
        ratio=2.
        if d['proceed_prefix_logprobs'] or d['no_ask_prefix_logprob'] is not None:
            raise ValueError('Spent-budget density must use the full-support mixture branch')
    if (abs(float(group['selection_weight'])-ratio)>2e-6
            or abs(d['prefix_importance']-ratio)>2e-6
            or abs(d['prefix_log_importance']-math.log(ratio))>2e-6):
        raise ValueError('Detached prefix importance ratio does not match the proposal')


def joint_logprobs(actor, group, *, head=None):
    import torch
    ask = group['ask_target']
    lp_ask = actor.distribution(ask['metadata']['feature_vector'], ask['candidates'], head=head)
    lp_write = [actor.distribution(target['metadata']['feature_vector'], target['candidates'], head=head)
                for target in group['write_targets']]
    return torch.stack([lp_ask[pair['ask_index']] + lp_write[pair['ask_index']][pair['write_index']]
                        for pair in group['action_pairs']])


def session_loss(logprobs, reference_logprobs, returns, weight, beta=.02):
    import torch
    if (logprobs.ndim != 1 or logprobs.shape != reference_logprobs.shape or logprobs.shape != returns.shape
            or not 0 < weight <= 2 or beta < 0):
        raise ValueError('Invalid macro-distribution, prefix ratio or regularizer')
    if not all(bool(torch.isfinite(x).all()) for x in (logprobs,reference_logprobs,returns)):
        raise ValueError('Nonfinite macro action distribution or return')
    probabilities=logprobs.exp()
    if abs(float(probabilities.detach().sum().cpu())-1)>1e-5:
        raise ValueError('Macro action probabilities are not normalized')
    q=returns.detach()
    baseline=(probabilities.detach()*q).sum().detach()
    advantages=q-baseline
    if bool(q.eq(q[0]).all()):advantages=torch.zeros_like(q)
    policy=-(probabilities*advantages).sum()
    kl=(logprobs.sum()*0 if logprobs.detach().equal(reference_logprobs) else
        (probabilities*(logprobs-reference_logprobs)).sum())
    return weight*(policy+beta*kl),policy,kl,baseline,advantages


def optimize_session(actor, reference_head, group, rewards, optimizer, *, beta=.02):
    import torch
    if group['sampling_contract'] != 'session_joint_prefix_mixture_IS_v1':
        raise ValueError('Unsupported macro-state sampling contract')
    if group['diagnostics']['actor_after'] != actor.credit_parameter_fingerprint():
        raise ValueError('Actor changed during macro outcome collection')
    if len(group['action_pairs']) != len(rewards) or len(rewards) != len(group['branches']):
        raise ValueError('Every legal macro action requires its own real terminal outcome')
    validate_binding(actor,group,rewards)
    targets=[group['ask_target'],*group['write_targets']]
    for target in targets:
        metadata=target['metadata']
        if (metadata['profile_sha256'] != actor.policy_profile()['profile_sha256']
                or metadata['candidates'] != target['candidates']
                or digest(metadata['feature_vector']) != metadata['feature_sha256']
                or actor.feature(target['observation'])['feature_sha256'] != metadata['feature_sha256']):
            raise ValueError('Public feature, profile or catalogue mismatch')
    lp=joint_logprobs(actor,group)
    old=torch.tensor([group['ask_target']['metadata']['old_logprobs'][pair['ask_index']]
                      + group['write_targets'][pair['ask_index']]['metadata']['old_logprobs'][pair['write_index']]
                      for pair in group['action_pairs']],device=actor.device)
    error=float((lp.detach()-old).abs().max().cpu())
    if error>2e-6:raise ValueError('Collection/training joint probabilities differ')
    with torch.no_grad():ref=joint_logprobs(actor,group,head=reference_head)
    q=torch.tensor([r['episodic_reward'] for r in rewards],device=actor.device,dtype=lp.dtype)
    losses=session_loss(lp,ref,q,float(group['selection_weight']),beta)
    optimizer.zero_grad(set_to_none=True);losses[0].backward()
    params=list(actor.head.parameters())
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in params):
        raise ValueError('Nonfinite controller gradient')
    norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():after=joint_logprobs(actor,group).exp().cpu().tolist()
    return dict(loss=float(losses[0].detach().cpu()),policy_loss=float(losses[1].detach().cpu()),
                kl=float(losses[2].detach().cpu()),baseline=float(losses[3].cpu()),
                advantages=losses[4].cpu().tolist(),returns=q.cpu().tolist(),
                probabilities_before=lp.detach().exp().cpu().tolist(),probabilities_after=after,
                grad_norm=float(norm.cpu()),selection_weight=float(group['selection_weight']),
                max_old_distribution_error=error)
