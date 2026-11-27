"""FP64 replay of the unchanged session-macro RL objective.

The original source remains archived and unchanged. This adapter keeps stored
log probabilities in the controller's FP64 distribution dtype, rather than
silently rounding them to FP32 before its collection/replay guard.
"""
from research.decision_memory_session_training import joint_logprobs, session_loss, validate_binding
from research.src.artifacts import digest


def optimize_session(actor, reference_head, group, rewards, optimizer, *, beta=.02):
    import torch
    if group['sampling_contract'] != 'session_joint_prefix_mixture_IS_v1':
        raise ValueError('Unsupported macro-state sampling contract')
    if group['diagnostics']['actor_after'] != actor.credit_parameter_fingerprint():
        raise ValueError('Actor changed during macro outcome collection')
    validate_binding(actor,group,rewards)
    for target in [group['ask_target'],*group['write_targets']]:
        metadata=target['metadata']
        if (metadata['profile_sha256'] != actor.policy_profile()['profile_sha256']
                or metadata['candidates'] != target['candidates']
                or digest(metadata['feature_vector']) != metadata['feature_sha256']
                or actor.feature(target['observation'])['feature_sha256'] != metadata['feature_sha256']):
            raise ValueError('Public feature, profile or catalogue mismatch')
    lp=joint_logprobs(actor,group)
    if lp.dtype != torch.float64:
        raise ValueError('Stable controller requires FP64 action distributions')
    old=torch.tensor([group['ask_target']['metadata']['old_logprobs'][pair['ask_index']]
                      + group['write_targets'][pair['ask_index']]['metadata']['old_logprobs'][pair['write_index']]
                      for pair in group['action_pairs']],device=actor.device,dtype=lp.dtype)
    error=float((lp.detach()-old).abs().max().cpu())
    if error>1e-12:raise ValueError('Collection/training joint probabilities differ')
    with torch.no_grad():ref=joint_logprobs(actor,group,head=reference_head)
    q=torch.tensor([reward['episodic_reward'] for reward in rewards],device=actor.device,dtype=lp.dtype)
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
                max_old_distribution_error=error,distribution_dtype='torch.float64')
