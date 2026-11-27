from types import SimpleNamespace

import pytest

from research.development_review import BudgetBackend, probe_episodes, summarize
from research.src.artifacts import digest
from research.src.data.development_v2 import build_development_v2


def test_summary_does_not_count_worlds_or_paraphrases_as_independent_users():
    episodes, rows = [], []
    for root, world, realization, success in [('a', 'x', '0', 1), ('a', 'x', '1', 1),
                                             ('a', 'y', '0', 0), ('b', 'z', '0', 0)]:
        identity = dict(root_id=root, latent_world_id=world, realization_id=realization)
        episodes.append({**identity, 'authoring': {'stratum': 's'}})
        rows.append({**identity, 'metrics': dict(C2=success, FJS=success, questions=0,
                     invalid_outputs=0, unsupported_operations=0),
                     'rewards_by_component': dict(R_long=success * 2, R_short=success)})
    result = summarize(rows, episodes)
    assert result['root_weighted']['FJS'] == .25
    assert result['counts']['FJS'] == 2
    assert result['roots'] == 2


def test_review_backend_rejects_unplanned_prefixes_and_enforces_budget_before_inference():
    class Backend:
        config = SimpleNamespace(max_new_tokens=64)
        calls = 0
        def generate(self, prompt, sample=False):
            self.calls += 1
            return prompt
    raw = Backend()
    backend = BudgetBackend(raw, {digest({'prompt': 'public', 'max_new_tokens': 64})}, 1)
    with pytest.raises(ValueError, match='Unplanned'):
        backend.generate('hidden')
    assert raw.calls == 0
    assert backend.generate('public') == 'public'
    with pytest.raises(RuntimeError, match='budget'):
        backend.generate('public')
    assert raw.calls == 1


def test_sampled_roots_are_balanced_and_preserve_all_compatible_worlds():
    episodes = build_development_v2()
    selected = probe_episodes(episodes)
    assert len(selected) == 12
    assert len({e['root_id'] for e in selected}) == 10
    for stratum in {e['authoring']['stratum'] for e in episodes}:
        assert len({e['root_id'] for e in selected if e['authoring']['stratum'] == stratum}) == 2
    assert all(e['realization_id'] == 'v0' for e in selected)
    for root in {e['root_id'] for e in selected}:
        assert {e['latent_world_id'] for e in selected if e['root_id'] == root} == {
            e['latent_world_id'] for e in episodes if e['root_id'] == root}
