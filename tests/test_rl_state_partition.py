"""Regression for the documented v1 train/test latent-state overlap."""
from collections import Counter
from copy import deepcopy
import unittest

from rl_feasibility.scenario_generator import (CATEGORIES, SPLIT_STATE_BUCKETS,
    generate_scenarios, semantic_fingerprint, state_bucket)


class SemanticStatePartitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 100 authored examples per category in each fold, not just distinct IDs.
        cls.folds = {split: generate_scenarios(1200, 5042 + i, split)
                     for i, split in enumerate(('train', 'validation', 'test'))}
        cls.signatures = {split: {semantic_fingerprint(s) for s in scenarios}
                          for split, scenarios in cls.folds.items()}

    def test_native_states_disjoint_in_all_three_fold_pairs(self):
        for left, right in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
            with self.subTest(pair=(left, right)):
                self.assertFalse(self.signatures[left] & self.signatures[right])

    def test_all_categories_have_one_hundred_instances_per_fold(self):
        for split, scenarios in self.folds.items():
            self.assertEqual(Counter(s['category'] for s in scenarios),
                             Counter({category: 100 for category in CATEGORIES}))

    def test_every_state_belongs_to_its_assigned_buckets(self):
        for split, scenarios in self.folds.items():
            self.assertTrue(all(state_bucket(s) in SPLIT_STATE_BUCKETS[split] for s in scenarios))

    def test_rejection_sampling_reproducible_preserves_small_category_coverage(self):
        first = generate_scenarios(12, 42)
        self.assertEqual(first, generate_scenarios(12, 42))
        self.assertEqual({s['category'] for s in first}, set(CATEGORIES))

    def test_hash_ignores_ids_surface_wording_split_and_record_order(self):
        original = self.folds['train'][0]
        changed = deepcopy(original)
        changed.update(scenario_id='new_id', user_id='new_user', split='test', category='renamed',
                       query='Otra redacción de la consulta.')
        changed['oracle_preferences'].reverse()
        for record in changed['oracle_preferences']:
            record.update(preference_id='different', user_id='another', evidence_id='new_evidence')
        changed['response_map']['meal']['text'] = 'Otra redacción de la respuesta.'
        self.assertEqual(semantic_fingerprint(original), semantic_fingerprint(changed))

    def test_hash_changes_when_latent_truth_changes(self):
        original = self.folds['train'][0]
        changed = deepcopy(original)
        changed['latent']['unresolvable'] = not changed['latent']['unresolvable']
        self.assertNotEqual(semantic_fingerprint(original), semantic_fingerprint(changed))

    def test_shift_remains_in_heldout_state_partition(self):
        shift = generate_scenarios(120, 5050, split='shift', distribution='shift')
        signatures = {semantic_fingerprint(s) for s in shift}
        self.assertFalse(signatures & self.signatures['train'])
        self.assertFalse(signatures & self.signatures['validation'])
        self.assertTrue(all(state_bucket(s) in (8, 9) for s in shift))
