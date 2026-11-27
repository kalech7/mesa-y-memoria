"""Generator/oracle contract tests; no parser or resolver imports."""
import ast
import copy
import inspect
import json
import unittest
from collections import Counter
from datetime import datetime

from data_generation import food_scenarios
from data_generation.food_scenarios import CATEGORIES, COMBINATIONS, make_food_dataset, validate_food_dataset


class FoodScenarioTests(unittest.TestCase):
    def setUp(self):
        self.public, self.gold, self.private = make_food_dataset()

    def test_three_required_seeds_reproducible_and_balanced(self):
        signatures = []
        for seed in (42, 43, 44):
            first = make_food_dataset(seed)
            self.assertEqual(first, make_food_dataset(seed))
            public, gold, private = first
            self.assertEqual(len(public), 240)
            self.assertEqual(Counter(c['category'] for c in public), {category: 12 for category in CATEGORIES})
            self.assertEqual(validate_food_dataset(public, gold, private)['splits'], {'dev': 120, 'test': 60, 'validation': 60})
            signatures.append(json.dumps(public, sort_keys=True))
        self.assertEqual(len(set(signatures)), 3)

    def test_public_projection_disjoint_from_hidden_and_gold(self):
        serialized = json.dumps(self.public)
        for key in ('true_state', 'uncommunicated', 'clarification_responses', 'acceptable_actions', 'expected_food_items', 'rationale'):
            self.assertNotIn(f'"{key}"', serialized)
        tampered = copy.deepcopy(self.public)
        tampered[0]['events'][0]['true_state'] = {'food': 'SECRET'}
        with self.assertRaises(ValueError):
            validate_food_dataset(tampered, self.gold, self.private)

    def test_users_and_wording_families_are_split_disjoint(self):
        for field in ('user_id', 'family'):
            values = {split: {case[field] for case in self.public if case['split'] == split} for split in ('dev', 'validation', 'test')}
            self.assertFalse(values['dev'] & values['validation'])
            self.assertFalse(values['dev'] & values['test'])
            self.assertFalse(values['validation'] & values['test'])
        # All requirements are evaluated in every split; scenarios, not categories, are retained.
        for split in ('dev', 'validation', 'test'):
            self.assertEqual({case['category'] for case in self.public if case['split'] == split}, set(CATEGORIES))

    def test_eight_combinations_covered_independently_and_have_competing_memories(self):
        rows = [(p, g) for p, g in zip(self.public, self.gold) if p['category'] == 'S08']
        self.assertEqual({(g['expected_context']['day_scope'], g['expected_context']['meal_scope']) for _, g in rows}, set(COMBINATIONS))
        for public, gold in rows:
            self.assertEqual(len(public['events']), 8)
            self.assertEqual(len(gold['expected_evidence_ids']), 1)
            self.assertEqual(len(gold['relevant_evidence_ids']), 8)

    def test_expectations_trace_to_visible_evidence_and_retained_originals(self):
        for public, gold in zip(self.public, self.gold):
            messages = {e['message_id']: e for e in public['events']}
            self.assertLessEqual(set(gold['expected_evidence_ids']), set(messages))
            self.assertLessEqual(set(gold['relevant_evidence_ids']), set(messages))
            for event in public['events']:
                self.assertLess(datetime.fromisoformat(event['at']), datetime.fromisoformat(public['reference_time']))
            if public['category'] in {'S10', 'S12', 'S13'}:
                self.assertGreaterEqual(len(gold['expected_evidence_ids']), 2)
                self.assertFalse(gold['expected_food_items'])
            if public['category'] == 'S11':
                self.assertEqual(public['events'][1]['supersedes'], [public['events'][0]['message_id']])
                self.assertIn('pollo', public['events'][0]['text'])

    def test_resolved_context_partition_is_not_an_unresolved_conflict(self):
        self.assertEqual(food_scenarios.GOLD_VERSION, "v2")
        for public, gold in zip(self.public, self.gold):
            if public['category'] == 'S09':
                self.assertIsNone(gold['expected_conflict_type'])
                self.assertEqual(gold['acceptable_actions'], ['ANSWER'])
                self.assertEqual(len(public['events']), 2)
            if public['category'] == 'S20':
                self.assertIn('Todos los días', public['events'][0]['text'])

    def test_unknown_context_never_implicitly_any(self):
        for public, gold in zip(self.public, self.gold):
            if public['category'] in {'S14', 'S15'}:
                self.assertEqual(gold['expected_context'], {'day_scope': None, 'meal_scope': None})
                self.assertEqual(gold['acceptable_actions'], ['CLARIFY'])
                self.assertTrue(gold['requires_clarification'])
                hidden = next(row for row in self.private if row['scenario_id'] == public['scenario_id'])
                expected_slot = 'day_scope' if public['category'] == 'S14' else 'meal_scope'
                self.assertEqual(set(hidden['clarification_responses']), {expected_slot})

    def test_temporal_fixture_contains_before_during_and_after_without_relabeling(self):
        rows = [(p, g) for p, g in zip(self.public, self.gold) if p['category'] == 'S20']
        self.assertEqual(Counter(g['expected_context']['date'] for _, g in rows),
                         {'2026-10-18': 4, '2026-10-19': 4, '2026-10-20': 4})
        for public, gold in rows:
            self.assertEqual(gold['expected_food_items'], ['pizza'] if '2026-10-19' in public['query'] else ['sopa'])
            self.assertEqual(len(public['events']), 2)

    def test_restrictions_and_polarity_have_explicit_safety_expectations(self):
        for public, gold in zip(self.public, self.gold):
            if public['category'] == 'S16':
                self.assertFalse(gold['expected_food_items'])
                self.assertEqual(len(gold['forbidden_food_items']), 1)
            if public['category'] == 'S17':
                self.assertEqual(gold['expected_food_items'], ['fruta'])
                self.assertEqual(gold['forbidden_food_items'], ['nueces'])
                self.assertIn('alergia', public['events'][0]['text'])

    def test_oracle_generator_has_no_inference_imports(self):
        tree = ast.parse(inspect.getsource(food_scenarios))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or '').startswith(('simulator', 'memory', 'agents', 'evaluation')))
            if isinstance(node, ast.Import):
                self.assertFalse(any(alias.name.startswith(('simulator', 'memory', 'agents', 'evaluation')) for alias in node.names))

    def test_bad_size_and_untraceable_gold_rejected(self):
        with self.assertRaises(ValueError):
            make_food_dataset(instances=19)
        tampered = copy.deepcopy(self.gold)
        tampered[0]['expected_evidence_ids'] = ['invented']
        with self.assertRaises(ValueError):
            validate_food_dataset(self.public, tampered, self.private)


if __name__ == '__main__':
    unittest.main()
