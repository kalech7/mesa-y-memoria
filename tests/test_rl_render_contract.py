"""Author-facing language contract, independent of parser and policy outputs."""
from copy import deepcopy
import unittest

from rl_feasibility.scenario_generator import _text, semantic_fingerprint, generate_scenarios


class AuthoredEvidenceRenderingTests(unittest.TestCase):
    @staticmethod
    def record(day='ANY', meal='ANY', source='USER_STATEMENT', polarity='LIKE'):
        return {'food_item': 'arroz', 'polarity': polarity, 'day_scope': day,
                'meal_scope': meal, 'source': source, 'context': {}}

    def test_every_general_scope_is_explicit_in_every_statement_family(self):
        day_words = {'WEEKDAY': 'días laborales', 'WEEKEND': 'fines de semana', 'ANY': 'cualquier día'}
        meal_words = {'BREAKFAST': 'desayuno', 'LUNCH': 'almuerzo',
                      'DINNER': 'cena', 'SNACK': 'merienda', 'ANY': 'cualquier tipo de comida'}
        for split in ('train', 'validation', 'test', 'shift'):
            for day, day_phrase in day_words.items():
                for meal, meal_phrase in meal_words.items():
                    with self.subTest(split=split, day=day, meal=meal):
                        record = self.record(day, meal)
                        before = deepcopy(record)
                        text = _text(record, split)
                        self.assertIn(day_phrase, text)
                        self.assertIn(meal_phrase, text)
                        self.assertEqual(record, before)
                        if day != 'ANY':
                            self.assertNotIn('cualquier día', text)
                        if meal != 'ANY':
                            self.assertNotIn('cualquier tipo de comida', text)

    def test_unknown_scopes_are_not_rendered_as_general(self):
        text = _text(self.record(None, None), 'train')
        self.assertNotIn('cualquier', text)
        self.assertNotIn('general', text)

    def test_order_history_describes_ordering_instead_of_stated_taste(self):
        for split in ('train', 'validation', 'test', 'shift'):
            text = _text(self.record(source='ORDER_HISTORY'), split)
            self.assertIn('pedí arroz', text)
            self.assertIn('cualquier día', text)
            self.assertIn('cualquier tipo de comida', text)
            for declarative_taste in ('me gusta', 'prefiero', 'disfruto', 'me quedo con'):
                self.assertNotIn(declarative_taste, text)

    def test_predefined_statement_wording_keeps_fold_stressors(self):
        for split, cue in {'train': 'me gusta', 'validation': 'prefiero',
                           'test': 'disfruto comer', 'shift': 'me quedo con'}.items():
            self.assertIn(cue, _text(self.record(), split))
            self.assertIn('no me gusta', _text(self.record(polarity='DISLIKE'), split))

    def test_company_place_and_temporal_uncertainty_remain_explicit(self):
        record = self.record()
        record.update(context={'company': 'friends', 'place': 'work'}, stability='OSCILLATING')
        text = _text(record, 'test')
        for phrase in ('con amigos', 'en la oficina', 'mis gustos cambian', 'no estoy seguro'):
            self.assertIn(phrase, text)

    def test_rendering_surface_does_not_change_semantic_state_fingerprint(self):
        record = self.record()
        scenario = {'oracle_preferences': [record], 'oracle_restrictions': [],
                    'visible_context': {}, 'latent': {'context': {}, 'acceptable_foods': ['arroz'],
                                                      'unresolvable': False}}
        fingerprint = semantic_fingerprint(scenario)
        for split in ('train', 'validation', 'test', 'shift'):
            scenario['events'] = [{'text': _text(record, split)}]
            self.assertEqual(semantic_fingerprint(scenario), fingerprint)


class RestrictionRenderingTests(unittest.TestCase):
    def test_explicit_allergy_scopes_match_authored_general_restriction(self):
        for split in ('train', 'validation', 'test', 'shift'):
            scenario = next(s for s in generate_scenarios(12, 42, split) if s['category'] == 'RESTRICTION')
            for restriction in scenario['oracle_restrictions']:
                evidence = next(e for e in scenario['events'] if e['id'] == restriction['evidence_id'])
                self.assertEqual(restriction['day_scope'], 'ANY')
                self.assertEqual(restriction['meal_scope'], 'ANY')
                self.assertIn('cualquier día', evidence['text'])
                self.assertIn('cualquier tipo de comida', evidence['text'])
                self.assertIn('alergia a ' + restriction['food_item'], evidence['text'])
                self.assertNotIn('me gusta', evidence['text'])
