"""Semantic contracts independent of SPEC-001's generated expected outputs."""
import unittest
from dataclasses import asdict
from simulator.preferences import parse_context, extract_preferences
from simulator.user import UserSimulator, interpret_question


def event(text, **kwargs):
    return {'message_id': 'e1', 'at': '2026-10-19T12:00:00+00:00', 'role': 'user',
            'source': 'USER_STATEMENT', 'text': text, **kwargs}


class FoodParserTests(unittest.TestCase):
    def test_four_meals_independently(self):
        for word, expected in [('desayuno', 'BREAKFAST'), ('almuerzo', 'LUNCH'), ('cena', 'DINNER'), ('merienda', 'SNACK')]:
            with self.subTest(meal=word):
                self.assertEqual(parse_context(f'¿Qué prefiero en el {word}?').meal_scope, expected)

    def test_configurable_meal_alias_does_not_change_default(self):
        self.assertIsNone(parse_context('comida principal').meal_scope)
        self.assertEqual(parse_context('comida principal', aliases={'LUNCH': ['comida principal']}).meal_scope, 'LUNCH')

    def test_days_ranges_and_partitions(self):
        cases = [('lunes a viernes', 'WEEKDAY', (0, 1, 2, 3, 4)), ('días laborales', 'WEEKDAY', (0, 1, 2, 3, 4)),
                 ('fines de semana', 'WEEKEND', (5, 6)), ('sábados y domingos', 'WEEKEND', (5, 6)), ('martes', 'WEEKDAY', (1,))]
        for text, scope, weekdays in cases:
            with self.subTest(text=text):
                parsed = parse_context(text)
                self.assertEqual((parsed.day_scope, parsed.weekdays), (scope, weekdays))

    def test_all_eight_day_meal_combinations(self):
        for days, day_scope in [('Entre semana', 'WEEKDAY'), ('Fines de semana', 'WEEKEND')]:
            for meal, expected in [('desayuno', 'BREAKFAST'), ('almuerzo', 'LUNCH'), ('cena', 'DINNER'), ('merienda', 'SNACK')]:
                with self.subTest(day=days, meal=meal):
                    parsed = parse_context(f'{days} para el {meal}')
                    self.assertEqual((parsed.day_scope, parsed.meal_scope), (day_scope, expected))

    def test_unknown_is_not_any(self):
        parsed = parse_context('Me gusta el tofu')
        self.assertIsNone(parsed.day_scope)
        self.assertIsNone(parsed.meal_scope)
        explicit = parse_context('En general, me gusta tofu')
        self.assertEqual((explicit.day_scope, explicit.meal_scope), ('ANY', 'ANY'))

    def test_dates_and_reference_are_deterministic(self):
        self.assertEqual(parse_context('cena del 2026-10-19').date, '2026-10-19')
        self.assertEqual(parse_context('almuerzo de mañana', '2026-10-19').date, '2026-10-20')
        self.assertEqual(parse_context('pasado mañana', '2026-10-19').date, '2026-10-21')
        self.assertIn('relative_date_without_reference', parse_context('hoy').unknown_conditions)
        self.assertIn('invalid_date', parse_context('2026-02-31').unknown_conditions)
        self.assertIsNone(parse_context('desde el 2026-10-15').date)

    def test_polarities_and_trace(self):
        for wording, polarity in [('me gusta', 'LIKE'), ('prefiero', 'LIKE'), ('no me gusta', 'DISLIKE'),
                                  ('no me agrada', 'DISLIKE'), ('me disgusta', 'DISLIKE'), ('me da igual', 'NEUTRAL')]:
            with self.subTest(wording=wording):
                extracted = extract_preferences(event(f'En general {wording} tofu.'), 'u')
                self.assertEqual(len(extracted.preferences), 1)
                preference = extracted.preferences[0]
                self.assertEqual((preference.polarity, preference.evidence_id, preference.user_id), (polarity, 'e1', 'u'))
                self.assertEqual(preference.timestamp, '2026-10-19T12:00:00+00:00')
                self.assertIsNone(preference.confidence)

    def test_context_conditions_and_unknown_condition(self):
        preference = extract_preferences(event('En casa, con mis amigos, me gusta pizza.'), 'u').preferences[0]
        self.assertEqual(preference.context, {'place': 'home', 'company': 'friends'})
        unknown = extract_preferences(event('Cuando llueve me gusta sopa.'), 'u').preferences[0]
        self.assertIn('unparsed_condition', unknown.scope_unknown)

    def test_allergy_separate_from_dislike_and_uncertain_not_safe(self):
        for text, certainty in [('Tengo alergia a las nueces.', 'CONFIRMED'), ('Puede que tenga alergia a las nueces.', 'UNCERTAIN')]:
            extracted = extract_preferences(event(text), 'u')
            self.assertFalse(extracted.preferences)
            self.assertEqual((extracted.restrictions[0].restriction_type, extracted.restrictions[0].certainty), ('ALLERGY', certainty))
        self.assertFalse(extract_preferences(event('No tengo alergia a las nueces.'), 'u').restrictions)

    def test_explicit_update_and_exception_preserve_evidence(self):
        update = extract_preferences(event('Corrijo: desde el 2026-10-15 prefiero lentejas en lugar de pollo para cualquier comida y cualquier día.', supersedes=['old']), 'u')
        self.assertEqual([(p.food_item, p.polarity) for p in update.preferences], [('lentejas', 'LIKE'), ('pollo', 'DISLIKE')])
        self.assertTrue(all(p.explicit_update and p.supersedes == ('old',) and p.valid_from == '2026-10-15' for p in update.preferences))
        exception = extract_preferences(event('Solo el 2026-10-19 para la cena prefiero pizza en lugar de sopa.'), 'u')
        self.assertTrue(all(p.temporary and p.valid_from == p.valid_to == '2026-10-19' for p in exception.preferences))

    def test_orders_keep_source_and_assistant_not_attributed(self):
        order = extract_preferences(event('Pedí carne.', source='order_history'), 'u').preferences[0]
        self.assertEqual(order.source, 'ORDER_HISTORY')
        self.assertFalse(extract_preferences(event('Me gusta carne.', role='assistant'), 'u').preferences)
        self.assertFalse(extract_preferences(event('Mi hijo prefiere arroz.'), 'u').preferences)

    def test_private_fields_are_ignored(self):
        visible = event('En general me gusta arroz.')
        poisoned = {**visible, 'hidden_state': {'preference': 'pollo'}, 'expected_food_items': ['pollo'], 'gold': 'pollo'}
        self.assertEqual(asdict(extract_preferences(visible, 'u')), asdict(extract_preferences(poisoned, 'u')))

    def test_oscillation_retains_both_visible_propositions(self):
        parsed = extract_preferences(event('En general, a veces me gusta carne y otras veces no me gusta carne.'), 'u')
        self.assertEqual([p.polarity for p in parsed.preferences], ['LIKE', 'DISLIKE'])
        self.assertTrue(all(p.stability == 'OSCILLATING' for p in parsed.preferences))

    def test_separate_sentence_scopes_and_invalid_metadata(self):
        parsed = extract_preferences(event('En el desayuno prefiero avena. En el almuerzo prefiero arroz.'), 'u')
        self.assertEqual([p.meal_scope for p in parsed.preferences], ['BREAKFAST', 'LUNCH'])
        self.assertTrue(all('multiple_meals' not in p.scope_unknown for p in parsed.preferences))
        invalid = extract_preferences(event('En general me gusta arroz.', valid_from='2026-02-31'), 'u')
        self.assertIsNone(invalid.preferences[0].valid_from)
        self.assertIn('invalid_valid_from', invalid.preferences[0].scope_unknown)

    def test_breakfast_word_never_introduces_negation(self):
        for phrase in ['En el desayuno me gusta avena.', 'En el desayuno prefiero avena.']:
            parsed = extract_preferences(event(phrase), 'u')
            self.assertEqual(parsed.preferences[0].polarity, 'LIKE')
        distinct = extract_preferences(event('En el desayuno me gusta avena. Me gusta arroz.'), 'u').preferences
        self.assertEqual([(p.food_item, p.meal_scope) for p in distinct], [('avena', 'BREAKFAST'), ('arroz', None)])

    def test_uncertainty_is_not_an_asserted_like(self):
        parsed = extract_preferences(event('En general, no sé si me gusta arroz.'), 'u')
        self.assertIn('uncertain_preference', parsed.preferences[0].scope_unknown)
        self.assertIn('unparsed_condition', parsed.preferences[0].scope_unknown)

    def test_uncertain_negated_allergy_remains_uncertain_evidence(self):
        parsed = extract_preferences(event('Quizás no soy alérgico a nueces.'), 'u')
        self.assertEqual(parsed.restrictions[0].certainty, 'UNCERTAIN')
        almonds = extract_preferences(event('Soy alérgico a las almendras.'), 'u')
        self.assertEqual(almonds.restrictions[0].food_item, 'almendras')

    def test_negated_and_unparsed_conditions_cannot_become_positive_scopes(self):
        parsed = parse_context('En general, no en casa, prefiero sopa.')
        self.assertNotIn('place', parsed.context)
        self.assertIn('negated_context', parsed.unknown_conditions)
        for phrase in ['Cuando entreno prefiero pizza.', 'Si tengo hambre prefiero pizza.', 'Excepto domingos prefiero pizza.', 'No los domingos prefiero pizza.']:
            with self.subTest(phrase=phrase):
                self.assertTrue(parse_context(phrase).unknown_conditions)
        self.assertIsNone(parse_context('No los domingos prefiero pizza.').day_scope)
        self.assertFalse(parse_context('En el almuerzo sí quiero arroz.').unknown_conditions)

    def test_legacy_all_meal_mismatches_and_third_person(self):
        for asked, requested in [('almuerzo', 'cena'), ('merienda', 'desayuno'), ('cena', 'almuerzo'), ('desayuno', 'merienda')]:
            self.assertEqual(interpret_question(f'¿Prefieres tofu en el {asked}?', f'Tofu para el {requested}', ['tofu']), 'other_scope')
        self.assertEqual(interpret_question('¿Tu hijo prefiere tofu?', 'Tofu para las comidas', ['tofu']), 'other_person')

    def test_legacy_specific_day_does_not_reveal_general_hidden_state(self):
        state = {'topic_aliases': ['tofu'], 'available_slots': {}, 'mode': 'known', 'preference': 'sin tofu', 'scope': 'mis comidas'}
        for question in ['¿Quieres tofu solo los domingos?', '¿Prefieres tofu los sábados?']:
            reply = UserSimulator(state).respond(question, '¿Qué quieres comer?')
            self.assertFalse(reply.resolved)
            self.assertNotIn('sin tofu', reply.text)
