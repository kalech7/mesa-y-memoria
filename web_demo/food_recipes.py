"""Short preparations for the dishes offered by the live food catalog."""
import re
from unicodedata import combining, normalize


_DIRECTIONS = {
    "Tortilla de papa y cebolla": (
        "Corta la papa en láminas y la cebolla en tiras. Cocínalas con un poco de aceite hasta que "
        "la papa esté tierna. Mézclalas con huevo batido y cuaja la tortilla por ambos lados en la sartén."
    ),
    "Arroz con lentejas y zanahoria": (
        "Cuece las lentejas en agua hasta que estén tiernas. Sofríe la zanahoria picada en un poco "
        "de aceite, añade el arroz y el agua que indique su envase. Cuando el arroz esté listo, "
        "incorpora las lentejas escurridas y calienta todo junto."
    ),
    "Pescado al horno con papas": (
        "Corta las papas en láminas, úntalas con aceite y hornéalas hasta que estén casi tiernas. "
        "Añade el pescado con limón y termina la cocción hasta que la carne esté opaca y se separe "
        "con facilidad con un tenedor."
    ),
    "Ensalada de quinua con pepino y tomate": (
        "Enjuaga la quinua y cuécela en agua siguiendo su envase. Déjala enfriar. "
        "Corta el pepino y el tomate, mézclalos con la quinua y añade un poco de aceite."
    ),
    "Garbanzos salteados con pimiento": (
        "Usa garbanzos ya cocidos y escurridos. Corta el pimiento en tiras, saltéalo en aceite "
        "hasta que se ablande e incorpora los garbanzos. Remueve hasta que estén calientes."
    ),
    "Avena con banano": (
        "Cuece la avena con agua, removiendo hasta que espese. Aplasta la mitad del banano "
        "y mézclala con la avena. Sirve con el resto del banano en rodajas."
    ),
    "Huevos revueltos con tomate": (
        "Pica el tomate y sofríelo con un poco de aceite. Añade los huevos batidos "
        "y remueve a fuego suave hasta que cuajen."
    ),
    "Yogur natural con mango": (
        "Pela el mango, separa la pulpa de la semilla y córtala en cubos. "
        "Sirve el yogur natural en un cuenco y reparte el mango por encima."
    ),
    "Papaya con banano": (
        "Pela la papaya, retira las semillas y corta la pulpa en cubos. Pela y corta "
        "el banano en rodajas. Mezcla ambas frutas justo antes de servir."
    ),
    "Agua con limón": (
        "Lava el limón, córtalo y exprímelo. Mezcla el jugo con agua poco a poco "
        "hasta conseguir la intensidad que prefieras."
    ),
    "Infusión de menta": (
        "Lava las hojas de menta. Vierte agua caliente sobre ellas, tapa la taza "
        "y deja reposar unos minutos. Retira las hojas antes de servir."
    ),
    "Jugo de maracuyá": (
        "Abre el maracuyá y extrae la pulpa. Mézclala con agua en la licuadora con pulsos cortos. "
        "Pasa el jugo por un colador para retirar las semillas y sirve."
    ),
    "Café pasado": (
        "Coloca el café molido en un filtro sobre la taza. Humedécelo con un poco de agua caliente "
        "y vierte el resto despacio. Deja que termine de gotear antes de retirar el filtro."
    ),
}


def folded(text):
    return "".join(c for c in normalize("NFKD", text.casefold()) if not combining(c))


def preparation_request(text):
    phrase = folded(text)
    return bool(re.search(
        r"\b(?:como\s+(?:(?:lo|la|se)\s+)?(?:prepar[oa]|hago|hace|cocin[oa])"
        r"|(?:dame|dime|explicame|quiero|cual es|tienes)\s+(?:la|una)\s+receta"
        r"|que\s+ingredientes\s+(?:lleva|necesito)|pasos\s+para\s+preparar)\b", phrase
    ))


def preparation(dish):
    instructions = _DIRECTIONS.get(dish)
    return f"{dish}. {instructions}" if instructions else None


def recipe_dish(text, previous=None):
    """Use the suggested dish only for a reference such as '¿cómo lo preparo?'."""
    phrase = folded(text).strip(" ¿?¡!.")
    for dish in _DIRECTIONS:
        if folded(dish) in phrase:
            return dish
    if re.search(r"\b(?:receta\s+(?:de|para)|pasos\s+para\s+preparar|ingredientes\s+lleva)\s+\S", phrase):
        return None
    if re.search(r"\b(?:prepar[oa]|cocin[oa]|hago|hace)\s+(?!hoy\b|eso\b|esto\b|en casa\b|sin\b)\S", phrase):
        return None
    return previous
