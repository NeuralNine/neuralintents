import itertools


def generate_messages_from_templates(intent_data: dict) -> tuple[list[str], list[tuple[int, int, str]]]:
    templates = intent_data['patterns']
    placeholders = intent_data['placeholders']

    placeholder_values = {placeholder: intent_data[category] for placeholder, category in placeholders.items()}

    combinations = itertools.product(*placeholder_values.values())

    sentences = []
    entities = []

    for template in templates:
        for combination in combinations:
            sentence = template
            entity_list = []
            for placeholder, value in zip(placeholder_values.keys(), combination):
                label = placeholders[placeholder]
                placeholder_position = sentence.index(placeholder)
                sentence = sentence[:placeholder_position] + value + sentence[placeholder_position+1:]
                entity_list.append((placeholder_position, placeholder_position+len(value), label))
            sentences.append(sentence)
            entities.append(entity_list)

    return sentences, entities

