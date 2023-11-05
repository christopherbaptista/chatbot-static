import json
import random

def calculate_average_precision(predictions, target):
    num_correct = 0
    total_precision = 0.0

    for i, pred in enumerate(predictions, start=1):
        if pred == target:
            num_correct += 1
            precision = num_correct / i
            total_precision += precision

    if num_correct == 0:
        return 0.0

    return total_precision / num_correct

def evaluate_map(intents_file):
    with open(intents_file, 'r') as json_data:
        intents_data = json.load(json_data)

    intents = intents_data['intents']

    average_precisions = []

    for intent in intents:
        tag = intent['tag']
        patterns = intent['patterns']

        # Simulate model predictions
        predictions = [tag] * len(patterns)

        # Shuffle predictions to simulate random order
        random.shuffle(predictions)

        average_precision = calculate_average_precision(predictions, tag)
        average_precisions.append(average_precision)

    mean_average_precision = sum(average_precisions) / len(average_precisions)
    return mean_average_precision

if __name__ == '__main__':
    intents_file = 'intents.json'
    map_score = evaluate_map(intents_file)
    print(f'Mean Average Precision (MAP): {map_score:.2f}')
