import random
import json
import torch
import time
from PIL import Image
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import networkx as nx
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)

bot_name = "TanyaDok"

def get_response(msg):
    model.eval()  # Set model to evaluation mode

    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    with torch.no_grad():
        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    return random.choice(intent['responses'])
        else:
            return "Maaf, saya tidak mengerti..."

# graph = nx.DiGraph()

# graph.add_node("Start")
# graph.add_node("User input")
# graph.add_node("Model")
# graph.add_node("Probabilitas")
# graph.add_node("Rule-based")
# graph.add_node("Response")
# graph.add_node("End")

# graph.add_edge("Start", "User input")
# graph.add_edge("User input", "Model")
# graph.add_edge("Model", "Probabilitas")
# graph.add_edge("Probabilitas", "Rule-based")
# graph.add_edge("Rule-based", "Response")
# graph.add_edge("Response", "End")

# # Add edge labels
# nx.set_edge_attributes(graph, {
#     ("Start", "User input"): {"label": "Masukan pengguna"},
#     ("User input", "Model"): {"label": "Proses model"},
#     ("Model", "Probabilitas"): {"label": "Hitung probabilitas"},
#     ("Probabilitas", "Rule-based"): {"label": "Bandingkan probabilitas"},
#     ("Rule-based", "Response"): {"label": {
#         "Ya": "Tampilkan respons",
#         "Tidak": "Tidak tampilkan respons"
#     }},
#     ("Response", "End"): {"label": "Selesai"}
# })

# # Add nodes for intents
# for intent in intents['intents']:
#     graph.add_node(intent['tag'])
#     graph.add_edge("Rule-based", intent['tag'], attr={"label": "Periksa intent"})

# # Add edges for intent checks
# for intent in intents['intents']:
#     for pattern in intent['patterns']:
#         graph.add_edge(intent['tag'], pattern, attr={"label": "Match pattern"})

# # Make tree
# for node in graph.nodes:
#     if node != "Start":
#         for child in graph.neighbors(node):
#             graph.add_edge(node, child)

# # Set layout
# pos = nx.drawing.nx_agraph.graphviz_layout(graph, prog="dot")

# # Draw graph
# nx.draw(graph, pos=pos, with_labels=True, arrows=True)
# plt.show()

if __name__ == "__main__":
    print("Halo selamat datang di TanyaDok! Tanyakan apa saja (type 'quit' to exit)")
    while True:
        sentence = input('Anda: ')
        if sentence == "quit":
            break

        start_time = time.time()  # Memulai pengukuran waktu

        resp = get_response(sentence)
        print(resp)

        end_time = time.time()  # Mengakhiri pengukuran waktu
        response_time = end_time - start_time
        print(f"Response time: {response_time} seconds")
