from span_marker import SpanMarkerModel

# Download from the 🤗 Hub
model = SpanMarkerModel.from_pretrained("C:\\Users\\skorina.ka\\Desktop\\MachineLearning\\ner-project\\models\\tomaarsen\\span-marker-bert-base-cased-fewnerd-fine-super\\checkpoint-final")
# Run inference
entities = model.predict("Amelia Earhart flew her single engine Lockheed Vega 5B across the Atlantic to Paris.")
print(entities)
# [{'span': 'Amelia Earhart', 'label': 'person-other', 'score': 0.7659597396850586, 'char_start_index': 0, 'char_end_index': 14},
#  {'span': 'Lockheed Vega 5B', 'label': 'product-airplane', 'score': 0.9725785851478577, 'char_start_index': 38, 'char_end_index': 54},
#  {'span': 'Atlantic', 'label': 'location-bodiesofwater', 'score': 0.7587679028511047, 'char_start_index': 66, 'char_end_index': 74},
#  {'span': 'Paris', 'label': 'location-GPE', 'score': 0.9892390966415405, 'char_start_index': 78, 'char_end_index': 83}]