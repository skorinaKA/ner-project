import spacy

# Load the spaCy model with the span_marker pipeline component
nlp = spacy.load("en_core_web_sm", exclude=["ner"])
nlp.add_pipe("span_marker", config={"model": "tomaarsen/span-marker-roberta-large-ontonotes5"})

# Feed some text through the model to get a spacy Doc
text = """Cleopatra VII, also known as Cleopatra the Great, was the last active ruler of the \
Ptolemaic Kingdom of Egypt. She was born in 69 BCE and ruled Egypt from 51 BCE until her \
death in 30 BCE."""
doc = nlp(text)

# And look at the entities
print([(entity, entity.label_) for entity in doc.ents])
"""
[(Cleopatra VII, "PERSON"), (Cleopatra the Great, "PERSON"), (the Ptolemaic Kingdom of Egypt, "GPE"),
(69 BCE, "DATE"), (Egypt, "GPE"), (51 BCE, "DATE"), (30 BCE, "DATE")]
"""