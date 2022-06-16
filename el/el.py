import spacy
from spacy.kb import KnowledgeBase
from spacy.util import minibatch, compounding
from spacy.pipeline import EntityRecognizer
from spacy.training.example import Example
import en_core_web_sm

from pathlib import Path
import random

from kb import create_kb

EL_TRAIN_DATA = [ ("The Baltic Sea stretches from 53°N to 66°N latitude and from 10°E to 30°E longitude.",
                   {"links": {(0, 14): {"Q545": 1.0}}}),
                  ("It is an arm of the Atlantic Ocean.",
                   {"links": {(16, 34): {"Q97": 1.0}}}),
                  ("In The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions.",
                   {"links": {(3, 20): {"Q98": 1.0}}}),
                  ("A marginal sea of the Atlantic, with limited water exchange between the two water bodies.",
                   {"links": {(18, 30): {"Q97": 1.0}}}),
                  ("The western part of Gdansk Bay is formed by the shallow waters of the Bay of Puck.",
                   {"links": {(20, 30): {"Q213367": 1.0}, (66, 81): {"Q2119446": 1.0}}}),
                  ("Gdansk Bay is known for its beaches.",
                   {"links": {(0, 10): {"Q213367": 1.0}}}),
                  ("The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions.",
                   {"links": {(0, 17): {"Q98": 1.0}}}),


                  # ("The South China Sea is a marginal sea of the Western Pacific Ocean",
                  #  {"entities": [(0, 19, "WATER"), (41, 66, "WATER")]}),
                  # ("The Gulf of Thailand and the Gulf of Tonkin are also part of the South China Sea.",
                  #  {"entities": [(0, 20, "WATER"), (25, 43, "WATER"), (61, 80, "WATER")]}),
                  # ("It communicates with the East China Sea via the Taiwan Strait.",
                  #  {"entities": [(21, 39, "WATER"), (44, 61, "WATER")]}),
                  # ("It communicates with the Java Sea via the Karimata and Bangka Strait.",
                  #  {"entities": [(21, 33, "WATER"), (38, 50, "WATER"), (55, 68, "WATER")]}),
                  # ("It communicates with the Philippine Sea via the Luzon Strait.",
                  #  {"entities": [(21, 39, "WATER"), (44, 60, "WATER")]}),
                  # ("It communicates with the Strait of Malacca via the Strait of Singapore.",
                  #  {"entities": [(21, 42, "WATER"), (47, 70, "WATER")]}),
                  # ("The controversial security deal struck between Solomon Islands and the Bank of China.",
                  #  {"entities": [(47, 62, "COUNTRY"), (79, 84, "COUNTRY")]}),
                  # ("A leading Solomon Islands official has defended his country’s right to choose its allies.",
                  #  {"entities": [(10, 25, "COUNTRY")]}),
                  # ("Beck is believed to have been involved in negotiating the deal with China.",
                  #  {"entities": [(68, 73, "COUNTRY")]}),
                  # ("Beck said that Solomon Islands faced domestic challenges.",
                  #  {"entities": [(10, 25, "COUNTRY")]}),
                  # ("Beck said that Solomon Islands faced domestic challenges.",
                  #  {"entities": [(15, 30, "COUNTRY")]}),
                  # ("Being significantly larger than other cities of Latvia, Riga is the country's primate city.",
                  #  {"entities": [(48, 54, "COUNTRY"), (56, 60, "CITY")]}),
                  # ("It is the seat of Lithuania's national government and the Vilnius District Municipality.",
                  #  {"entities": [(18, 27, "COUNTRY"), (54, 74, "AREA")]}),
                  # ("Before World War II, Vilnius was one of the largest Jewish centres in Europe. ",
                  #  {"entities": [(21, 28, "CITY"), (70, 76, "AREA")]}),
                  # ("Its Jewish influence has led to its nickname 'the Jerusalem of Lithuania'. ",
                  #  {"entities": [(50, 59, "CITY"), (63, 72, "COUNTRY")]}),
                  # ("Napoleon called it 'the Jerusalem of the North' as he was passing through in 1812.",
                  #  {"entities": [(24, 33, "CITY")]}),
                  # ("In 2009, Vilnius was the European Capital of Culture, together with Linz, Austria.",
                  #  {"entities": [(9, 16, "CITY"), (68, 72, "CITY"), (74, 81, "COUNTRY")]}),
                  # ("As many as 800 languages are spoken in New York.",
                  #  {"entities": [(39, 47, "CITY")]}),
                  # ("", {"entities": [(0, 0, "COUNTRY")]}),
               ]

# Location of saved model
model_dir = Path.cwd() / "output_nlp"


def get_kb(vocal):
    nlp = spacy.load("output_nlp")
    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=1)
    kb.from_disk("output_kb")
    return kb


def train_el():
    nlp = spacy.load("output_nlp")

    random.shuffle(EL_TRAIN_DATA)

    TRAIN_DOCS = []
    for text, annotation in EL_TRAIN_DATA:
        doc = nlp(text)
        TRAIN_DOCS.append((doc, annotation))

    entity_linker = nlp.create_pipe("entity_linker", config={"incl_prior": False})
    entity_linker.set_kb(get_kb)
    nlp.add_pipe("entity_linker")

    pipe_exceptions = ['entity_linker']
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.create_optimizer()
        sizes = compounding(1.0, 4.0, 1.001)
        # Training for 100 iterations
        for itn in range(100):
            # shuffle examples before training
            random.shuffle(EL_TRAIN_DATA)
            # batch up the examples using spaCy's minibatch
            batches = minibatch(EL_TRAIN_DATA, size=sizes)
            # dictionary to store losses
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                example = []
                # Update the model with iterating each text
                for i in range(len(texts)):
                    doc = nlp.make_doc(texts[i])
                    example.append(Example.from_dict(doc, annotations[i]))

                # Calling update() over the iteration
                nlp.update(example, drop=0.2, losses=losses, sgd=optimizer)
            if itn % 50 == 0:
                print(itn, "Losses", losses)
    print(itn, "Losses", losses)

    nlp.to_disk(Path.cwd() / "output_nlp")

def test_el():
    text = "The biggest ocean is The Pacific Ocean."
    nlp = spacy.load("output_nlp")
    doc = nlp(text)

    for ent in doc.ents:
        print(ent.text, ent.label_, ent.kb_id_)


if __name__ == "__main__":
    train_el()
    # test_el()
