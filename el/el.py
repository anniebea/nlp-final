import spacy
from spacy.kb import KnowledgeBase
from spacy.pipeline import EntityLinker
from spacy.util import minibatch, compounding
from spacy.training.example import Example
import en_core_web_sm
from spacy import displacy

from pathlib import Path
import random

from kb import create_kb

EL_TRAIN_DATA = [ ("The Baltic Sea stretches from 53°N to 66°N latitude and from 10°E to 30°E longitude.", {"links": {(0, 14): {"Q545": 1.0}}}),
                  ("It is an arm of the Atlantic Ocean.", {"links": {(16, 34): {"Q97": 1.0}}}),
                  ("The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions.", {"links": {(0, 17): {"Q98": 1.0}}}),
                  ("A marginal sea of the Atlantic, with limited water exchange between the two water bodies.", {"links": {(18, 30): {"Q97": 1.0}}}),
                  ("The western part of Gdansk Bay is formed by the shallow waters of the Bay of Puck.", {"links": {(20, 30): {"Q213367": 1.0}, (66, 81): {"Q2119446": 1.0}}}),
                  ("Gdansk Bay is known for its beaches.", {"links": {(0, 10): {"Q213367": 1.0}}}),
                  ("The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions.", {"links": {(0, 17): {"Q98": 1.0}}}),
                  ("The South China Sea is a marginal sea of the Western Pacific Ocean", {"links": {(0, 19): {"Q37660": 1.0}, (41, 66): {"Q98": 1.0}}}),
                  ("The Gulf of Thailand and the Gulf of Tonkin are also part of the South China Sea.", {"links": {(0, 20): {"Q131217": 1.0}, (25, 43): {"Q212428": 1.0}, (61, 80): {"Q37660": 1.0}}}),
                  ("It communicates with the East China Sea via the Taiwan Strait.", {"links": {(21, 39): {"Q45341": 1.0}, (44, 61): {"Q98": 1.0}}}),
                  ("It communicates with the Java Sea via the Karimata and Bangka Strait.", {"links": {(21, 33): {"Q49364": 1.0}, (38, 50): {"Q24894940": 1.0}, (55, 68): {"Q732650": 1.0}}}),
                  ("It communicates with the Philippine Sea via the Luzon Strait.", {"links": {(21, 39): {"Q159183": 1.0}, (44, 60): {"Q908741": 1.0}}}),
                  ("It communicates with the Strait of Malacca via the Straits of Singapore.", {"links": {(21, 42): {"Q48359": 1.0}, (47, 71): {"Q205655": 1.0}}}),
                  ("The controversial security deal struck between Solomon Islands and the Bank of China.", {"links": {(47, 62): {"Q685": 1.0}, (79, 84): {"Q148": 1.0}}}),
                  ("A leading Solomon Islands official has defended his country’s right to choose its allies.", {"links": {(10, 25): {"Q685": 1.0}}}),

                  ("Riga was founded in 1201 and is a former Hanseatic League member.",
                    {"links": {(0, 4): {"Q1773": 1.0}}}),
                  ("Riga's historical centre is a UNESCO World Heritage Site.",
                    {"links": {(0, 4): {"Q1773": 1.0}}}),
                  ("Riga was the European Capital of Culture in 2014, along with Umea in Sweden. ",
                    {"links": {(0, 4): {"Q1773": 1.0}, (61, 65): {"Q25579": 1.0}, (69, 75): {"Q34": 1.0}}}),
                  ("In 2016, Riga received over 1.4 million visitors.",
                    {"links": {(9, 13): {"Q1773": 1.0}}}),
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

    nlp.add_pipe("sentencizer")
    entity_linker = nlp.add_pipe("entity_linker", config={"incl_prior": False, "entity_vector_length": 300}, last=True)
    entity_linker.set_kb(create_kb)

    pipe_exceptions = ['entity_linker']
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.create_optimizer()
        # optimizer = nlp.resume_training()
        sizes = compounding(4.0, 10.0, 1.001)
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
                    # print(texts[i])
                    example.append(Example.from_dict(doc, annotations[i]))

                # Calling update() over the iteration
                nlp.update(example, drop=0.2, losses=losses, sgd=optimizer)
            if itn % 50 == 0:
                print(itn, "Losses", losses)
    print(itn, "Losses", losses)

    nlp.to_disk(Path.cwd() / "output_el")


def get_examples(nlp):
    examples = []
    for i in range(len(EL_TRAIN_DATA)):
        doc = nlp.make_doc(EL_TRAIN_DATA[i][0])
        examples.append(Example.from_dict(doc, EL_TRAIN_DATA[i][1]))
    return examples


def test_el():
    nlp = spacy.load(Path.cwd() / "output_el")
    nlp.resume_training()
    # analysis = nlp.analyze_pipes(pretty=True)
    # print(analysis)

    article = "It is located near the Baltic Sea and the Pacific Ocean and Riga."
    doc = nlp.make_doc(article)

    print(doc.ents)
    for ent in doc.ents:
        print(ent.text, ent.label_, ent.kb_id_)

    colors = {"Q545": "#6281cc", "Q97": "#60ad47", "Q98": "#d95b5b", "CITY": "#de8d5b", "MANMADE": "#96e3e3"}
    options = {"colors": colors}
    # displacy.serve(doc, style="ent", options=options) #http://localhost:5000  - this is what makes the visual

if __name__ == "__main__":
    train_el()
    test_el()
