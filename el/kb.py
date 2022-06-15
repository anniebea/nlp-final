import spacy
from spacy.kb import KnowledgeBase
from spacy.training.example import Example

import csv
from pathlib import Path

TEXTS = [
        "The Baltic Sea stretches from 53°N to 66°N latitude and from 10°E to 30°E longitude.",
        "Belgium, also known as Kingdom of Belgium, is a country in Europe"
        ]

TRAIN_DATA = [("The Baltic Sea stretches from 53°N to 66°N latitude and from 10°E to 30°E longitude.",
               {"entities": [(0, 14, "Q545")]}),
              ("It is an arm of the Atlantic Ocean.",
               {"entities": [(16, 34, "Q97")]}),
              ("The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions.",
               {"entities": [(0, 17, "Q98")]}),
              ("A marginal sea of the Atlantic, with limited water exchange between the two water bodies.",
               {"entities": [(18, 30, "Q97")]}),
              ]

nlp = spacy.load("en_core_web_lg")
vocab = nlp.vocab

# optimizer = nlp.resume_training()
# model_dir = Path("C:\\Users\\anitr\\OneDrive\\Documents\\__University\\6sem\\DAB-VAL\\NLP-final\\el\\output_model")


# List of pipes you want to train
pipe_exceptions = ['entity_linker']

# List of pipes which should remain unaffected in training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]


# Loading entities from csv file
def load_entities():
    entites_loc = Path("D:\\Uni\\6SEM\\NLP\\extracted_v2\\till_Q166717_item.csv")

    names = dict()
    descriptions = dict()
    aliases = dict()
    with entites_loc.open("r", encoding="utf8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        next(csvreader)
        for row in csvreader:
            qid = row[1]
            name = row[2]
            desc = row[3]
            aliasstring = row[4]
            aliasstring = aliasstring[:-1]
            alias_list = aliasstring.split(";")
            for itm in alias_list:
                if itm != '':
                    if aliases.get(itm) is not None:
                        # print(str(qid) + " +++ " + str(itm))
                        aliases[itm].append(qid)
                    else:
                        # print(str(qid) + " --- " + str(itm))
                        aliases[itm] = [qid]

            names[qid] = name
            descriptions[qid] = desc
    return names, descriptions, aliases


# creating knowledge base
def create_kb(vocabs):
    name_dict, desc_dict, alias_dict = load_entities()

    kb = KnowledgeBase(vocab=vocab, entity_vector_length=300)

    for qid, desc in desc_dict.items():
        desc_doc = nlp(desc)
        desc_enc = desc_doc.vector
        kb.add_entity(entity=qid, entity_vector=desc_enc, freq=342)  # frequency is arbitrary for now

    for qid, name in name_dict.items():
        if alias_dict.get(name) is not None:
            alias_dict[name].append(qid)
        else:
            alias_dict[name] = [qid]

    for alias, qids in alias_dict.items():
        count = len(qids)
        probs = list()
        for i in range(count):
            probs.append(0.0005)
        kb.add_alias(alias=alias, entities=qids, probabilities=probs)

    # print(f"Entities in the KB: {kb.get_entity_strings()}")       # will print all the QIDs
    # print(f"Aliases in the KB: {kb.get_alias_strings()}")         # will print all aliases
    # print(f"Candidates in the KB: {[c.entity_ for c in kb.get_alias_candidates('PT')]}")    # will print all candidates for "Malta"

    output_dir = Path.cwd()
    kb.to_disk(output_dir / "output_kb")
    nlp.to_disk(output_dir / "output_nlp")
    return kb


# Function to update and retrain model
def update_model():
    global nlp, linker, model_dir, TEXTS, optimizer

    # Importing requirements
    from spacy.util import minibatch, compounding
    import random

    # Begin training by disabling other pipeline components
    with nlp.disable_pipes(*other_pipes):
        sizes = compounding(1.0, 4.0, 1.001)
        # Training for 100 iterations
        for itn in range(100):
            # shuffle examples before training
            random.shuffle(TEXTS)
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TEXTS, size=sizes)
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
                nlp.update(example, sgd=optimizer, drop=0.35, losses=losses)
                print("Losses", losses)

    # Saving the model to the output directory
    nlp.to_disk(model_dir)
    print("Saved model")


if __name__ == "__main__":

    # kb = create_kb()

    testtext = "Belgium, also known as Kingdom of Belgium, is a country in Europe"
    doc = nlp(testtext)

    linker = nlp.add_pipe("entity_linker")
    linker.set_kb(create_kb)
    ds = linker.predict([doc])

    # optimizer = nlp.initialize()
    # losses = entity_linker.update(examples, sgd=optimizer)

    # other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "entity_linker"]
    # # here should be the call to update model
    # with nlp.disable_pipes(*other_pipes):
    #     optimizer = nlp.initialize()
    #     # ...
    #     nlp.update()
    #
    # doc = nlp(TEXTS[0])
    #
    # for ent in doc.ents:
    #     print(ent.text, ent.label_, ent.kb_id_)
