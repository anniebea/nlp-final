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
              ("The western part of Gdansk Bay is formed by the shallow waters of the Bay of Puck.",
               {"entities": [(20, 30, "Q213367"), (66, 81, "Q2119446")]}),
              ("Gdansk Bay is known for its beaches.",
               {"entities": [(0, 10, "Q213367")]}),
              ("The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions.",
               {"entities": [(0, 17, "Q98")]}),
              ("The South China Sea is a marginal sea of the Western Pacific Ocean",
               {"entities": [(0, 19, "Q37660"), (41, 66, "Q98")]}),
              ("The Gulf of Thailand and the Gulf of Tonkin are also part of the South China Sea.",
               {"entities": [(0, 20, "Q131217"), (25, 43, "Q212428"), (61, 80, "Q37660")]}),
              ("It communicates with the East China Sea via the Taiwan Strait.",
               {"entities": [(21, 39, "Q45341"), (44, 61, "Q127031")]}),
              ("It communicates with the Java Sea via the Karimata and Bangka Strait.",
               {"entities": [(21, 33, "Q49364"), (38, 50, "Q24894940"), (55, 68, "Q732650")]}),
              ("The controversial security deal struck between Solomon Islands and the Bank of China.",
               {"entities": [(47, 62, "Q685"), (79, 84, "Q148")]}),
              ("A leading Solomon Islands official has defended his country’s right to choose its allies.",
               {"entities": [(10, 25, "Q685")]}),

              ("Beck is believed to have been involved in negotiating the deal with China.",
               {"entities": [(68, 73, "Q148")]}),
              ("Beck said that Solomon Islands faced domestic challenges.",
               {"entities": [(15, 30, "Q685")]}),
              ("Being significantly larger than other cities of Latvia, Riga is the country's primate city.",
               {"entities": [(48, 54, "Q211"), (56, 60, "Q1773")]}),
              ("It is also the largest city in the three Baltic states.",
               {"entities": [(41, 54, "Q39731")]}),
              ("It is home to one tenth of the three Baltic states' combined population. ",
               {"entities": [(37, 50, "Q39731")]}),
              ("The city lies on the Gulf of Riga at the mouth of the Daugava river where it meets the Baltic Sea.",
               {"entities": [(17, 33, "Q174731"), (50, 67, "Q8197"), (83, 97, "Q545")]}),

              ("Riga was founded in 1201 and is a former Hanseatic League member.",
               {"entities": [(0, 4, "Q1773")]}),
              ("Riga's historical centre is a UNESCO World Heritage Site.",
               {"entities": [(0, 4, "Q1773")]}),
              ("Riga was the European Capital of Culture in 2014, along with Umea in Sweden. ",
               {"entities": [(0, 4, "Q1773"), (61, 65, "Q25579"), (69, 75, "Q34")]}),
              ("In 2016, Riga received over 1.4 million visitors.",
               {"entities": [(9, 13, "Q1773")]}),

              ("Vilnius is the capital and largest city of Lithuania, with a population of 592,389 as of 2022.",
               {"entities": [(0, 7, "Q216"), (43, 52, "Q37")]}),
              ("It is the seat of Lithuania's national government and the Vilnius District Municipality.",
               {"entities": [(18, 27, "Q37"), (54, 74, "Q118903")]}),
              ("Before World War II, Vilnius was one of the largest Jewish centres in Europe. ",
               {"entities": [(21, 28, "Q216")]}),
              ]

nlp = spacy.load("en_core_web_lg")

# Loading entities from csv file
def load_entities():
    entites_loc = Path("D:\\Uni\\6SEM\\DAB-VAL\\NLP-final\\custom_kb.csv") # Change to appropriate path

    names = dict()
    descriptions = dict()
    aliases = dict()
    i = 0
    with entites_loc.open("r", encoding="utf8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        next(csvreader)
        for row in csvreader:
            i += 1
            print("processing row: " + str(i))
            qid = row[1]
            name = row[2]
            desc = row[3]
            aliasstring = row[4]
            aliasstring = aliasstring[:-1]
            alias_list = aliasstring.split(";")
            for itm in alias_list:
                if itm != '':
                    if aliases.get(itm) is not None:
                        aliases[itm].append(qid)
                    else:
                        aliases[itm] = [qid]

            names[qid] = name
            descriptions[qid] = desc
        print("All rows processed")
    return names, descriptions, aliases


# creating knowledge base
def create_kb(vocab=nlp.vocab):
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


if __name__ == "__main__":
    create_kb()
    print("Successfully created KB!")
