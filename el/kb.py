import spacy
from spacy.kb import KnowledgeBase

import csv
from pathlib import Path


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


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_lg")
    text = "China has damaged the society of Tibet in such an extensive way that if Tibet ever regains its freedom, it will take many years to correct the damage inflicted by the Chinese government."
    doc = nlp(text)
    name_dict, desc_dict, alias_dict = load_entities()

    # print("Cat Bells:")
    # print(alias_dict["Cat Bells"])

    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=300)

    for qid, desc in desc_dict.items():
        desc_doc = nlp(desc)
        desc_enc = desc_doc.vector
        kb.add_entity(entity=qid, entity_vector=desc_enc, freq=342)   # frequency is arbitrary for now

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

    # qids = name_dict.keys()
    # probs = [0.00003 for qid in qids]
    # kb.add_alias(alias="aliases", entities=qids, probabilities=probs)

    # print(f"Entities in the KB: {kb.get_entity_strings()}")   # will print all the QIDs
    # print(f"Aliases in the KB: {kb.get_alias_strings()}")     # will print all aliases

    output_dir = Path.cwd()
    kb.to_disk(output_dir / "output_kb")
    nlp.to_disk(output_dir / "output_nlp")
