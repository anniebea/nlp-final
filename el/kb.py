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
        for row in csvreader:
            qid = row[1]
            name = row[2]
            desc = row[3]
            aliasstring = row[4]
            aliasstring = aliasstring[:-1]
            alias = aliasstring.split(";")
            names[qid] = name
            descriptions[qid] = desc
            aliases[qid] = alias
    return names, descriptions, aliases


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_lg")
    text = "China has damaged the society of Tibet in such an extensive way that if Tibet ever regains its freedom, it will take many years to correct the damage inflicted by the Chinese government."
    doc = nlp(text)
    name_dict, desc_dict, alias_dict = load_entities()

    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=300)

    for qid, desc in desc_dict.items():
        desc_doc = nlp(desc)
        desc_enc = desc_doc.vector
        kb.add_entity(entity=qid, entity_vector=desc_enc, freq=342)   # frequency is arbitrary for now

    for qid, name in name_dict.items():
        if name != '':
            kb.add_alias(alias=name, entities=[qid], probabilities=[1])

    for qid, aliases in alias_dict.items():
        for alias in aliases:
            if alias != '':
                kb.add_alias(alias=alias, entities=[qid], probabilities=[0.3])


    qids = name_dict.keys()
    probs = [0.00003 for qid in qids]
    kb.add_alias(alias="aliases", entities=qids, probabilities=probs)

    # print(f"Entities in the KB: {kb.get_entity_strings()}")   # will print all the QIDs
    # print(f"Aliases in the KB: {kb.get_alias_strings()}")     # will print all aliases
    print(f"Candidates for 'Alps': {[c.entity_ for c in kb.get_alias_candidates('Alps')]}")
    # print(f"Candidates for 'Annie': {[c.entity_ for c in kb.get_alias_candidates('Annie')]}")

    # output_dir = Path.cwd()
    # kb.dump(output_dir / "my_kb")
    # nlp.to_disk(output_dir / "my_nlp")
