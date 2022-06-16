import sys
from ner.named_entity_recognizer import *
from wiki.wikidata_interaction import *


def main(argv):
    print(getWikidata('Q20145'))
    print(getWikidata('Q20145').description)
    print(getUrlFromQid('Q20145'))


if __name__ == "__main__":
    main(sys.argv)

text = "Olive Morris (1952–1979) was a Jamaican-born and British-based community leader and activist. She participated in the Black nationalist, feminist and squatters' rights campaigns of the 1970s. She joined the British Black Panthers, occupied buildings in Brixton, South London, and became a key organizer in the Black Women's Movement in the United Kingdom. In London, Morris co-founded the Brixton Black Women's Group and the Organization of Women of African and Asian Descent; when she studied at the Victoria University of Manchester, she was involved in the Manchester Black Women's Co-operative and also travelled to China with the Society for Anglo-Chinese Understanding. After graduating, Morris returned to Brixton and worked at the Brixton Community Law Centre. She then received a diagnosis of non-Hodgkin lymphoma and died shortly afterwards at the age of 27. Her life and work have been commemorated by both official organizations and the activist group Remembering Olive Collective."
print('\n' + 'Combined found entities:' + '\n' + str(find_entities(text)))