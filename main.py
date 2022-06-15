import sys
from ner.named_entity_recognizer import *
from wiki.wikidata_interaction import *


def main(argv):
    print(getWikidata('Q20145'))
    print(getWikidata('Q20145').description)
    print(getUrlFromQid('Q20145'))


if __name__ == "__main__":
    main(sys.argv)

text = "The well-known country of New Zealand is a small, resourceful nation located 1,000 miles off Australia's south east coast.  New Zealand has an impressive economy that continues to grow, a physical landscape that attracts people from around the globe, and although small, New Zealand is a respected nation for its advanced civilization and stable government.  The geography of this prestigious nation can be described through five principal categories, the physical geography,  the cultural geography, the citizens' standard of living, the government, and the nation's economy. New Zealand is located in the southern hemisphere, with an absolute location of 37 degrees south longitude to 48 degrees south longitude and 167 degrees east latitude to 177 degrees east latitude.  It is composed of two major islands named the North and South Islands, and the total land area of the nation, approximately divided equally between the two islands, is 103,470 square miles.  Surprisingly, only 2 percent of the land area is arable.  New Zealand has an abundance of natural resources, explaining why the country is so wealthy compared to other nations.  These resources include fertile grazing land, oil and gas, iron, coal, timber, and excellent fishing waters."
print('\n' + 'Combined found entities:' + '\n' + str(find_entities(text)))