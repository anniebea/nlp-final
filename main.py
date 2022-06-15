import sys
from ner.named_entity_recognizer import *
from wiki.wikidata_interaction import *


def main(argv):
    print(getWikidata('Q20145'))
    print(getWikidata('Q20145').description)
    print(getUrlFromQid('Q20145'))


if __name__ == "__main__":
    main(sys.argv)

text = "Vilnius is the capital and largest city of Lithuania, with a population of 592,389 as of 2022. The population of Vilnius's functional urban area, which stretches beyond the city limits, is estimated at 718,507 (as of 2020), while according to the Vilnius territorial health insurance fund, there were 732,421 permanent inhabitants as of October 2020 in Vilnius city and Vilnius district municipalities combined. Vilnius is situated in southeastern Lithuania and is the second-largest city in the Baltic states, but according to the Bank of Latvia is expected to become the largest in 2025. It is the seat of Lithuania's national government and the Vilnius District Municipality. Vilnius is classified as a Gamma global city according to GaWC studies, and is known for the architecture in its Old Town, declared a UNESCO World Heritage Site in 1994. Before World War II, Vilnius was one of the largest Jewish centres in Europe. Its Jewish influence has led to its nickname 'the Jerusalem of Lithuania'. Napoleon called it 'the Jerusalem of the North' as he was passing through in 1812. In 2009, Vilnius was the European Capital of Culture, together with Linz, Austria. In 2021, Vilnius was named among top-25 fDi's Global Cities of the Future – one of the most forward-thinking cities with the greatest potential in the World."
print('\n' + 'Combined found entities:' + '\n' + str(find_entities(text)))