import sys
from wiki.wikidata_interaction import *


def main(argv):
    print(getWikidata('Q20145'))
    print(getWikidata('Q20145').description)
    print(getUrlFromQid('Q20145'))


if __name__ == "__main__":
    main(sys.argv)
