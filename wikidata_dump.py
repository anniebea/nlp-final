# Get Wikidata dump records as a JSON stream (one JSON object per line)
#
# Source: https://akbaritabar.netlify.app/how_to_use_a_wikidata_dump#Sample_Python_script
#
# Command to run script (must be executed in the virtual environment):
# py wikidata_dump.py "somelocation\latest-all.json.bz2"
# On the current machine, these are the locations:
# "D:\Uni\6SEM\NLP\latest-all.json.bz2"
#
# Packages used:
# ordered-set
#
# Author's comment:
#     Modified script taken from this link: "https://www.reddit.com/r/LanguageTechnology/comments/7wc2oi/does_anyone_know_a_good_python_library_code/dtzsh2j/"


import bz2
import json
import pandas as pd
import pydash

i = 0
# an empty dataframe which will save items information
# you need to modify the columns in this data frame to save your modified data
df_record_all = pd.DataFrame(columns=[
    'id',
    'english_label',
    'english_desc',
    'english_aliases'
])


def wikidata(filename):
    with bz2.open(filename, mode='rt') as f:
        f.read(2) # skip first two bytes: "{\n"
        for line in f:
            try:
                yield json.loads(line.rstrip(',\n'))
            except json.decoder.JSONDecodeError:
                continue


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__
    )
    parser.add_argument(
        'dumpfile',
        help=(
            'a Wikidata dumpfile from: '
            'https://dumps.wikimedia.org/wikidatawiki/entities/'
            'latest-all.json.bz2'
        )
    )
    args = parser.parse_args()

    for record in wikidata(args.dumpfile):
        if pydash.has(record, 'claims.P625') and pydash.has(record, 'labels.en.value') and pydash.has(record, 'descriptions.en.value'):
            print('i = ' + str(i) + ' item ' + record['id'] + '  started!' + '\n')
            item_id = pydash.get(record, 'id')
            english_label = pydash.get(record, 'labels.en.value')
            english_desc = pydash.get(record, 'descriptions.en.value')
            english_aliases = ""
            if pydash.has(record, 'aliases.en'):
                tempset = set()
                for itm in pydash.get(record, 'aliases.en'):
                    tempset.add(itm['value'])
                for itm in tempset:
                    english_aliases += itm + ";"
            df_record = pd.DataFrame(
                {
                    'id': item_id,
                    'english_label': english_label,
                    'english_desc': english_desc,
                    'english_aliases': english_aliases
                }, index=[i])
            df_record_all = df_record_all.append(df_record, ignore_index=True)
            i += 1
            print(i)
            if (i % 5000 == 0):
                pd.DataFrame.to_csv(df_record_all,
                                    path_or_buf='D:\\Uni\\6SEM\\NLP\\extracted_v2\\till_' + record['id'] + '_item.csv')
                print('i = ' + str(i) + ' item ' + record['id'] + '  Done!')
                print('CSV exported')
                df_record_all = pd.DataFrame(columns=[
                    'id',
                    'english_label',
                    'english_desc',
                    'english_aliases'
                ])
            else:
                continue
    # pd.DataFrame.to_csv(df_record_all,
    #                     path_or_buf='D:\\Uni\\6SEM\\NLP\\extracted\\final_csv_till_' + record['id'] + '_item.csv')
    # print('i = ' + str(i) + ' item ' + record['id'] + '  Done!')
    # print('All items finished, final CSV exported!')