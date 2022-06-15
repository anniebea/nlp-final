"""Wikidata interaction function definitions

Wikidata package project can be found here: https://pypi.org/project/Wikidata/

Sources:
[1] https://stackoverflow.com/questions/37079989/how-to-get-wikipedia-page-from-wikidata-id
[2] https://stackoverflow.com/questions/70229075/extract-aliases-from-wikidata-dump-using-python
"""
import requests
from wikidata.client import Client


def getWikidata(qid):
    client = Client()
    entity = client.get(qid, load=True)
    return entity


def getUrlFromQid(qid, lang='en', debug=False):  # source: [1]
    url = (
        'https://www.wikidata.org/w/api.php'
        '?action=wbgetentities'
        '&props=sitelinks/urls'
        f'&ids={qid}'
        '&format=json')
    json_response = requests.get(url).json()
    if debug:
        print(qid, url, json_response)

    entities = json_response.get('entities')
    if entities:
        entity = entities.get(qid)
        if entity:
            sitelinks = entity.get('sitelinks')
            if sitelinks:
                if lang:   # filter only the specified language
                    sitelink = sitelinks.get(f'{lang}wiki')
                    if sitelink:
                        wiki_url = sitelink.get('url')
                        if wiki_url:
                            return wiki_url
                else:   # future-proofing for multiple language support
                    wiki_urls = {}
                    for key, sitelink in sitelinks.items():
                        wiki_url = sitelink.get('url')
                        if wiki_url:
                            wiki_urls[key] = wiki_url
                    return wiki_urls
    return None
