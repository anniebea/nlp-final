import nltk
import spacy
from spacy import displacy
from spacy.tokens import Span
from pathlib import Path
import en_core_web_sm
import en_core_web_lg

# pip install -U spacy==2.3.7
# python -m spacy download en_core_web_sm-2.2.0 --direct

nlp = en_core_web_sm.load()

# Getting the pipeline component
ner = nlp.get_pipe('ner')

# New label to add
LABELS = ["WATER", "AREA", "COUNTRY", "CITY", "MANMADE"]

# Training examples in the required format
TRAIN_DATA =[ ("The Baltic Sea stretches from 53°N to 66°N latitude and from 10°E to 30°E longitude.", {"entities": [(0, 14, "WATER")]}),
              ("It is an arm of the Atlantic Ocean.", {"entities": [(16, 34, "WATER")]}),
              ("The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions.", {"entities": [(0, 17, "WATER")]}),
              ("A marginal sea of the Atlantic, with limited water exchange between the two water bodies.", {"entities": [(18, 30, "WATER")]}),
              ("It includes the Gulf of Bothnia, the Bay of Bothnia, the Gulf of Finland, the Gulf of Riga and the Bay of Gdansk.", {"entities": [(12, 31, "WATER"), (33, 51, "WATER"), (53, 72, "WATER"), (74, 90, "WATER"), (95, 112, "WATER")]}),
              ("The Baltic Sea is connected by artificial waterways to the White Sea via the canal and to the German Bight of the North Sea via that canal.", {"entities": [(0, 14, "WATER"), (55, 68, "WATER"), (90, 106, "WATER"), (110, 123, "WATER")]}),
              ("The western part of Gdansk Bay is formed by the shallow waters of the Bay of Puck.", {"entities": [(20, 30, "WATER"), (66, 81, "WATER")]}),
              ("Gdansk Bay is known for its beaches.", {"entities": [(0, 10, "WATER")]}),
              ("The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions.", {"entities": [(0, 17, "WATER")]}),
              ("The South China Sea is a marginal sea of the Western Pacific Ocean", {"entities": [(0, 19, "WATER"), (41, 66, "WATER")]}),
              ("The Gulf of Thailand and the Gulf of Tonkin are also part of the South China Sea.", {"entities": [(0, 20, "WATER"), (25, 43, "WATER"), (61, 80, "WATER")]}),
              ("It communicates with the East China Sea via the Taiwan Strait.", {"entities": [(21, 39, "WATER"), (44, 61, "WATER")]}),
              ("It communicates with the Java Sea via the Karimata and Bangka Strait.", {"entities": [(21, 33, "WATER"), (38, 50, "WATER"), (55, 68, "WATER")]}),
              ("It communicates with the Philippine Sea via the Luzon Strait.", {"entities": [(21, 39, "WATER"), (44, 60, "WATER")]}),
              ("It communicates with the Sulu Sea via the straits around Palawan (e.g. the Mindoro and Balabac Strait).", {"entities": [(21, 34, "WATER"), (57, 64, "AREA"), (71, 82, "WATER"), (87, 101, "WATER")]}),
              ("It communicates with the Strait of Malacca via the Strait of Singapore.", {"entities": [(21, 42, "WATER"), (47, 70, "WATER")]}),
              ("The controversial security deal struck between Solomon Islands and the Bank of China.", {"entities": [(47, 62, "COUNTRY"), (79, 84, "COUNTRY")]}),
              ("A leading Solomon Islands official has defended his country’s right to choose its allies.", {"entities": [(10, 25, "COUNTRY")]}),
              ("He is speaking to the Guardian in his first interview since the deal between China and Solomon Islands was leaked.", {"entities": [(77, 82, "COUNTRY"), (87, 102, "COUNTRY")]}),
              ("Australia should question whether it had been fair to Solomon Islands in its intense scrutiny of the deal.", {"entities": [(0, 9, "COUNTRY"), (54, 69, "COUNTRY")]}),
              ("Beck is believed to have been involved in negotiating the deal with China.", {"entities": [(68, 73, "COUNTRY")]}),
              ("Beck said that Solomon Islands faced domestic challenges.", {"entities": [(10, 25, "COUNTRY")]}),
              ("Chronic unemployment, as well as frustrations with the policies and leadership of the prime minister, Manasseh Sogavare, were thought to be behind riots in Honiara last year that left three people dead.", {"entities": [(156, 163, "CITY")]}),
              ("Riga is the capital of Latvia and is home to 605802 inhabitants (2022), which is a third of Latvia's population.", {"entities": [(0, 4, "CITY"), (23, 29, "COUNTRY"), (92, 98, "COUNTRY")]}),
              ("Being significantly larger than other cities of Latvia, Riga is the country's primate city.", {"entities": [(48, 54, "COUNTRY"), (56, 60, "CITY")]}),
              ("It is also the largest city in the three Baltic states.", {"entities": [(41, 54, "AREA")]}),
              ("It is home to one tenth of the three Baltic states' combined population. ", {"entities": [(37, 50, "AREA")]}),
              ("The city lies on the Gulf of Riga at the mouth of the Daugava river where it meets the Baltic Sea.", {"entities": [(17, 33, "WATER"), (50, 67, "WATER"), (83, 97, "WATER")]}),
              ("Riga's territory covers 307.17 km2 (118.60 sq mi) and lies 1–10 m (3.3–32.8 ft) above sea level, on a flat and sandy plain.", {"entities": [(0, 4, "CITY")]}),
              ("Riga was founded in 1201 and is a former Hanseatic League member.", {"entities": [(0, 4, "CITY")]}),
              ("Riga's historical centre is a UNESCO World Heritage Site.", {"entities": [(0, 4, "CITY")]}),
              ("Riga was the European Capital of Culture in 2014, along with Umea in Sweden. ", {"entities": [(0, 4, "CITY"), (61, 65, "CITY"), (69, 75, "COUNTRY")]}),
              ("Riga hosted the 2006 NATO Summit, the Eurovision Song Contest 2003, the 2006 IIHF Men's World Ice Hockey Championships, 2013 World Women's Curling Championship and the 2021 IIHF World Championship.", {"entities": [(0, 4, "CITY")]}),
              ("It is home to the European Union's office of European Regulators for Electronic Communications (BEREC).", {"entities": [(18, 32, "AREA")]}),
              ("In 2016, Riga received over 1.4 million visitors.", {"entities": [(9, 13, "CITY")]}),
              ("The city is served by Riga International Airport, the largest and busiest airport in the Baltic states.", {"entities": [(22, 48, "MANMADE"), (89, 102, "AREA")]}),
              ("Riga is a member of Eurocities, the Union of the Baltic Cities (UBC) and Union of Capitals of the European Union (UCEU).", {"entities": [(0, 4, "CITY"), (98, 112, "AREA")]}),
              ("Tallinn is the most populous, primate, and capital city of Estonia.", {"entities": [(0, 7, "CITY"), (59, 66, "COUNTRY")]}),
              ("It is situated on a bay in north Estonia, on the shore of the Gulf of Finland of the Baltic Sea.", {"entities": [(33, 40, "COUNTRY"), (58, 77, "WATER"), (81, 95, "WATER")]}),
              ("Tallinn has a population of 437811 (as of 2022) and administratively lies in the Harju maakond (county).", {"entities": [(0, 7, "CITY"), (77, 103, "AREA")]}),
              ("Tallinn is the main financial, industrial, and cultural centre of Estonia.", {"entities": [(0, 7, "CITY"), (66, 73, "COUNTRY")]}),
              ("It is located 187 km (116 mi) northwest of the country's second largest city Tartu, however only 80 km (50 mi) south of Helsinki, Finland.", {"entities": [(77, 82, "CITY"), (120, 128, "CITY"), (130, 137, "COUNTRY")]}),
              ("The city is located 320 km (200 mi) west of Saint Petersburg, Russia, 300 km (190 mi) north of Riga, Latvia, and 380 km (240 mi) east of Stockholm, Sweden.", {"entities": [(44, 60, "CITY"), (62, 68, "COUNTRY"), (95, 99, "CITY"), (101, 107, "COUNTRY"), (137, 146, "CITY"), (148, 154, "COUNTRY")]}),
              ("From the 13th century until the first half of the 20th century, Tallinn was known in most of the world by variants of its other historical name Reval.", {"entities": [(64, 71, "CITY"), (144, 149, "CITY")]}),
              ("Tallinn received city rights in 1248, however the earliest evidence of human population in the area dates back nearly 5,000 years.", {"entities": [(0, 7, "CITY")]}),
              ("The first recorded claim over the place was laid by Denmark after a successful raid in 1219 led by King Valdemar II.", {"entities": [(52, 59, "COUNTRY")]}),
              ("In the 14-16th centuries Tallinn grew in importance as the northernmost member city of the Hanseatic League.", {"entities": [(25, 32, "CITY")]}),
              ("Tallinn Old Town is one of the best-preserved medieval cities in Europe and is listed as a UNESCO World Heritage Site.", {"entities": [(0, 16, "MANMADE"), (65, 71, "AREA")]}),
              ("Tallinn has the highest number of start-ups per person among European countries and is the birthplace of many international high-technology companies, including Skype, Bolt and Wise.", {"entities": [(0, 7, "CITY")]}),
              ("The city is home to the headquarters of the European Union's IT agency, and to the NATO Cyber Defence Centre of Excellence.", {"entities": [(44, 58, "AREA")]}),
              ("In 2007, Tallinn was listed among the top-10 digital cities in the world.", {"entities": [(9, 16, "CITY")]}),
              ("Tokyo (formerly Edo, historically Tokio, and officially the Tokyo Metropolis) is the capital and largest city of Japan.", {"entities": [(0, 5, "CITY"), (16, 19, "CITY"), (34, 39, "CITY"), (56, 76, "CITY"), (113, 118, "COUNTRY")]}),
              ("It is located at the head of Tokyo Bay.", {"entities": [(29, 38, "WATER")]}),
              ("The prefecture forms part of the Kanto region on the central Pacific coast of Japan's main island of Honshu.", {"entities": [(29, 45, "AREA"), (78, 83, "COUNTRY"), (101, 107, "AREA")]}),
              ("Tokyo is the political and economic center of the country.", {"entities": [(0, 5, "CITY")]}),
              ("It is the seat of the Emperor of Japan and the national government.", {"entities": [(33, 38, "COUNTRY")]}),
              ("Originally the city was a fishing village, named Edo.", {"entities": [(49, 52, "CITY")]}),
              ("The city became a prominent political center in 1603, when it became the seat of the Tokugawa shogunate.", {"entities": [(85, 103, "COUNTRY")]}),
              ("By the mid-18th century, Edo was one of the most populous cities in the world at over one million.", {"entities": [(25, 28, "COUNTRY")]}),
              ("Following the end of the shogunate in 1868, the imperial capital in Kyoto was moved to the city, which was renamed Tokyo (literally 'eastern capital').", {"entities": [(68, 73, "CITY"), (115, 120, "CITY")]}),
              ("Tokyo was devastated by the 1923 Great Kanto earthquake, and again by Allied bombing raids during World War II.", {"entities": [(0, 5, "CITY"), (39, 44, "AREA")]}),
              ("Beginning in the 1950s, the city underwent rapid reconstruction and expansion, going on to lead Japan's post-war economic recovery.", {"entities": [(96, 101, "COUNTRY")]}),
              ("Tokyo is the largest urban economy in the world by gross domestic product.", {"entities": [(0, 5, "CITY")]}),
              ("It is part of an industrial region that includes the cities of Yokohama, Kawasaki, and Chiba. ", {"entities": [(63, 71, "CITY"), (73, 81, "CITY"), (87, 92, "CITY")]}),
              ("Tokyo is Japan's leading center of business and finance.", {"entities": [(0, 5, "CITY"), (9, 14, "COUNTRY")]}),
              ("In 2020, it ranked fourth on the Global Financial Centres Index, behind New York City, London, and Shanghai. ", {"entities": [(72, 85, "CITY"), (87, 93, "CITY"), (99, 107, "CITY")]}),
              ("The Tokyo Metro Ginza Line is the oldest underground metro line in East Asia (1927).", {"entities": [(4, 9, "CITY"), (67, 76, "AREA")]}),
              ("Japan's Shinkansen bullet train system.", {"entities": [(0, 5, "COUNTRY")]}),
              ("Notable districts of Tokyo include Chiyoda (the site of the National Diet Building and the Imperial Palace), Shinjuku (the city's administrative center), and Shibuya (a commercial, cultural and business hub).", {"entities": [(21, 26, "CITY"), (35, 42, "AREA"), (60, 82, "MANMADE"), (87, 106, "MANMADE"), (109, 117, "AREA"), (158, 165, "AREA")]}),
              ("Palawan officially the Province of Palawan, is an archipelagic province of the Philippines that is located in the region of Mimaropa.", {"entities": [(0, 7, "AREA"), (19, 42, "AREA"), (75, 90, "COUNTRY"), (124, 132, "AREA")]}),
              ("The capital city is Puerto Princesa.", {"entities": [(20, 35, "CITY")]}),
              ("Palawan is known as the Philippines' Last Frontier and as the Philippines' Best Island.", {"entities": [(0, 7, "AREA"), (20, 35, "COUNTRY"), (58, 73, "COUNTRY")]}),
              ("The islands of Palawan stretch between Mindoro island in the northeast and Borneo in the southwest.", {"entities": [(15, 22, "AREA"), (39, 53, "AREA"), (75, 81, "AREA")]}),
              ("It lies between the South China Sea and the Sulu Sea.", {"entities": [(16, 35, "WATER"), (40, 52, "WATER")]}),
              ("The province is named after its largest island Palawan Island. ", {"entities": [(47, 61, "AREA")]}),
              ("In 2019, it was proposed to divide Palawan into three separate provinces, though it was rejected by the local population in a 2021 plebiscite.", {"entities": [(35, 42, "AREA")]}),
              ("Mimaropa (usually capitalized in official government documents), formally known as the Southwestern Tagalog Region, is an administrative region in the Philippines.", {"entities": [(0, 8, "AREA"), (83, 114, "AREA"), (147, 162, "COUNTRY")]}),
              ("It is one of two regions in the country having no land border with another region (the other being Eastern Visayas).", {"entities": [(99, 114, "AREA")]}),
              ("The name is an acronym combination of its constituent provinces: Mindoro (divided into Occidental Mindoro and Oriental Mindoro), Marinduque, Romblon and Palawan.", {"entities": [(65, 72, "AREA"), (87, 105, "AREA"), (110, 126, "AREA"), (129, 139, "AREA"), (141, 148, "AREA"), (153, 160, "AREA")]}),
              ("The region was part of the now-defunct Southern Tagalog region until May 17, 2002.", {"entities": [(39, 62, "AREA")]}),
              ("On May 23, 2005, Palawan and the highly urbanized city of Puerto Princesa were moved to the region of Western Visayas by Executive Order No. 429.", {"entities": [(17, 24, "AREA"), (58, 73, "CITY"), (102, 117, "AREA")]}),
              ("On July 17, 2016, Republic Act No. 10879 formally established the Southwestern Tagalog Region to be known as Mimaropa discontinuing the 'Region IV-B' designation, however no boundary changes were involved.", {"entities": [(62, 93, "AREA"), (109, 117, "AREA"), (137, 148, "AREA")]}),
              ("Calapan is Mimaropa's regional center.", {"entities": [(0, 7, "CITY"), (11, 19, "AREA")]}),
              ("However, most regional government offices such as the Department of Public Works and Highways and the Department of Budget and Management are in Quezon City, Metro Manila.", {"entities": [(145, 156, "CITY"), (158, 170, "AREA")]}),
              ("However, on August 19, 2005, then-President Arroyo issued Administrative Order No. 129 to put in abeyance Executive Order No. 429 pending a review.", {"entities": []}),
              #("", {"entities": [(0, 0, "COUNTRY")]}),
              #("", {"entities": [(0, 0, "COUNTRY")]}),
              #("", {"entities": [(0, 0, "COUNTRY")]}),
              #("", {"entities": [(0, 0, "COUNTRY")]}),
              #("", {"entities": [(0, 0, "COUNTRY")]}),
           ]

# Adding labels to the `ner`
for LABEL in LABELS:
    ner.add_label(LABEL)

# Resume training
optimizer = nlp.resume_training()
move_names = list(ner.move_names)

# List of pipes you want to train
pipe_exceptions = ['ner']

# List of pipes which should remain unaffected in training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

# Location of saved model
model_dir = Path(__file__).parent
should_update_model = 0

if (should_update_model == 1):
    # Importing requirements
    from spacy.util import minibatch, compounding
    import random

    # Begin training by disabling other pipeline components
    with nlp.disable_pipes(*other_pipes):
      sizes = compounding(1.0, 4.0, 1.001)
      # Training for 500 iterations     
      for itn in range(500):
        # shuffle examples before training
        random.shuffle(TRAIN_DATA)
        # batch up the examples using spaCy's minibatch
        batches = minibatch(TRAIN_DATA, size=sizes)
        # dictionary to store losses
        losses = {}
        for batch in batches:
          texts, annotations = zip(*batch)
          # Calling update() over the iteration
          nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
          print("Losses", losses)

    # Saving the model to the output directory
    nlp.to_disk(model_dir)
    print("Saved model")

# Loading the model from the directory
print("Loading created model...")
nlp = spacy.load(model_dir)
assert nlp.get_pipe("ner").move_names == move_names

#text = input("Input text: ")
text = "Mimaropa (usually capitalized in official government documents), formally known as the Southwestern Tagalog Region, is an administrative region in the Philippines. It was also formerly designated as Region IV-B until 2016. It is one of two regions in the country having no land border with another region (the other being Eastern Visayas). The name is an acronym combination of its constituent provinces: Mindoro (divided into Occidental Mindoro and Oriental Mindoro), Marinduque, Romblon and Palawan. The region was part of the now-defunct Southern Tagalog region until May 17, 2002. On May 23, 2005, Palawan and the highly urbanized city of Puerto Princesa were moved to the region of Western Visayas by Executive Order No. 429. However, on August 19, 2005, then-President Arroyo issued Administrative Order No. 129 to put in abeyance Executive Order No. 429 pending a review. On July 17, 2016, Republic Act No. 10879 formally established the Southwestern Tagalog Region to be known as Mimaropa discontinuing the 'Region IV-B' designation, however no boundary changes were involved. Calapan is Mimaropa's regional center. However, most regional government offices such as the Department of Public Works and Highways and the Department of Budget and Management are in Quezon City, Metro Manila. "

article = nlp(text)
found_entities = dict([(str(x), x.label_) for x in nlp(str(article)).ents])
print('\n' + 'Found entities:' + '\n' + str(found_entities))

colors = {"WATER": "#6281cc", "AREA": "#60ad47", "COUNTRY": "#d95b5b", "CITY": "#de8d5b", "MANMADE": "#96e3e3"}
options = {"colors": colors}

print('\n' + "Loading existing model...")
nlp = en_core_web_lg.load()
original_entities = dict([(str(x), 'Uncategorized') for x in nlp(str(article)).ents if x.label_ == 'GPE' or x.label_ == 'LOC'])
print('\n' + 'Found entities:' + '\n' + str(original_entities))
#article = nlp(text)
#options = {"ents": ["GPE", "LOC"]}

print('\n' + "Combining results...")
combined_entities = original_entities
combined_entities.update(found_entities)
print('\n' + 'Combined found entities:' + '\n' + str(combined_entities))

displacy.serve(article, style="ent", options=options) #http://localhost:5000
