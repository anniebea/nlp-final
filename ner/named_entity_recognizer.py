import nltk
import spacy
from spacy import displacy
from spacy.tokens import Span
from spacy.training.example import Example
from pathlib import Path
import en_core_web_sm
import en_core_web_lg

# pip install -U spacy==3.3.1
# python -m spacy download en_core_web_sm-3.3.0 --direct
# python -m spacy download en_core_web_lg-3.3.0 --direct

nlp = en_core_web_sm.load()

# Getting the pipeline component
ner = nlp.get_pipe('ner')

# Location of saved model
model_dir = Path(__file__).parent

# New label to add
LABELS = ["WATER", "AREA", "COUNTRY", "CITY", "MANMADE"]

# Training examples in the required format
TRAIN_DATA = [("The Baltic Sea stretches from 53°N to 66°N latitude and from 10°E to 30°E longitude.",
               {"entities": [(0, 14, "WATER")]}),
              ("It is an arm of the Atlantic Ocean.", {"entities": [(16, 34, "WATER")]}),
              ("The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions.",
               {"entities": [(0, 17, "WATER")]}),
              ("A marginal sea of the Atlantic, with limited water exchange between the two water bodies.",
               {"entities": [(18, 30, "WATER")]}),
              (
              "It includes the Gulf of Bothnia, the Bay of Bothnia, the Gulf of Finland, the Gulf of Riga and the Bay of Gdansk.",
              {"entities": [(12, 31, "WATER"), (33, 51, "WATER"), (53, 72, "WATER"), (74, 90, "WATER"),
                            (95, 112, "WATER")]}),
              (
              "The Baltic Sea is connected by artificial waterways to the White Sea via the canal and to the German Bight of the North Sea via that canal.",
              {"entities": [(0, 14, "WATER"), (55, 68, "WATER"), (90, 106, "WATER"), (110, 123, "WATER")]}),
              ("The western part of Gdansk Bay is formed by the shallow waters of the Bay of Puck.",
               {"entities": [(20, 30, "WATER"), (66, 81, "WATER")]}),
              ("Gdansk Bay is known for its beaches.", {"entities": [(0, 10, "WATER")]}),
              ("The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions.",
               {"entities": [(0, 17, "WATER")]}),
              ("The South China Sea is a marginal sea of the Western Pacific Ocean",
               {"entities": [(0, 19, "WATER"), (41, 66, "WATER")]}),
              ("The Gulf of Thailand and the Gulf of Tonkin are also part of the South China Sea.",
               {"entities": [(0, 20, "WATER"), (25, 43, "WATER"), (61, 80, "WATER")]}),
              ("It communicates with the East China Sea via the Taiwan Strait.",
               {"entities": [(21, 39, "WATER"), (44, 61, "WATER")]}),
              ("It communicates with the Java Sea via the Karimata and Bangka Strait.",
               {"entities": [(21, 33, "WATER"), (38, 50, "WATER"), (55, 68, "WATER")]}),
              ("It communicates with the Philippine Sea via the Luzon Strait.",
               {"entities": [(21, 39, "WATER"), (44, 60, "WATER")]}),
              (
              "It communicates with the Sulu Sea via the straits around Palawan (e.g. the Mindoro and Balabac Strait).",
              {"entities": [(21, 33, "WATER"), (57, 64, "AREA"), (71, 82, "WATER"), (87, 101, "WATER")]}),
              ("It communicates with the Strait of Malacca via the Strait of Singapore.",
               {"entities": [(21, 42, "WATER"), (47, 70, "WATER")]}),
              ("The controversial security deal struck between Solomon Islands and the Bank of China.",
               {"entities": [(47, 62, "COUNTRY"), (79, 84, "COUNTRY")]}),
              ("A leading Solomon Islands official has defended his country’s right to choose its allies.",
               {"entities": [(10, 25, "COUNTRY")]}),
              (
              "He is speaking to the Guardian in his first interview since the deal between China and Solomon Islands was leaked.",
              {"entities": [(77, 82, "COUNTRY"), (87, 102, "COUNTRY")]}),
              (
              "Australia should question whether it had been fair to Solomon Islands in its intense scrutiny of the deal.",
              {"entities": [(0, 9, "COUNTRY"), (54, 69, "COUNTRY")]}),
              ("Beck is believed to have been involved in negotiating the deal with China.",
               {"entities": [(68, 73, "COUNTRY")]}),
              ("Beck said that Solomon Islands faced domestic challenges.", {"entities": [(15, 30, "COUNTRY")]}),
              (
              "Chronic unemployment, as well as frustrations with the policies and leadership of the prime minister, Manasseh Sogavare, were thought to be behind riots in Honiara last year that left three people dead.",
              {"entities": [(156, 163, "CITY")]}),
              (
              "Riga is the capital of Latvia and is home to 605802 inhabitants (2022), which is a third of Latvia's population.",
              {"entities": [(0, 4, "CITY"), (23, 29, "COUNTRY"), (92, 98, "COUNTRY")]}),
              ("Being significantly larger than other cities of Latvia, Riga is the country's primate city.",
               {"entities": [(48, 54, "COUNTRY"), (56, 60, "CITY")]}),
              ("It is also the largest city in the three Baltic states.", {"entities": [(41, 54, "AREA")]}),
              ("It is home to one tenth of the three Baltic states' combined population. ",
               {"entities": [(37, 50, "AREA")]}),
              ("The city lies on the Gulf of Riga at the mouth of the Daugava river where it meets the Baltic Sea.",
               {"entities": [(17, 33, "WATER"), (50, 67, "WATER"), (83, 97, "WATER")]}),
              (
              "Riga's territory covers 307.17 km2 (118.60 sq mi) and lies 1–10 m (3.3–32.8 ft) above sea level, on a flat and sandy plain.",
              {"entities": [(0, 4, "CITY")]}),
              ("Riga was founded in 1201 and is a former Hanseatic League member.", {"entities": [(0, 4, "CITY")]}),
              ("Riga's historical centre is a UNESCO World Heritage Site.", {"entities": [(0, 4, "CITY")]}),
              ("Riga was the European Capital of Culture in 2014, along with Umea in Sweden. ",
               {"entities": [(0, 4, "CITY"), (61, 65, "CITY"), (69, 75, "COUNTRY")]}),
              (
              "Riga hosted the 2006 NATO Summit, the Eurovision Song Contest 2003, the 2006 IIHF Men's World Ice Hockey Championships, 2013 World Women's Curling Championship and the 2021 IIHF World Championship.",
              {"entities": [(0, 4, "CITY")]}),
              (
              "It is home to the European Union's office of European Regulators for Electronic Communications (BEREC).",
              {"entities": [(18, 32, "AREA")]}),
              ("In 2016, Riga received over 1.4 million visitors.", {"entities": [(9, 13, "CITY")]}),
              (
              "The city is served by Riga International Airport, the largest and busiest airport in the Baltic states.",
              {"entities": [(22, 48, "MANMADE"), (89, 102, "AREA")]}),
              (
              "Riga is a member of Eurocities, the Union of the Baltic Cities (UBC) and Union of Capitals of the European Union (UCEU).",
              {"entities": [(0, 4, "CITY"), (98, 112, "AREA")]}),
              ("Tallinn is the most populous, primate, and capital city of Estonia.",
               {"entities": [(0, 7, "CITY"), (59, 66, "COUNTRY")]}),
              ("It is situated on a bay in north Estonia, on the shore of the Gulf of Finland of the Baltic Sea.",
               {"entities": [(33, 40, "COUNTRY"), (58, 77, "WATER"), (81, 95, "WATER")]}),
              (
              "Tallinn has a population of 437811 (as of 2022) and administratively lies in the Harju maakond (county).",
              {"entities": [(0, 7, "CITY"), (77, 103, "AREA")]}),
              ("Tallinn is the main financial, industrial, and cultural centre of Estonia.",
               {"entities": [(0, 7, "CITY"), (66, 73, "COUNTRY")]}),
              (
              "It is located 187 km (116 mi) northwest of the country's second largest city Tartu, however only 80 km (50 mi) south of Helsinki, Finland.",
              {"entities": [(77, 82, "CITY"), (120, 128, "CITY"), (130, 137, "COUNTRY")]}),
              (
              "The city is located 320 km (200 mi) west of Saint Petersburg, Russia, 300 km (190 mi) north of Riga, Latvia, and 380 km (240 mi) east of Stockholm, Sweden.",
              {"entities": [(44, 60, "CITY"), (62, 68, "COUNTRY"), (95, 99, "CITY"), (101, 107, "COUNTRY"),
                            (137, 146, "CITY"), (148, 154, "COUNTRY")]}),
              (
              "From the 13th century until the first half of the 20th century, Tallinn was known in most of the world by variants of its other historical name Reval.",
              {"entities": [(64, 71, "CITY"), (144, 149, "CITY")]}),
              (
              "Tallinn received city rights in 1248, however the earliest evidence of human population in the area dates back nearly 5,000 years.",
              {"entities": [(0, 7, "CITY")]}),
              (
              "The first recorded claim over the place was laid by Denmark after a successful raid in 1219 led by King Valdemar II.",
              {"entities": [(52, 59, "COUNTRY")]}),
              (
              "In the 14-16th centuries Tallinn grew in importance as the northernmost member city of the Hanseatic League.",
              {"entities": [(25, 32, "CITY")]}),
              (
              "Tallinn Old Town is one of the best-preserved medieval cities in Europe and is listed as a UNESCO World Heritage Site.",
              {"entities": [(0, 16, "MANMADE"), (65, 71, "AREA")]}),
              (
              "Tallinn has the highest number of start-ups per person among European countries and is the birthplace of many international high-technology companies, including Skype, Bolt and Wise.",
              {"entities": [(0, 7, "CITY")]}),
              (
              "The city is home to the headquarters of the European Union's IT agency, and to the NATO Cyber Defence Centre of Excellence.",
              {"entities": [(44, 58, "AREA")]}),
              ("In 2007, Tallinn was listed among the top-10 digital cities in the world.",
               {"entities": [(9, 16, "CITY")]}),
              (
              "Tokyo (formerly Edo, historically Tokio, and officially the Tokyo Metropolis) is the capital and largest city of Japan.",
              {"entities": [(0, 5, "CITY"), (16, 19, "CITY"), (34, 39, "CITY"), (56, 76, "CITY"),
                            (113, 118, "COUNTRY")]}),
              ("It is located at the head of Tokyo Bay.", {"entities": [(29, 38, "WATER")]}),
              (
              "The prefecture forms part of the Kanto region on the central Pacific coast of Japan's main island of Honshu.",
              {"entities": [(29, 45, "AREA"), (78, 83, "COUNTRY"), (101, 107, "AREA")]}),
              ("Tokyo is the political and economic center of the country.", {"entities": [(0, 5, "CITY")]}),
              ("It is the seat of the Emperor of Japan and the national government.",
               {"entities": [(33, 38, "COUNTRY")]}),
              ("Originally the city was a fishing village, named Edo.", {"entities": [(49, 52, "CITY")]}),
              (
              "The city became a prominent political center in 1603, when it became the seat of the Tokugawa shogunate.",
              {"entities": [(85, 103, "COUNTRY")]}),
              ("By the mid-18th century, Edo was one of the most populous cities in the world at over one million.",
               {"entities": [(25, 28, "COUNTRY")]}),
              (
              "Following the end of the shogunate in 1868, the imperial capital in Kyoto was moved to the city, which was renamed Tokyo (literally 'eastern capital').",
              {"entities": [(68, 73, "CITY"), (115, 120, "CITY")]}),
              (
              "Tokyo was devastated by the 1923 Great Kanto earthquake, and again by Allied bombing raids during World War II.",
              {"entities": [(0, 5, "CITY"), (39, 44, "AREA")]}),
              (
              "Beginning in the 1950s, the city underwent rapid reconstruction and expansion, going on to lead Japan's post-war economic recovery.",
              {"entities": [(96, 101, "COUNTRY")]}),
              ("Tokyo is the largest urban economy in the world by gross domestic product.",
               {"entities": [(0, 5, "CITY")]}),
              ("It is part of an industrial region that includes the cities of Yokohama, Kawasaki, and Chiba. ",
               {"entities": [(63, 71, "CITY"), (73, 81, "CITY"), (87, 92, "CITY")]}),
              ("Tokyo is Japan's leading center of business and finance.",
               {"entities": [(0, 5, "CITY"), (9, 14, "COUNTRY")]}),
              (
              "In 2020, it ranked fourth on the Global Financial Centres Index, behind New York City, London, and Shanghai. ",
              {"entities": [(72, 85, "CITY"), (87, 93, "CITY"), (99, 107, "CITY")]}),
              ("The Tokyo Metro Ginza Line is the oldest underground metro line in East Asia (1927).",
               {"entities": [(4, 9, "CITY"), (67, 76, "AREA")]}),
              ("Japan's Shinkansen bullet train system.", {"entities": [(0, 5, "COUNTRY")]}),
              (
              "Notable districts of Tokyo include Chiyoda (the site of the National Diet Building and the Imperial Palace), Shinjuku (the city's administrative center), and Shibuya (a commercial, cultural and business hub).",
              {"entities": [(21, 26, "CITY"), (35, 42, "AREA"), (60, 82, "MANMADE"), (87, 106, "MANMADE"),
                            (109, 117, "AREA"), (158, 165, "AREA")]}),
              (
              "Palawan officially the Province of Palawan, is an archipelagic province of the Philippines that is located in the region of Mimaropa.",
              {"entities": [(0, 7, "AREA"), (19, 42, "AREA"), (75, 90, "COUNTRY"), (124, 132, "AREA")]}),
              ("The capital city is Puerto Princesa.", {"entities": [(20, 35, "CITY")]}),
              ("Palawan is known as the Philippines' Last Frontier and as the Philippines' Best Island.",
               {"entities": [(0, 7, "AREA"), (20, 35, "COUNTRY"), (58, 73, "COUNTRY")]}),
              ("The islands of Palawan stretch between Mindoro island in the northeast and Borneo in the southwest.",
               {"entities": [(15, 22, "AREA"), (39, 53, "AREA"), (75, 81, "AREA")]}),
              ("It lies between the South China Sea and the Sulu Sea.",
               {"entities": [(16, 35, "WATER"), (40, 52, "WATER")]}),
              ("The province is named after its largest island Palawan Island. ", {"entities": [(47, 61, "AREA")]}),
              (
              "In 2019, it was proposed to divide Palawan into three separate provinces, though it was rejected by the local population in a 2021 plebiscite.",
              {"entities": [(35, 42, "AREA")]}),
              (
              "Mimaropa (usually capitalized in official government documents), formally known as the Southwestern Tagalog Region, is an administrative region in the Philippines.",
              {"entities": [(0, 8, "AREA"), (83, 114, "AREA"), (147, 162, "COUNTRY")]}),
              (
              "It is one of two regions in the country having no land border with another region (the other being Eastern Visayas).",
              {"entities": [(99, 114, "AREA")]}),
              (
              "The name is an acronym combination of its constituent provinces: Mindoro (divided into Occidental Mindoro and Oriental Mindoro), Marinduque, Romblon and Palawan.",
              {"entities": [(65, 72, "AREA"), (87, 105, "AREA"), (110, 126, "AREA"), (129, 139, "AREA"),
                            (141, 148, "AREA"), (153, 160, "AREA")]}),
              ("The region was part of the now-defunct Southern Tagalog region until May 17, 2002.",
               {"entities": [(39, 62, "AREA")]}),
              (
              "On May 23, 2005, Palawan and the highly urbanized city of Puerto Princesa were moved to the region of Western Visayas by Executive Order No. 429.",
              {"entities": [(17, 24, "AREA"), (58, 73, "CITY"), (102, 117, "AREA")]}),
              (
              "On July 17, 2016, Republic Act No. 10879 formally established the Southwestern Tagalog Region to be known as Mimaropa discontinuing the 'Region IV-B' designation, however no boundary changes were involved.",
              {"entities": [(62, 93, "AREA"), (109, 117, "AREA"), (137, 148, "AREA")]}),
              ("Calapan is Mimaropa's regional center.", {"entities": [(0, 7, "CITY"), (11, 19, "AREA")]}),
              (
              "However, most regional government offices such as the Department of Public Works and Highways and the Department of Budget and Management are in Quezon City, Metro Manila.",
              {"entities": [(145, 156, "CITY"), (158, 170, "AREA")]}),
              (
              "However, on August 19, 2005, then-President Arroyo issued Administrative Order No. 129 to put in abeyance Executive Order No. 429 pending a review.",
              {"entities": []}),
              (
              "I went to China at 4th April, with my grandma and grandpa. That day, we got up at 7 o*clock in the morning. We went on the ship at 9:40am. When we reached China, we saw our relation, they saw us and helped us to carry the luggage. They are very kind to us. After about 1 hour, traveled by car, we reached home, it was about 1 o*clock, we were very hungry, they*ve already cooked the lunch for us. After lunch, we walked around in the village . When we back home, we chatted until dinner was ready. There was on toilet there, so my grandpa and me cannot eat more and more although all things were very good. After a terrible night, we went to another village. We went in the bus about 3 hours, after we reached there, it was about 4:30pm. I don*t think we could walked around the village again. The house that we lived was designed by my grandpa, we lived happily (very happy) there. -We had water, Town gas, light, TV * * Oh, great! We went back to Hong Kong at 8th. After a the tea break at the pipe, we got onto the ship at 4:00pm. I thought I was too full, I was seasick! We reached home at exactly 6 o*clock. My mom was already cooked the dinner for us. After we enjoyed the dinner, I went back home. At home , I had a hot bath that it was one of the best enjoys in my life.",
              {"entities": [(10, 15, "COUNTRY"), (155, 160, "COUNTRY"), (949, 958, "CITY")]}),
              (
              "We arrived at Omar Torrijos airport via American Airlines early in the afternoon. We purchased our required tourist cards (3 balboas, as US dollars are called in Panama) at the airport, then caught a taxi for the 18 mile ride to our downtown hotel. The ride in the battered, un-airconditioned car was rather expensive (30 balboas), but the driver spoke English and was very friendly. We arrived at the hotel and checked in. While my dad was checking in I bought a guidebook in the hotel lobby and read up on the history of Panama City. The original city was founded in 1519 by Pedro Arias Davila, known as Pedrarias the Cruel, because of his eradication of all but three of the local Indian tribes during his tenure in Panama. Davila used the city as a place to store Incan gold  before it was shipped to Spain. The original city was sacked and burned in 1671 by a group of buccaneers led by Henry Morgan. The city was rebuilt within a year, this time on a peninsula 18 miles away and surrounded by a strong wall. This old Spanish city is now the in the middle downtown Panama City.",
              {"entities": [(14, 35, "MANMADE"), (162, 168, "COUNTRY"), (523, 534, "CITY"), (719, 725, "COUNTRY"),
                            (805, 810, "COUNTRY"), (1070, 1081, "CITY")]}),
              (
              "Brazil lies between thirty five degrees west longitude and seventy five degrees west longitude.  Brazil also runs between five degrees north latitude and thirty five degrees south latitude.  Brazil is located in mainly the eastern part of South America.  This country sits in mostly the southern hemisphere of the world.  Being completely on the west side of the world, Brazil is not all in the south side of the world.  With the equator running through north Brazil, a small portion of Brazil, a small portion of Brazil is in the northern hemisphere.  Brazil is bordered by a number of South American countries.  Brazil borders Uruguay to the north; Argentina, Paraguay, Bolivia, and Peru to the east; Bogota to the southeast; Venezuela, Guyana, Suriname, and French Guiana to the south; and the Atlantic Ocean to the west.",
              {"entities": [(0, 6, "COUNTRY"), (97, 103, "COUNTRY"), (191, 197, "COUNTRY"), (239, 252, "AREA"),
                            (370, 376, "COUNTRY"), (460, 466, "COUNTRY"), (487, 493, "COUNTRY"), (514, 520, "COUNTRY"),
                            (553, 559, "COUNTRY"), (614, 620, "COUNTRY"), (629, 636, "COUNTRY"), (651, 660, "COUNTRY"),
                            (662, 670, "COUNTRY"), (672, 679, "COUNTRY"), (685, 689, "COUNTRY"), (703, 709, "COUNTRY"),
                            (728, 737, "COUNTRY"), (739, 745, "COUNTRY"), (747, 755, "COUNTRY"), (761, 774, "COUNTRY"),
                            (793, 811, "WATER")]}),
              (
              "The well-known country of New Zealand is a small, resourceful nation located 1,000 miles off Australia's south east coast. New Zealand has an impressive economy that continues to grow, a physical landscape that attracts people from around the globe, and although small, New Zealand is a respected nation for its advanced civilization and stable government. The geography of this prestigious nation can be described through five principal categories, the physical geography, the cultural geography, the citizens' standard of living, the government, and the nation's economy. New Zealand is located in the southern hemisphere, with an absolute location of 37 degrees south longitude to 48 degrees south longitude and 167 degrees east latitude to 177 degrees east latitude. It is composed of two major islands named the North and South Islands, and the total land area of the nation, approximately divided equally between the two islands, is 103,470 square miles. Surprisingly, only 2 percent of the land area is arable. New Zealand has an abundance of natural resources, explaining why the country is so wealthy compared to other nations. These resources include fertile grazing land, oil and gas, iron, coal, timber, and excellent fishing waters.",
              {"entities": [(26, 37, "COUNTRY"), (93, 102, "COUNTRY"), (123, 134, "COUNTRY"), (270, 281, "COUNTRY"),
                            (574, 585, "COUNTRY"), (813, 822, "AREA"), (827, 840, "AREA"), (1018, 1029, "COUNTRY")]}),
              (
              "New Zealand's climate is basically moderate year round because of the nearby ocean that regulates the climate.  New Zealand enjoys a marine west coast climate, that on average produces sixty to eighty degree temperatures in January and forty to sixty degree temperatures in July.  Because it is surrounded by the ocean, New Zealand receives immense quantities of precipitation on both islands.  The average annual precipitation on the North Island is thirty to forty inches and on the South Island it is forty to fifty inches.  This climate produces mixed forests, mid-latitude deciduous forests, and temperate grassland vegetation.  The land is blanketed with small lakes and rivers that drain the highlands and empty into the ocean.  The extraordinary diversity of the physical geography found in the United States seems to have been duplicated in this relatively small country, where the ski slopes and the beaches may be only an hour apart.",
              {"entities": [(0, 11, "COUNTRY"), (112, 123, "COUNTRY"), (320, 331, "COUNTRY"), (431, 447, "AREA"),
                            (481, 497, "AREA"), (799, 816, "COUNTRY")]}),
              (
              "New Zealand's government has contributed to its impressive standard of living.  New Zealand achieved independence from the United Kingdom on September 26, 1907.  The government was placed in Wellington, on the North Island, and still remains there today as the capital.  The government is a constitutional monarchy that was designed to resemble the United Kingdom government.  It includes an executive branch, legislative branch, judicial branch, and a King and Queen employed only as figureheads.  The military is divided into three branches, the New Zealand army, the Royal New Zealand Navy, and the Royal New Zealand Air Force.",
              {"entities": [(0, 11, "COUNTRY"), (80, 91, "COUNTRY"), (119, 137, "COUNTRY"), (191, 201, "CITY"),
                            (206, 222, "AREA"), (345, 363, "COUNTRY"), (548, 559, "COUNTRY"), (576, 587, "COUNTRY"),
                            (608, 619, "COUNTRY")]}),
              (
              "Thailand is a country in South East Asia.  Its neighboring countries are Cambodia on the east, Burma (now called Myanmar) on the west, Laos on the north, and Malaysia on the south.  The main river in Thailand is the Chao Phraya River which flows south out of the Mae Nam River.  The word nam means water in Thai.  Most of the rivers in Thailand start with Mae Nam something.  The Chao Phraya River (pronounced chow pee-ah) starts near the city of Singha Buri and flows south through Bangkok, the capital, and into the Gulf of Siam.",
              {"entities": [(0, 8, "COUNTRY"), (25, 40, "AREA"), (73, 81, "COUNTRY"), (95, 100, "COUNTRY"),
                            (113, 120, "COUNTRY"), (135, 139, "COUNTRY"), (158, 166, "COUNTRY"), (200, 208, "COUNTRY"),
                            (212, 233, "WATER"), (259, 276, "WATER"), (336, 344, "COUNTRY"), (376, 397, "WATER"),
                            (447, 458, "CITY"), (483, 490, "CITY"), (514, 530, "WATER")]}),
              ("Vilnius is the capital and largest city of Lithuania, with a population of 592,389 as of 2022.",
               {"entities": [(0, 7, "CITY"), (43, 52, "COUNTRY")]}),
              (
              "The population of Vilnius's functional urban area, which stretches beyond the city limits, is estimated at 718,507 (as of 2020).",
              {"entities": [(18, 0, "CITY")]}),
              (
              "According to the Vilnius territorial health insurance fund, there were 732,421 permanent inhabitants as of October 2020 in Vilnius city and Vilnius district municipalities combined.",
              {"entities": [(17, 24, "CITY"), (123, 135, "CITY"), (140, 156, "AREA")]}),
              (
              "Vilnius is situated in southeastern Lithuania and is the second-largest city in the Baltic states, but according to the Bank of Latvia is expected to become the largest in 2025.",
              {"entities": [(0, 7, "CITY"), (36, 45, "COUNTRY"), (80, 97, "AREA"), (128, 134, "COUNTRY")]}),
              ("It is the seat of Lithuania's national government and the Vilnius District Municipality.",
               {"entities": [(18, 27, "COUNTRY"), (54, 74, "AREA")]}),
              (
              "Vilnius is classified as a Gamma global city according to GaWC studies, and is known for the architecture in its Old Town, declared a UNESCO World Heritage Site in 1994.",
              {"entities": [(0, 7, "CITY")]}),
              ("Before World War II, Vilnius was one of the largest Jewish centres in Europe. ",
               {"entities": [(21, 28, "CITY"), (70, 76, "AREA")]}),
              ("Its Jewish influence has led to its nickname 'the Jerusalem of Lithuania'. ",
               {"entities": [(50, 59, "CITY"), (63, 72, "COUNTRY")]}),
              ("Napoleon called it 'the Jerusalem of the North' as he was passing through in 1812.",
               {"entities": [(24, 33, "CITY")]}),
              ("In 2009, Vilnius was the European Capital of Culture, together with Linz, Austria.",
               {"entities": [(9, 16, "CITY"), (68, 72, "CITY"), (74, 81, "COUNTRY")]}),
              (
              "In 2021, Vilnius was named among top-25 fDi's Global Cities of the Future – one of the most forward-thinking cities with the greatest potential in the World.",
              {"entities": [(9, 16, "CITY")]}),
              (
              "New York, often called New York City (NYC), is the most populous city of both New York State and the United States.",
              {"entities": [(0, 8, "CITY"), (23, 36, "CITY"), (78, 92, "AREA"), (97, 114, "COUNTRY")]}),
              (
              "With a 2020 population of 8,804,190 distributed over 300 square miles (780 km2) and divided into five boroughs, New York City is also the most densely populated major city in the United States.",
              {"entities": [(112, 125, "CITY"), (175, 192, "COUNTRY")]}),
              (
              "Located at the southern tip of the state of New York, the city is the center of the New York metropolitan area, the largest metropolitan area in the world by urban area.",
              {"entities": [(31, 52, "AREA"), (80, 110, "AREA")]}),
              (
              "With over 20.1 million people in its metropolitan statistical area and 23.5 million in its combined statistical area as of 2020, New York is one of the world's most populous megacities.",
              {"entities": [(129, 137, "CITY")]}),
              (
              "New York City has been described as the cultural, financial, and media capital of the world, and is a significant influence on commerce, entertainment, research, technology, education, politics, tourism, dining, art, fashion, and sports.",
              {"entities": [(0, 13, "CITY")]}),
              (
              "Home to the headquarters of the United Nations, New York is an important center for international diplomacy, an established safe haven for global investors.",
              {"entities": [(48, 56, "CITY")]}),
              (
              "The five boroughs - Brooklyn (Kings County), Queens (Queens County), Manhattan (New York County), the Bronx (Bronx County), and Staten Island (Richmond County) - were created when local governments were consolidated into a single municipal entity in 1898.",
              {"entities": [(20, 28, "AREA"), (30, 42, "AREA"), (45, 51, "AREA"), (53, 66, "AREA"), (69, 78, "AREA"),
                            (80, 95, "AREA"), (98, 107, "AREA"), (109, 121, "AREA"), (128, 141, "AREA"),
                            (143, 158, "AREA")]}),
              (
              "The city and its metropolitan area constitute the premier gateway for legal immigration to the United States.",
              {"entities": [(91, 108, "COUNTRY")]}),
              ("As many as 800 languages are spoken in New York.", {"entities": [(39, 47, "CITY")]}),
              # ("", {"entities": [(0, 0, "COUNTRY")]}),
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


# Function to update and retrain model
def update_model():
    global nlp, ner, model_dir, LABELS, TRAIN_DATA, optimizer, move_names, pipe_exceptions, other_pipes

    # Importing requirements
    from spacy.util import minibatch, compounding
    import random

    # Begin training by disabling other pipeline components
    with nlp.disable_pipes(*other_pipes):
        sizes = compounding(1.0, 4.0, 1.001)
        # Training for 100 iterations
        for itn in range(100):
            # shuffle examples before training
            random.shuffle(TRAIN_DATA)
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=sizes)
            # dictionary to store losses
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)

                example = []
                # Update the model with iterating each text
                for i in range(len(texts)):
                    doc = nlp.make_doc(texts[i])
                    example.append(Example.from_dict(doc, annotations[i]))

                # Calling update() over the iteration
                nlp.update(example, sgd=optimizer, drop=0.35, losses=losses)
                print("Losses", losses)

    # Saving the model to the output directory
    nlp.to_disk(model_dir)
    print("Saved model")


def find_entities(text):
    global nlp, ner, model_dir, LABELS, TRAIN_DATA, optimizer, move_names, pipe_exceptions, other_pipes

    should_update_model = input("Do you want to retrain the model (0/1): ")

    if (should_update_model != '0'):
        update_model()

    # Loading the model from the directory
    print("Loading created model...")
    nlp = spacy.load(model_dir)
    assert nlp.get_pipe("ner").move_names == move_names

    # text = input("Input text: ")

    article = nlp(text)
    found_entities = dict([(str(x), x.label_) for x in nlp(str(article)).ents])
    print('\n' + 'Found entities:' + '\n' + str(found_entities))

    colors = {"WATER": "#6281cc", "AREA": "#60ad47", "COUNTRY": "#d95b5b", "CITY": "#de8d5b", "MANMADE": "#96e3e3"}
    options = {"colors": colors}

    print('\n' + "Loading existing model...")
    nlp = en_core_web_lg.load()
    original_entities = dict(
        [(str(x), 'Uncategorized') for x in nlp(str(article)).ents if x.label_ == 'GPE' or x.label_ == 'LOC'])
    print('\n' + 'Found entities:' + '\n' + str(original_entities))

    print('\n' + "Combining results...")
    combined_entities = original_entities
    combined_entities.update(found_entities)

    # displacy.serve(article, style="ent", options=options) #http://localhost:5000  # this is what makes the visual

    return combined_entities
