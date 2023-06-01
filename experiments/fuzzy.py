import spacy
from spaczz.matcher import FuzzyMatcher

# nlp = spacy.blank("en")
nlp = spacy.load("es_core_news_sm")

text = """Regalia - Cosa del Este  3 Recámaras 3.5 Baños 2 Estacionamientos  Cuaro & Baño Servicio  Den 293 Metros 2  Piso Bajo - Modelo A Línea Blanca & Muebles  $4,500 Mensuales,
*Alquilo Av. Balboa* *PH GRANDBAY TOWER* 
ALQUILER:  *Amoblado*  *$. 1,450.00* *Negociable* Metraje: *105 M2* - Habitación principal con baño, ducha y walking closet. - Habitación secundaria con closet y baño. - Sala - Comedor - Cocina abierta - Lavandería. - Un (1) parking. *Información* *Nelly* *LIC PJ 0505-07* ☎️ *67882098* ☎️ *393-5895*

"""  # Spelling errors intentional.
doc = nlp(text)

matcher = FuzzyMatcher(nlp.vocab)
matcher.add("NAME", [nlp("Costa del Este")])
matcher.add("GPE", [nlp("Cuarto")])
matches = matcher(doc)

# for match_id, start, end, ratio, pattern in matches:
#     print(match_id, doc[start:end], ratio, pattern)


for token in doc:
    token_text = token.text
    token_pos = token.pos_
    token_dep = token.dep_
    
    print(f"{token_text:<12}{token_pos:<10}{token_dep:<10}")

for ent in doc.ents:
    print(ent.text, ent.label_)