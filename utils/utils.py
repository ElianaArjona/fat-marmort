import re
from unidecode import unidecode
import spacy 

# Load the Spanish language model in spaCy
nlp = spacy.load("es_core_news_sm")

specifications_dict = {
            # Apartment Specifications
            "lb": "linea blanca",
            "linea blanca":"linea blanca",
            "l b":"linea blanca",
            "amoblado":"amoblado",
            "muebles":"amoblado",
            "semi":"amoblado",
            
            # Locations
            "costa del este": "costa del este",
            "cde": "costa del este",
            "coco del mar":"coco del mar",
            "paitilla" : "paitilla",
            "sf": "san francisco",
            "san francisco": "san francisco",
            "cds": "ciudad del saber",
            "ciudad del saber": "ciudad del saber",
            "albrook":"albrook",
            "revertido": "revertidas",
            "revertida": "revertidas",

            # Verbs
            "busca":"busqueda",
            "necesito":"busqueda",
            "oferzco":"oferta",
            "oferto":"oferta",
            "venta":"venta",
            "alquiler":"alquiler",
            "compra":"venta",

            # House Type
            "casa":"casa",
            "hosue":"casa",
            "estudio":"estudio",            
            "apt":"apt",
            "aparatmento":"apt",
            "aprmt":"apt",
            "ph":"apt"
            }

def remove_emojis_and_colon_text(text):
        # Step 2: Remove emojis and any text between colons (e.g. :smile: or :round_pushpin:)
        emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags=re.UNICODE)

        # Remove emojis
        text = emoji_pattern.sub(r"", str(text))
        
        # Remove text between colons
        text = re.sub(r":[^:\s]+:", "", text)
        
        return text


def remove_special_characters(text):
    # Remove accents from the text
    text = unidecode(text)
#     sentence = []
    
    # Remove special characters except for the dollar sign ($)
    for c in text:
        if c == '$' or c.isalnum():
           sentence = re.sub(r'[^A-Za-z0-9$ ]+', '', text)
           sentence = re.sub(r'\s+', ' ', sentence).strip()
        
#     text = ''.join(c for c in text if c.isalnum() or c == '$')
    return sentence

def remove_spanish_stopwords(text):
    
    # Tokenize the text
    doc = nlp(text)
    
    # Remove stopwords from the tokenized text
    tokens_without_stopwords = [token.text for token in doc if not token.is_stop]
    
    # Reconstruct the text without stopwords
    text_without_stopwords = ' '.join(tokens_without_stopwords)
    
    return text_without_stopwords
