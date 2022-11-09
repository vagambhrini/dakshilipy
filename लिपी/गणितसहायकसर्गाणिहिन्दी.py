import pandas as pd
from दाक्षिलिपीहिन्दी import *

def पदस्थान(a_list, value):
    try:
        return a_list.index(value)
    except ValueError:
        return None

def श्रेणिमेंप्रथमसमीकरणपुष्टीक्रमांक(श्रेणिपात्र, समीकरणपात्र):
  return next((क्रमांक for क्रमांक, पद in enumerate(श्रेणिपात्र) if समीकरणपात्र(पद)), None)

"""
def सारिणी(क):
    यदि पादसंख्या (क) == 0: सर्गफल अज्ञात
    फल = pd.DataFrame(क, index=क्रमश्रेणी(1, पादसंख्या (क)+1), columns = क्रमश्रेणी( 1,पादसंख्या(क[0])+1 ))
    सर्गफल फल 
"""

def सारिणी(क):
    if  len(क) == 0: return None
    फल = pd.DataFrame(क, index=क्रमश्रेणी(1, len (क)+1), columns = क्रमश्रेणी( 1,len(क[0])+1 ))
    return फल 

def दर्शयसारिणी(क):
    दर्शय(सारिणी(क))
