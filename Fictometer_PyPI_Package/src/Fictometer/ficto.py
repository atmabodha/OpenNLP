from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from .tags import n_adj, n_adv, n_pronoun, n_verb, n_noun
import pickle
import pkg_resources

import warnings
warnings.filterwarnings('ignore')
    
def pos_count(text):
    tagged_words= pos_tag(word_tokenize(text), tagset='universal')
    taglist=[]
    for k in tagged_words:
        taglist.append(k[1])
    
    adj=n_adj(taglist)
    adv=n_adv(taglist)
    # noun=n_noun(taglist)
    # verb=n_verb(taglist)
    pronoun=n_pronoun(taglist)
    
    return [adj, adv, pronoun]

def pos_ratio(pc):
    RADJPRON = pc[0]/pc[2]
    RADVADJ = pc[1]/pc[0]
    
    return [RADJPRON, RADVADJ]


def fictometer(pr):
    sav_file = pkg_resources.resource_filename(__name__, 'fictometer.sav')
    logreg = pickle.load(open(sav_file, 'rb'))
    output = logreg.predict([[pr[0], pr[1]]])
    prob = logreg.predict_proba([[pr[0], pr[1]]])

    if output == 1:
        return "Fiction", f"confidence = {prob}"
    else:
        return "Non-Fiction", f"confidence = {prob}"
    
def help():
    
    print("""pos_count(text):
It takes a text as input and returns a list having the count of 'adjectives', 'adverbs' and 'pronouns' in that text respectively.

pos_ratio(pc):
It takes list returned by pos_count() as input and returns a list having 'adjective/pronoun' ratio and 'adverb/adjective' ratio respectively.

fictometer(pr):
It takes list returned by pos_ratio() as input and returns a tuple having 'result' and 'confidence'.""")
