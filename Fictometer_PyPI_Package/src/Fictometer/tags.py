import warnings
warnings.filterwarnings('ignore')

# Define functions to count the number of POS tags in the text.

# This function counts the number of adjectives
def n_adj(text):
    adj=0
    for i in text:
        if i[0] == 'J' or i == 'ADJ':
            adj=adj+1
    return adj

# This function counts the number of nouns
def n_noun(text):
    noun=0
    for i in text:
        if ((i[0] == 'N') and (i[1] != 'C')) or i == 'NOUN':
            noun=noun+1
    return noun

# This function counts the number of verbs
def n_verb(text):
    verb=0
    for i in text:
        if i[0] == 'V' or i == 'VERB':
            verb=verb+1
    return verb

# This function counts the number of pronouns
def n_pronoun(text):
    pronoun=0
    for i in text:
        if (i[0] == 'P') or (i[:3] in ['WP$','WPO','WPS']) or i == 'PRON':
            pronoun=pronoun+1
    return pronoun

# This function counts the number of adverbs
def n_adv(text):
    adv=0
    for i in text:
        if (i[0] == 'R') or (i[:3] in ['WRB']) or i == 'ADV':
            adv=adv+1
    return adv

# # This function outputs the universal high level tag using a finer tag as input
# def func_utag(tag):
#     if tag[0] == 'J' or tag == 'ADJ':
#         utag='ADJ'
#     elif ((tag[0] == 'N') and (tag[1] != 'C')) or tag == 'NOUN':
#         utag='NOUN'
#     elif tag[0] == 'V' or tag == 'VERB':
#         utag='VERB'
#     elif (tag[0] == 'P') or (tag[:3] in ['WP$','WPO','WPS']) or tag == 'PRON':
#         utag='PRON'
#     elif (tag[0] == 'R') or (tag[:3] in ['WRB']) or tag == 'ADV':
#         utag='ADV'
#     else:
#         utag='unknown'
#     return utag

# # This function outputs True or False depending on whether the input tag is one of the 5 high level universal tags or not.
# def func_is5tag(tag):
#     if tag in ['ADJ','ADV','NOUN','PRON','VERB']:
#         is5tag=True
#     else:
#         is5tag=False
#     return is5tag