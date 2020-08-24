'''
Data sprint day 2. Word2vec models for 'us' vs. 'them' terms and other experiments
Tom Willaert, VUB AI Lab tom@ai.vub.ac.be
'''

#%% 
#import these libraries 
from gensim.models import Word2Vec
from gensim.models import phrases
from sklearn.decomposition import PCA
from matplotlib import pyplot
from nltk import tokenize
import pandas as pd
import mwparserfromhell as mw
from bs4 import BeautifulSoup
from tqdm import tqdm
import re


#%%
#GENERATE MODELS FOR FIRST ED SNAPSHOT LOWERCASE (2010)

sentences_2010 = []
data = pd.read_csv('ED_data/ED_data_2010.csv')
print(data.head())
for text in tqdm(data['body']):
    if type(text) is str:
        text = text.lower() #lowercase
        tokens = [tokenize.word_tokenize(t) for t in tokenize.sent_tokenize(text)]
        sentences_2010.extend(tokens)

#train models
print('training (unigram) model')
model1a = Word2Vec(sentences_2010) #default model, min_counts = 5
model1a.save("models/ED_4cat_snapshot1_lowercase.model")

print('training bigram model')
bigrams = phrases.Phrases(sentences_2010)
model1b = Word2Vec(bigrams[sentences_2010]) #default model, min_counts = 5
model1b.save("models/ED_4cat_snapshot1_bigrams_lowercase.model")


#%%
#GENERATE MODELS FOR SECOND ED SNAPSHOT LOWERCASE (2020)
sentences_2020_lower = []

data = pd.read_csv('ED_data/ED_data_2020.csv')
print(data.head())
for text in tqdm(data['body']):
    text = text.lower() #make lowercase
    tokens = [tokenize.word_tokenize(t) for t in tokenize.sent_tokenize(text)]
    sentences_2020_lower.extend(tokens)


print('training first 2020 model')
model2a = Word2Vec(sentences_2020_lower) #default model, min_counts = 5
model2a.save("models/ED_4cat_snapshot2_lowercase.model")

print('training second 2020 model')
bigrams = phrases.Phrases(sentences_2020_lower)
model2b = Word2Vec(bigrams[sentences_2020_lower]) #default model, min_counts = 5
model2b.save("models/ED_4cat_snapshot2_bigrams_lowercase.model")


#%%
#GENERATE MODELS FOR THIRD ED SNAPSHOT LOWERCASE (2020, excluding 2010 set)
sentences_2020b = []

data = pd.read_csv('ED_data/ED_data_2010_2020.csv')
for text in tqdm(data['body']):
    text = text.lower() #make lowercase
    tokens = [tokenize.word_tokenize(t) for t in tokenize.sent_tokenize(text)]
    sentences_2020b.extend(tokens)

#train models
print('training first 2020b model')
model3a = Word2Vec(sentences_2020b) #default model, min_counts = 5
model3a.save("models/ED_4cat_snapshot3_lowercase.model")

print('training second 2020b model')
bigrams = phrases.Phrases(sentences_2020b)
model3b = Word2Vec(bigrams[sentences_2020b]) #default model, min_counts = 5
model3b.save("models/ED_4cat_snapshot3_bigrams_lowercase.model")


# %%
#load models for analysis

model1a = Word2Vec.load("models/ED_4cat_snapshot1_lowercase.model")
model1b = Word2Vec.load("models/ED_4cat_snapshot1_bigrams_lowercase.model")

model2a = Word2Vec.load("models/ED_4cat_snapshot2_lowercase.model")
model2b = Word2Vec.load("models/ED_4cat_snapshot2_bigrams_lowercase.model")

model3a = Word2Vec.load("models/ED_4cat_snapshot3_lowercase.model")
model3b = Word2Vec.load("models/ED_4cat_snapshot3_bigrams_lowercase.model")

#%% overall term analysis CSV

def get_terms_df(terms, model_to_use, class_name, year_label):
    terms_df = pd.DataFrame()
    terms_complete = []
    terms_complete.extend(terms)
    for term in terms:
        if not term == term.lower():
            terms_complete.append(term.lower())


    for term in terms_complete:    
        print(term)
        if term in model_to_use.wv.vocab:
            print(term)
            similar_terms = []
            for similar_term in model_to_use.most_similar(positive = [term]):
                similar_terms.append(similar_term)
            table_row = {'term': term, 'class': class_name, 'year': year_label, 'words used in similar contexts (Gensim word2vec)' : similar_terms}
            terms_df = terms_df.append(table_row, ignore_index=True)
    return terms_df



terms = ['nigger','internets', 'fag', 'chris-chan', 'cam_whore', 'incel']

completed_df = pd.DataFrame()

df_2010 = get_terms_df(terms, model1b, 'lowercase', '2010')
df_2020 = get_terms_df(terms, model2b, 'lowercase', '2020')
df_2020b = get_terms_df(terms, model3b, 'lowercase', '2010-2020')

completed_df = completed_df.append(df_2010)
completed_df = completed_df.append(df_2020)
completed_df = completed_df.append(df_2020b)

completed_df = completed_df[['term', 'class', 'year', 'words used in similar contexts (Gensim word2vec)']] 
completed_df.to_csv('outputs/terms_overview.csv')  



# %% 
#get analysis csv file per term
def get_df_per_term(terms, model_to_use, year_label):
    terms_df = pd.DataFrame()
    terms_complete = []
    terms_complete.extend(terms)
    for term in terms:
        if not term == term.lower():
            terms_complete.append(term.lower())

    for term in terms_complete:    
        print(term)
        if term in model_to_use.wv.vocab:
            term_df = pd.DataFrame()
            print(term)
            for similar_term in model_to_use.most_similar(positive = [term]):    
                table_row = {'term': term, 'year': year_label, 'similar word (word2vec)': similar_term[0], 'similarity score (word2vec)': similar_term[1]}
                term_df = term_df.append(table_row, ignore_index=True)
            filename = 'outputs/' + term + year_label + '.csv'
            term_df = term_df[['term', 'year', 'similar word (word2vec)', 'similarity score (word2vec)']]
            term_df.to_csv(filename)  
    return terms_df



terms = ['nigger','internets', 'fag', 'chris-chan', 'cam_whore', 'incel']


df_2010 = get_df_per_term(terms, model1b, '2010')
df_2020 = get_df_per_term(terms, model2b, '2020')
df_2020b = get_df_per_term(terms, model3b, '2010-2020')




'''
SCRATCHPAD
'''
#%% ANALYSIS OF US and THEM TERMS


us_terms = ['EDiot', 'Dramacrat', 'You', 'Anon', 'Lulz', 'Troll', 'iPod', 'Internet','drama', 'Anonymous', 'OTI', 'GNAA']

them_terms = ['Chad', 'whore', 'Wikipedophile', 'nigger', 'kike', 'Liberalism', 'Furfag', 'Jews', 'Hipster', 'Jailbait', 'Islam', 'faggot', 'Mexico', 'Moderator', 'Tard', 'Christian', 'Yahweh', 'Cracker', 'Redneck', 'Otherkin', 'Brony', 'Chris-chan', 'Newfag', 'Memefag']



terms = ['aol', 'firefox', '4chan', 'myspace', 'ebay', 'no show', 'sif', 'spelling', 'mebbe', 'net', 'isp', 'nsfw',
'googlebanger', 'interweb', 'interwebz', 'encyclopedia_dramatica', 'internet_explorer', 'irc', 'tubes', 'america_online',
'miso', 'a/w', 'tumblr', 'srsly', 'honda-tech', 'philski', 'bbl', 'addiction', 'wifi', 'netizen', 'tiri', 'shoop', 'fwd', 'skweezy_jibbs',
'imdb', 'dsl', 'iwc', 'brb', 'mansplaining', 'nacho_vidal', 'danisnotonfire', 'qft', '({.})', 'wi-fi', '(y)', 
'hawt','nsfw','intarnet', 'wikipedia', 'the_cloud', 'nextgenupdate', 'backseat_surfer', 'chuck_norris', 
'walrusguy', 'internet_girlfriend', 's/n', 'suxorz', 'tfw', 'swear_word', 'gope', 'sifs', 'linksys', 'itc', 'sauce','bubb_rubb','furries',
'xenimus', 'gdi', 'warboards', 'disinterneted', 'xxoo', 'virus','windoze','idiot_proof', 'capped','interpreneur', 'guise',
'innernet']

with open('ED_data/internet_slang.txt') as f:
    lines = [line.rstrip() for line in f]
    for line in lines:
        terms.append(str(line))
#%%
#calculations experiment
#arithmetic = model.similar_by_vector(model['Encyclopedia_Dramatica'] - model['us'] + model['them']) 
arithmetic = model.similar_by_vector(model['EDiots'] - model['us'] + model['them']) 
arithmetic

#%%
# visualization experiment



# fit a 2d PCA model to the vectors
X = model1b[model1b.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)

# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model1b.wv['nigger'])
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()


# %%

data1 = pd.read_csv('ED_data/4cat_ED_snapshot1.csv')
data2 = pd.read_csv('ED_data/4cat_ED_snapshot2_clean.csv')
data3 = pd.read_csv('ED_data/4cat_ED_snapshot3.csv')
# %%
print(data1.info())
print(data2.info())
print(data3.info())
# %%
