'''
Code for encyclopedia dramatica data sprint day 1
1. linguistic diversity of encyclopedia dramatica vs. other texts (incl. neologisms)
2. Linguistic diversity / neologisms over time?
3. Percentage of links that are also neologisms?

'''

# %% 
#import these libraries 

import nlp_tools as nlp 
import glob
import pandas as pd
import matplotlib.pyplot as plt
import re
# %%
#get data and append to corpus
corpus = []
for filepath in sorted(glob.glob('article_samples/*.txt')):
    print('opening ' + filepath)
    text = open(filepath).read()
    corpus.append(text)
    

# %% 
#number of neologisms / total number of words?

reference = nlp.load_reference_corpus()
# %%

neologism_track = []

for text in corpus:

    #map neologism distribution
    neologisms = nlp.get_neologisms(text, reference)
    sorted_neologisms = {k: v for k, v in sorted(neologisms.items(), key=lambda item: item[1])}

    neologism_track.append(sorted_neologisms)

    #plot distribution
    plt.figure(figsize = (20, 10))
    plt.bar(*zip(*sorted_neologisms.items()))
    plt.xticks(rotation=70)
    plt.show()

    #get the total neologism count
    values = sorted_neologisms.values()
    neologism_count = sum(values)

    #get the doc size (number of tokens)
    doc_size = nlp.get_doc_size(text)

    #calculate neologism/size ration

    ratio = neologism_count/doc_size
    print('neologisms to text ratio: ', ratio)



# %%
#Track added or removed neologisms
print(neologism_track)





# %%



# %%
