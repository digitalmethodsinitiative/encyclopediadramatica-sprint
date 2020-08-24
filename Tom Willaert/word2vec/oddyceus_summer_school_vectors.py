# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### Us vs. Them Dimension ###
# 
# Inspired by the works of Austin C. Kozlowski, Matt Taddy and James A. Evans ([2019](https://arxiv.org/pdf/1803.09288.pdf)) we intend to measure representation of individuals and entities on an in-group out-group space. To do se we focus on three online communities: the /pol/ forum on 4chan.org, the_donald subreddit and breibart comment section. 
# 
# From a high dimensionnal space defined by word embeddings we create an "us-them" dimensions from a list of hand-picked antagonist and protagonist terms. Each embeddings are trained months between  2016-05 and 2017-03. We then project political actors, media outlets, communities along this dimension.
# 
# We are interested in the change of these entities between the media and through time.
# 
# We first load the word embeddings for all of the media and the 12 months (May 2016 to May 2017). Then we load the antagonist and protagonist terms.

# %%
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, KeyedVectors

sns.set_style("dark")

# %%
def load_models(platforms, months):
    models = {}
    for platform in platforms:
        platform_dict = {}
        directory = "../data/models/{}/w2v_models/".format(platform)
        for filename in os.listdir(directory):
            if filename.endswith(".bin") and not filename == "2017-04.bin":
                filepath = os.path.join(directory, filename)
                month = os.path.splitext(filename)[0]
                model = KeyedVectors.load_word2vec_format(filepath,binary=platform=='4chan_pol')
                platform_dict[month] = model
        models[platform] = platform_dict
    return models

# load models
platforms = ["4chan_pol","breitbart","the_donald"]

months= ["2016-05","2016-06","2016-07","2016-08","2016-09","2016-10",
         "2016-11","2016-12","2017-01","2017-02","2017-03"]
models = load_models(platforms, months)

# load entities
# entities = ['trump','bernie','hillary']


# load seed terms
antagonist = open('../data/seeds/antagonist.txt').read().splitlines()
protagonist = open('../data/seeds/protagonist.txt').read().splitlines()

# %% [markdown]
# We then calculate a vector to represent the prototypical antagonist vector and the prototypical protagonist vector. The Us vs. Them dimension is the difference between the vectors.

# %%
def dimension_create(positive,negative , name, model):
    """ takes two list of words expected to be negative and positive on the newly created dimension. 
        name the dimension.
        return the updated model """
    for platform in model:
        for month in model[platform]:
            word_matrix = model[platform][month]
            antagonist_vector = np.mean([word_matrix[meme] for meme in negative if meme in word_matrix], axis=0)
            protagonist_vector = np.mean([word_matrix[meme] for meme in positive if meme in word_matrix], axis=0)

            new_dimension = antagonist_vector - protagonist_vector
            word_matrix.add(name, new_dimension )
            models[platform][month] = word_matrix
    return models


# %%
models = dimension_create(negative =  antagonist,
                     positive = protagonist,
                     name = 'usthem_dimension', 
                     model = models)


# %%
def rank_over_time(platforms, entities):
    measured_similarity = {"platform": [], "entity":[], "month":[], "rank":[]}
    for platform in platforms:
        for month in models[platform]:
            ranking = pd.DataFrame(models[platform][month].most_similar('usthem_dimension', topn=2000000))
            ranking = ranking.reset_index()
            for entity in entities:
                rank = abs(1 - ranking.loc[ranking[0] == entity].index[0] / len(ranking.index))
                measured_similarity["platform"].append(platform)
                measured_similarity["entity"].append(entity)
                measured_similarity["month"].append(month)
                measured_similarity["rank"].append(rank)

    return measured_similarity


# %%
#save to csv
# df.to_csv("../data/politican_against_them.csv", sep='\t', encoding='utf-8')


# %%
entities = ['science','army']
platform = ['4chan_pol', 'the_donald', 'breitbart']

df = pd.DataFrame(rank_over_time(platforms, entities))
plt.figure(figsize=(10, 6))
ax = sns.lineplot(x="month", y="rank", hue="entity", data=df.loc[df['platform'] == platform[1]])
plt.xlabel('time')
plt.ylabel('distance')


# %%
plt.figure(figsize=(10, 6))
ax = sns.lineplot(x="month", y="rank", hue="platform", data=df)
plt.xlabel('time')
plt.ylabel('similarity')


# %%
df.head()


# %%
# plotting entities on two created dimension
def bi_dimensional_rank(entities, dimensions, model):
    """ dimension is a named list"""
    measured_similarity = {"platform": [], "entity":[], "month":[],"rank":[], "dimension":[]}
    for platform in model:
        print(platform)
        for month in model[platform]:
            print(month)
            for dimension in dimensions: 
                print(dimension)
                ranking = pd.DataFrame(model[platform][month].most_similar(dimension, topn=2000000))
                ranking = ranking.reset_index()
                print(ranking)
                for entity in entities:
                    print(entity)
                    rank = abs(1 - ranking.loc[ranking[0] == entity].index[0] / len(ranking.index))
                    measured_similarity["platform"].append(platform)
                    measured_similarity["entity"].append(entity)
                    measured_similarity["month"].append(month)
                    measured_similarity["rank"].append(rank)
                    measured_similarity["rank"].append(dimension)
                    print("ok")

    return measured_similarity


# %%
def rank_over_time_2(platforms, entities, dimension, model):
    measured_similarity = {"platform": [], "entity":[], "month":[], "rank":[]}
    for platform in model:
        for month in model[platform]:
            ranking = pd.DataFrame(model[platform][month].most_similar(dimension, topn=2000000))
            ranking = ranking.reset_index()
            for entity in entities:
                rank = abs(1 - ranking.loc[ranking[0] == entity].index[0] / len(ranking.index))
                measured_similarity["platform"].append(platform)
                measured_similarity["entity"].append(entity)
                measured_similarity["month"].append(month)
                measured_similarity["rank"].append(rank)

    return measured_similarity

models = dimension_create(positive = antagonist, 
                     negative = protagonist,
                     name = 'us_them', 
                     model = models)

models = dimension_create(positive = ["conservative"],
                     negative = ["liberal"],
                     name = 'conservative_lib',
                     model = models2)
df2 = pd.DataFrame(rank_over_time_2(platforms, entities,"us_them",models2))


# %%
# print(models2["4chan_pol"]["2016-05"].vectors)
#print(models2["4chan_pol"])
# pd.DataFrame(models2["4chan_pol"]["2017-01"]["us_them"])
models2["4chan_pol"]["2017-01"].most_similar("us_them")


# %%
#entities = ['hillary','trump', 'bernie']
entities = ['science','joe', 'cop']
# platform = ['4chan_pol', 'the_donald', 'breitbart']

df = pd.DataFrame(rank_over_time(platforms, entities))
#plt.figure(figsize=(10, 6))
#ax = sns.lineplot(x="month", y="rank", hue="entity", data=df.loc[df['platform'] == platform[0]])
#plt.xlabel('time')
#plt.ylabel('distance')

sns.set_style("white")
g = sns.FacetGrid(df, col="platform",height=5, hue="entity")
g.map(sns.lineplot, "month", "rank").add_legend()


# %%
models2 = dimension_create(positive = antagonist, 
                     negative = protagonist,
                     name = 'us_them', 
                     model = models)

models2 = dimension_create(positive = ["conservative"],
                     negative = ["liberal"],
                     name = 'conservative_lib',
                     model = models2)

dx = bi_dimensional_rank(["trump","hillary"],["us_them","conservative_lib"], models2)


