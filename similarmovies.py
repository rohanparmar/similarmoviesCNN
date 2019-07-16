
# coding: utf-8

# # Similar Movies CNN

# Loading list of movies and ratings

# In[3]:


import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('D:\\Tutorials\Data Science and Machine Learning with Python\\DataScience-Python3\\ml-100k\\u.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")

m_cols = ['movie_id', 'title']
movies = pd.read_csv('D:\\Tutorials\\Data Science and Machine Learning with Python\\DataScience-Python3\\ml-100k\\u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

ratings = pd.merge(movies, ratings)


# In[4]:


ratings.head()


# NaN -> Movies that haven't been rated

# In[5]:


movieRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
movieRatings.head()


# Taking example of Star Wars ->

# In[6]:


starWarsRatings = movieRatings['Star Wars (1977)']
starWarsRatings.head()


# Finding correlation vectors using corrwith(): ->

# In[7]:


similarMovies = movieRatings.corrwith(starWarsRatings)
similarMovies = similarMovies.dropna()
df = pd.DataFrame(similarMovies)
df.head(10)


# In[8]:


similarMovies.sort_values(ascending=False)


# Constructing new data frame:

# In[9]:


import numpy as np
movieStats = ratings.groupby('title').agg({'rating': [np.size, np.mean]})
movieStats.head()


# Applying review threshold of minimum 100 ->

# In[10]:


popularMovies = movieStats['rating']['size'] >= 100
movieStats[popularMovies].sort_values([('rating', 'mean')], ascending=False)[:15]


# Joining with original set of ratings ->

# In[11]:


df = movieStats[popularMovies].join(pd.DataFrame(similarMovies, columns=['similarity']))


# In[12]:


df.head()


# Sorting result by similarity score ->

# In[13]:


df.sort_values(['similarity'], ascending=False)[:15]

