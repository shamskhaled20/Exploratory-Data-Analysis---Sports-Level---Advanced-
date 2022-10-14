#!/usr/bin/env python
# coding: utf-8

# 1. Loading Data:

# The dataset is borrowed from https://bit.ly/34SRn3b .

# In[7]:


#importing required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


#Loading data into dataframe

data_deliveries = pd.read_csv('Downloads/Indian Premier League/deliveries.csv')
data_deliveries.head()


# In[9]:


#Loading data into dataframe

data_matches = pd.read_csv('Downloads/Indian Premier League/matches.csv')
data_matches.head()


# 2. Familiarizing with Data:

# Analysing Deliveries Dataset

# In[16]:


#Shape of dataframe

data_deliveries.shape


# In[17]:


data_deliveries.describe()


# In[18]:


data_deliveries.info()


# In[19]:


#Listing the features of the dataset

data_deliveries.columns


# In[20]:


#checking for null value

data_deliveries.isna().sum()


# In[21]:


#unique value in dataset

data_deliveries.nunique() 


# Analysing Matches Dataset

# In[23]:


#Shape of dataframe

data_matches.shape


# In[24]:


#Listing the features of the dataset

data_matches.columns


# In[25]:


#Information about the dataset

data_matches.info()


# In[26]:


#checking for null value

data_matches.isna().sum()


# In[27]:


# describtion of dataset

data_matches.describe()


# In[28]:


#unique value in dataset

data_matches.nunique()   


# In[29]:


data_matches.rename(columns={'win_by_runs':'Bat_1', 'win_by_wickets':'Ball_1'}, inplace=True)


# In[30]:


data_matches.duplicated().sum()


# 3. Visualizing the data:

# In[31]:


plt.figure(figsize=(10,6))
"""style1  = {'family': 'Times New Roman', 'color': 'Tomato', 'size': 25}
style2  = {'family': 'Times New Roman', 'color': 'DodgerBlue', 'size': 20}"""
sns.barplot( data_matches['toss_winner'].value_counts().index,data_matches['toss_winner'].value_counts().values)
plt.title('Toss Win Count by Team' )
plt.xlabel('Players' )
plt.xticks(rotation=90)
plt.ylabel('Count' )
plt.show()


# In[32]:


plt.subplots(figsize=(15,8))
"""style1  = {'family': 'Times New Roman', 'color': 'Tomato', 'size': 25}
style2  = {'family': 'Times New Roman', 'color': 'DodgerBlue', 'size': 20}"""
sns.barplot(data_matches['player_of_match'].value_counts()[:10].index, data_matches['player_of_match'].value_counts()[:10].values)
plt.title('Top 10 Players of Match')
plt.xlabel('Players' )
plt.ylabel('Count')
plt.show()


# In[33]:


data_matches.corr().abs()


# In[34]:


sns.heatmap(data_matches.corr(), annot=True, cmap='magma');


# In[35]:


data_matches= data_matches.drop(['dl_applied', 'season'], axis=1)
data_matches.head()


# In[36]:


plt.figure(figsize=(10,10))
sns.countplot(data_matches['toss_winner'])
plt.xlabel('Teams')
plt.ylabel('Count')
plt.title('Teams that Won the Toss')
plt.xticks(rotation=90)
plt.show()


# In[37]:


print('Team that won most matches by Batting First: ',data_matches.iloc[data_matches[data_matches['Bat_1'].ge(1)].Bat_1.idxmax()]['winner'])


# 4. Data Analysis:
# 

# 4.1. Merging the two Datasets into a new Dataset and Reading it (join on match-id)
# 

# In[38]:


data_merge=pd.merge(data_deliveries, data_matches, left_on='match_id', right_on='id')
data_merge.head()


# In[39]:


print('Shape:', data_merge.shape)
print('Size:', data_merge.size)


# In[40]:


data_merge.isna().sum()


# In[41]:


data_merge['player_dismissed'].fillna(value='NA', inplace=True)
data_merge.isnull().sum()


# In[42]:


data_merge.duplicated().sum()


# In[43]:


data_merge.drop_duplicates()


# In[44]:


data_merge.corr().abs()


# In[45]:


plt.figure(figsize=(15,15))
sns.heatmap(data_merge.corr(), annot=True, linewidth=1, cmap='Oranges');


# Number of Matches Played in Each Stadium
# 

# In[47]:


delivery=data_matches
delivery.venue.value_counts()

plt.figure(figsize=(10,10))
sns.countplot(data=delivery, x='venue')
plt.xticks(rotation=90)
plt.show()


# Most matches have been played in Eden Gardens followed by Wankhede Stadium. Teams who win toss choose to field first
# 
# 

# In[48]:


team_stats = pd.DataFrame({'Total Matches played': data_matches.team1.value_counts() + data_matches.team2.value_counts(), 'Total won': data_matches.winner.value_counts(), 'Toss won': data_matches.toss_winner.value_counts(), 
                          'Total lost': ((data_matches.team1.value_counts() + data_matches.team2.value_counts()) - data_matches.winner.value_counts())})
team_stats = team_stats.reset_index()
team_stats.rename(columns = {'index':'Teams'}, inplace = True)
winloss = team_stats['Total won'] / team_stats['Total Matches played']
winloss = pd.DataFrame({'Winloss Ratio': team_stats['Total won'] / team_stats['Total Matches played']})
winloss= winloss.round(2)
team_stats = team_stats.join(winloss)
team_stats


# In[49]:


plt.subplots(figsize=(10,7))
data_matches['toss_winner'].value_counts().plot.bar(width=0.8)
plt.title("Maximum Toss Won");


# In[50]:


Tosswin_matchwin=data_matches[data_matches['toss_winner']==data_matches['winner']]
slices=[len(Tosswin_matchwin),(len(data_matches)-len(Tosswin_matchwin))]
labels=['Yes','No']
plt.pie(slices,labels=labels,startangle=90,shadow=True,explode=(0,0),autopct='%1.1f%%')
plt.title("Teams who had won Toss and Won the match");


# In[51]:


delivery.toss_decision.value_counts().plot(kind='bar')
plt.xticks(rotation=0);


# In[52]:


delivery
delivery['team_toss_win']=np.where((delivery.toss_winner==delivery.winner),1,0)
plt.figure(figsize=(10,8))
sns.countplot('team_toss_win', data=delivery, hue='toss_decision')
plt.xlabel("Winning the Toss vs Winning the Match")
plt.ylabel("Frequency")
plt.title("Toss Wins vs Victory");


# Batsmen overview
# 

# In[53]:


batsmen = data_deliveries.groupby("batsman").agg({'ball': 'count','batsman_runs': 'sum'})
batsmen.rename(columns={'ball':'balls', 'batsman_runs': 'runs'}, inplace=True)
batsmen = batsmen.sort_values(['balls','runs'], ascending=False)
batsmen['batting_strike_rate'] = batsmen['runs']/batsmen['balls'] * 100
batsmen['batting_strike_rate'] = batsmen['batting_strike_rate'].round(2)
batsmen.head(10)


# In[54]:


#utility function used later
def trybuild(lookuplist, buildlist):
    alist = []
    for i in buildlist.index:
        try:
            #print(i)
            alist.append(lookuplist[i])
            #print(alist)
        except KeyError:
            #print('except')
            alist.append(0)
    return alist


# In[55]:


TopBatsman = batsmen.sort_values(['balls','runs'], ascending=False)[:20]
TopBatsman


# In[56]:


alist = []
for r in data_deliveries.batsman_runs.unique():
    lookuplist = data_deliveries[data_deliveries.batsman_runs == r].groupby('batsman')['batsman'].count()
    batsmen[str(r) + 's'] = trybuild(lookuplist, batsmen)
    try:
        alist.append(lookuplist[r])
    except KeyError:
        alist.append(0)
TopBatsman = batsmen.sort_values(['balls','runs'], ascending=False)[:20]
TopBatsman.head(10)


# In[57]:


#Build a dictionary of Matches player by each batsman
played = {}
def BuildPlayedDict(x):
    #print(x.shape, x.shape[0], x.shape[1])
    for p in x.batsman.unique():
        if p in played:
            played[p] += 1
        else:
            played[p] = 1

data_deliveries.groupby('match_id').apply(BuildPlayedDict)
import operator


# In[58]:


TopBatsman['matches_played'] = [played[p] for p in TopBatsman.index]
TopBatsman['average']= TopBatsman['runs']/TopBatsman['matches_played']

TopBatsman['6s/match'] = TopBatsman['6s']/TopBatsman['matches_played']  
TopBatsman['6s/match'].median()

TopBatsman['4s/match'] = TopBatsman['4s']/TopBatsman['matches_played']  
TopBatsman['4s/match']
TopBatsman.head()


# In[59]:


plt.figure(figsize=(10,8))
plt.bar(np.arange(len(TopBatsman)),TopBatsman['runs'])
plt.xticks(ticks=np.arange(len(TopBatsman)),labels=TopBatsman.index,rotation=90)
plt.xlabel('Batsmen')
plt.ylabel('Runs')
plt.title('Total Runs')
plt.show()


# In[60]:


plt.figure(figsize=(10,8))
plt.bar(np.arange(len(TopBatsman)),TopBatsman['batting_strike_rate'])
plt.xticks(ticks=np.arange(len(TopBatsman)),labels=TopBatsman.index,rotation=90)
plt.xlabel('Batsmen')
plt.ylabel('Strike Rate')
plt.title('Batsmen Strike Rate')
plt.show()


# In[61]:


data_deliveries.groupby('batsman')['batsman_runs'].agg("sum").sort_values(ascending= False).head().plot(kind='bar', color='Green')
plt.title("Top 5 Batsmen");


# Bowler information
# 

# In[62]:


bowler_wickets = data_deliveries.groupby('bowler').aggregate({'ball': 'count', 'total_runs': 'sum', 'player_dismissed' : 'count'})
bowler_wickets.columns = ['runs','balls','wickets']
TopBowlers = bowler_wickets.sort_values(['wickets'], ascending=False)[:20]
TopBowlers


# In[63]:


TopBowlers['economy'] = TopBowlers['runs']/(TopBowlers['balls']/6)
TopBowlers = TopBowlers.sort_values(['economy'], ascending=True)[:20]
TopBowlers


# In[64]:


plt.figure(figsize=(10,5))
plt.bar(np.arange(len(TopBowlers)),TopBowlers['economy'],color='y')
plt.xticks(ticks=np.arange(len(TopBowlers)),labels=TopBowlers.index,rotation=90)
plt.xlabel('Bowler')
plt.ylabel('economy')
plt.title('Bowlers Economy')
plt.show()


# In[65]:


plt.figure(figsize=(10,8))
plt.bar(np.arange(len(TopBowlers)),TopBowlers['wickets'],color='GREEN')
plt.xticks(ticks=np.arange(len(TopBowlers)),labels=TopBowlers.index,rotation=90)
plt.xlabel('Bowler')
plt.ylabel('wickets')
plt.title('Bowlers Wickets')
plt.show()


# In[66]:


data_deliveries.groupby('bowler')['player_dismissed'].count().sort_values(ascending=False).head(5).plot(kind='bar', color='r')
plt.title("Top 5 Bowlers")


# In[ ]:




