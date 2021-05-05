#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install ta


# In[1]:


import pandas as pd
import _pickle as pickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')
df = pd.read_csv('BTC-USD.csv')


# In[2]:


# Datetime conversion
df['Date'] = pd.to_datetime(df.Date)
# Setting the index
df.set_index('Date', inplace=True)

df


# In[28]:


def RSI(df, periods=14):
    """
    Calculates the Relative Strength Index
    
    **Values must be descending**
    """
    
    df = df.diff()
    
    lst = []
    
    for i in range(len(df)):
        if i < periods:
            
            # Appending NaNs for instances unable to look back on
            lst.append(np.nan)
            
        else:
            
            # Calculating the Relative Strength Index          
            avg_gain = (sum([x for x in df[i:periods+i] if x >= 0]) / periods)
            avg_loss = (sum([abs(x) for x in df[i:periods+i] if x <= 0]) / periods)


            rs = avg_gain / avg_loss

            rsi = 100 - (100 / (1 + rs))

            lst.append(round(rsi, 2))

            
    return lst


# In[30]:


df['RSI'] = RSI(df.Open)


# In[31]:


# Importing Library
import ta
# TA's RSI
df['ta_rsi'] = ta.momentum.rsi(df.Open)


# In[32]:


df


# In[40]:


plt.figure()
plt.plot(df['RSI'])
plt.plot(df['ta_rsi'])


# In[ ]:




