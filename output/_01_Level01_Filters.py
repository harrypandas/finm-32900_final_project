#!/usr/bin/env python
# coding: utf-8

# # Appendix B: Level 1 Filters
# 
# #### After loading the data we apply four "Level 1" filters" detailed below:
# 
# * `“Identical Except Price” Filter:` The OptionMetrics data set contain duplicate observations, defined as two or more quotes with identical option type, strike, expiration date, and price. In each such case, we eliminate all but one of the quotes.
# 
# * `“Identical Except Price” Filter:` There are a few sets of quotes with identical terms (type, strike, and maturity) but different prices. When this occurs, we
# keep the quote whose T-bill-based implied volatility is closest to that of its moneyness neighbors, and delete the others. 
# 
# * `“Bid = 0” Filter:` We remove quotes of zero for bids, thereby avoiding lowvalued options. Also, a zero bid may indicate censoring as negative bids cannot be recorded.
# 
# * `“Volume = 0” Filter:` We remove quotes of zero for volumes, thereby avoiding lowtraded options. 
# 
# 

# In[ ]:


import sys
sys.path.insert(1, './../src/')

import pandas as pd
import numpy as np
import config
from pathlib import Path 
import time 
import seaborn as sns
import matplotlib.pyplot as plt

import load_option_data_01 
import filter_option_data_01 as f1

OUTPUT_DIR = Path(config.OUTPUT_DIR)
DATA_DIR = Path(config.DATA_DIR)
WRDS_USERNAME = config.WRDS_USERNAME

START_DATE_01 =config.START_DATE_01
END_DATE_01 = config.END_DATE_01

START_DATE_02 =config.START_DATE_02
END_DATE_02 = config.END_DATE_02

NOTE_START = START_DATE_01
NOTE_END = END_DATE_01


# # Level 1 Filters: 

# ## Load Data

# In[ ]:


optm_l1_load = load_option_data_01.load_all_optm_data(data_dir=DATA_DIR,
											wrds_username=WRDS_USERNAME, 
											startDate=NOTE_START,
											endDate=NOTE_END)
optm_l1_load = f1.getSecPrice(optm_l1_load)
optm_l1_load = f1.calc_moneyness(optm_l1_load)


# In[ ]:


optm_l1_load.head()


# ## Plot Loaded Data

# In[ ]:


plt.hist(optm_l1_load['date'], bins=30)
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('Distribution of OptionMetrics Pricing Data')
plt.show()


# In[ ]:


datplot = optm_l1_load
datplot['log_iv'] = np.log(datplot['impl_volatility'])
fig, ax = plt.subplots(1,3, figsize=(12,8))
axes = ax.flatten()

axes[0].hist(datplot['impl_volatility'], bins=250, color='darkblue')
axes[0].set_xlabel('IV')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of IV')
axes[0].grid()

axes[1].hist(datplot['log_iv'], bins=250, color='grey')
axes[1].set_xlabel('log(IV)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of log(IV)')
axes[1].grid()

j = 2
axes[j].hist(datplot['mnyns'], bins=250, color='grey')
axes[j].set_xlabel('Moneyness')
axes[j].set_ylabel('Frequency')
axes[j].set_title('Distribution of Moneyness')
axes[j].grid()



plt.suptitle('Level 1 Filtered Data')
plt.tight_layout()
plt.show()


# ## Filter Duplicates

# The OptionMetrics data set contain duplicate observations,
# defined as two or more quotes with identical option type, strike, expiration date, and price. In each such case, we eliminate all but one of the quotes. 
# 
# Replicating this step we found there was only one duplicate observation on March 27, 2007.

# In[ ]:


optm_l1_id = f1.delete_identical_filter(optm_l1_load)
optm_l1_load['best_mid']= (optm_l1_load['best_bid'] + optm_l1_load['best_offer'])/2


# In[ ]:


duplicate_counts = optm_l1_load.groupby(['date', 'cp_flag', 'strike_price', 'exdate', 'best_mid']).size().reset_index(name='count')
duplicate_counts = duplicate_counts.loc[duplicate_counts['count'] > 1]

fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(x='date', y='count', hue='cp_flag', data=duplicate_counts)
ax.set_xlabel('date')
ax.set_ylabel('count')
ax.set_title('Distribution of Duplicate Observations - Identical Terms')
plt.show()


# ## Filter Identical in all but price

# The OptionMetrics data set contain duplicate observations,
# defined as two or more quotes with identical option type, strike, expiration
# date, and price. In each such case, we eliminate all but one of the quotes.

# In[ ]:


optm_l1_idxp = f1.delete_identical_but_price_filter(optm_l1_id)


# In[ ]:


duplicate_counts = optm_l1_id.groupby(['date', 'cp_flag', 'strike_price', 'exdate']).size().reset_index(name='count')
duplicate_counts = duplicate_counts.loc[duplicate_counts['count'] > 1]

fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(x='date', y='count', hue='cp_flag', data=duplicate_counts)
ax.set_xlabel('date')
ax.set_ylabel('count')
ax.set_title('Distribution of Duplicate Observations - Identical Terms Except Price')
plt.show()


# ## Filter Options with Bid = 0 

# We remove quotes of zero for bids, thereby avoiding low-valued options. Also, a zero bid may indicate censoring as negative bids
# cannot be recorded.

# In[ ]:


optm_l1_zbid = f1.delete_zero_bid_filter(optm_l1_id)


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot distribution of best_bid for optm_l1_id
axes[0].hist(optm_l1_id['best_bid'], bins=30)
axes[0].set_xlabel('Best Bid')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Best Bid - Pre-filter')

# Plot distribution of best_bid for optm_l1_zbid
axes[1].hist(optm_l1_zbid['best_bid'], bins=30, color='darkred')
axes[1].set_xlabel('Best Bid')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Best Bid - Post-filter')

plt.tight_layout()
plt.show()


# ## Filter Options with Vol = 0

# In Table B.1 the paper includes a filter to exclude options where the volume is zero. Based on the table, after applying the filter the number of options deleted is zero. From our analysis, there are over 2 million rows with a volume of zero. As a result, we decided to not apply this filter to avoid dramatically skewing our results away from the original table.

# In[ ]:


optm_l1_zvol = f1.delete_zero_volume_filter(optm_l1_zbid)


# In[ ]:


zero_vol_rows = optm_l1_zbid[optm_l1_zbid['volume'] == 0]
plt.hist(zero_vol_rows['date'], bins=30)
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('Distribution of Options with Zero Volume')
plt.show()


# ## Summarize Level 1 Filters

# After application of the level 1 filters, quotes with zero bids was the primary driver of deleted observations.

# In[ ]:


df2, df2_sum, df2_B1 = f1.appendixBfilter_level1(optm_l1_load)
df2_B1 = df2_B1.reset_index().rename(columns={'index': 'Substep'}).set_index(['Step', 'Substep']).map('{:,.0f}'.format)
df2_B1 = df2_B1.map(lambda x: '' if str(x).lower() == 'nan' else x)
df2_B1

