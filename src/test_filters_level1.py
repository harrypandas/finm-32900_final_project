import pandas as pd
from pandas.testing import assert_frame_equal
import load_option_data_01 as l1
import filter_option_data_01 as f1
import config
import datetime
import random 
DATA_DIR = config.DATA_DIR


'''
This unit test creates a sample datastructure and tests the functions in filter_option_data_01.py. 
A step by step guide is given at the end. 
'''
# TEST_START = '1996-01-01'
# TEST_END = '2012-01-31'

# df_test = l1.load_all_optm_data(DATA_DIR, startDate = TEST_START, endDate = TEST_END )
# df_test = f1.getSecPrice(df_test)
# df_test = f1.calc_moneyness(df_test)

# df_test_ID = f1.delete_identical_filter(df_test)

# df_test_ID_But_Price = f1.delete_identical_but_price_filter(df_test_ID)

# df_test_Zero_Bid = f1.delete_zero_bid_filter(df_test_ID_But_Price)

# df_test_Zero_Vol = f1.delete_zero_volume_filter(df_test_Zero_Bid)

# ourValues =  {'id' : 0, "except_price": 10, "bid": 272078, "vol": 2093744, "all": 1044748}
# paperValues = {'id' : 1, "except_price": 11, "bid": 272048, "vol": 0, "all": 3138336}

fabValues = {'id' : 1, "except_price": 1, "bid": 4, "vol": 3}#, "all": 3138336}
testdict = fabValues

columns = ['secid', 'date',  'cp_flag', 'exdate',
       'impl_volatility',  'volume',  'best_bid',
       'best_offer', 'strike_price',  'mnyns']
df = pd.DataFrame(columns = columns)


df['secid'] = [ 108105.0]*10
df['date'] = ['1996-01-05']*2 + ['1996-01-06']*4 + ['1996-01-10']*4
df['date'] = pd.to_datetime(df['date'])
df['strike_price'] = [10]*2 + [1,.8,.9,.9] + [10]*2+[4]*2
df['cp_flag'] = ['C']*7+['P']*3
df['exdate'] =  ['1996-01-11']*2 + ['1996-01-11']*7 + ['1998-01-11']
df['exdate'] = pd.to_datetime(df['exdate'])
df['best_offer'] = [1,1]+[4]*2+[5, 1,2,3,4,4]

df['mnyns'] = [1, 1] + [1, .8, .9, .9]+ [2]*4

df['impl_volatility'] = [.1, .1] + [.1, .11, .19, .1] + [.1]*4

df['volume'] = [1]*7 + [0]*3

df['best_bid'] = [0]*4 + [1]*6

df['name'] = ['i1', 'i2', 'di1', 'di2', 'di3', 'di4'] + ['vb']*4
i = f1.delete_identical_filter(df)
di = f1.delete_identical_but_price_filter(i)
zb = f1.delete_zero_bid_filter(di)
zv = f1.delete_zero_volume_filter(zb)




def test_identical_filter():
	""" Test the identical filter
	"""
	
	d =i.shape[0]- df.shape[0]
	assert -d == testdict['id']

def test_identical_but_price_filter():
	""" Test the identical but price filter
	"""
	d = (di.shape[0]- i.shape[0])
	assert  -d == testdict['except_price']

def test_zero_bid_filter(): 
	""" Test the zero bid filter
	"""
	d = (zb.shape[0]- i.shape[0])
	assert -d == testdict['bid']

def test_zero_vol_filter():
	""" Test the zero volume filter
	"""
	d = (zv.shape[0]- zb.shape[0])
	assert  -d == testdict['vol']







'''
illustrate the tricky filters: 
>df.groupby(['secid', 'cp_flag', 'date', 'exdate', 'strike_price', 'mnyns']).apply(lambda x: x[['impl_volatility', 'name']])
Starting dataframe:
                                                             impl_volatility name
secid    cp_flag date       exdate     strike_price mnyns
108105.0 C       1996-01-05 1996-01-11 10.0         1.0   0             0.10   i1
                                                          1             0.10   i2
                 1996-01-06 1996-01-11 0.8          0.8   3             0.11  di2
                                       0.9          0.9   4             0.19  di3
                                                          5             0.10  di4
                                       1.0          1.0   2             0.10  di1
                 1996-01-10 1996-01-11 10.0         2.0   6             0.10   vb
         P       1996-01-10 1996-01-11 4.0          2.0   8             0.10   vb
                                       10.0         2.0   7             0.10   vb
                            1998-01-11 4.0          2.0   9             0.10   vb
Apply remove identical filters: 
>i.groupby(['secid', 'cp_flag', 'date', 'exdate', 'strike_price', 'mnyns']).apply(lambda x: x[['impl_volatility', 'name']])
                                                             impl_volatility name
secid    cp_flag date       exdate     strike_price mnyns
108105.0 C       1996-01-05 1996-01-11 10.0         1.0   0             0.10   i1
                 1996-01-06 1996-01-11 0.8          0.8   3             0.11  di2
                                       0.9          0.9   4             0.19  di3
                                                          5             0.10  di4
                                       1.0          1.0   2             0.10  di1
                 1996-01-10 1996-01-11 10.0         2.0   6             0.10   vb
         P       1996-01-10 1996-01-11 4.0          2.0   8             0.10   vb
                                       10.0         2.0   7             0.10   vb
                            1998-01-11 4.0          2.0   9             0.10   vb

Here we see one of the first two options drop out, since they are identical in ['secid', 'cp_flag', 'strike_price','date', 'exdate', 'best_offer'].



Apply delete_identical_but_price_filter:
>di.groupby(['secid', 'cp_flag', 'date', 'exdate', 'strike_price', 'mnyns']).apply(lambda x: x[['impl_volatility', 'name']])
                                                             impl_volatility name
secid    cp_flag date       exdate     strike_price mnyns
108105.0 C       1996-01-05 1996-01-11 10.0         1.0   0             0.10   i1
                 1996-01-06 1996-01-11 0.8          0.8   2             0.11  di2
                                       0.9          0.9   3             0.10  di4
                                       1.0          1.0   1             0.10  di1
                 1996-01-10 1996-01-11 10.0         2.0   4             0.10   vb
         P       1996-01-10 1996-01-11 4.0          2.0   6             0.10   vb
                                       10.0         2.0   5             0.10   vb
                            1998-01-11 4.0          2.0   7             0.10   vb


Here we see that one of the options that have moneyness of 0.9 is kept. 
The option that is kept is the one with implied volatility closest to its in the money neighbor

Take another look at i: 

> i
      secid       date cp_flag     exdate  impl_volatility  volume  best_bid  best_offer  strike_price  mnyns name
0  108105.0 1996-01-05       C 1996-01-11             0.10       1         0           1          10.0    1.0   i1
2  108105.0 1996-01-06       C 1996-01-11             0.10       1         0           4           1.0    1.0  di1
3  108105.0 1996-01-06       C 1996-01-11             0.11       1         0           4           0.8    0.8  di2
4  108105.0 1996-01-06       C 1996-01-11             0.19       1         1           5           0.9    0.9  di3
5  108105.0 1996-01-06       C 1996-01-11             0.10       1         1           1           0.9    0.9  di4
6  108105.0 1996-01-10       C 1996-01-11             0.10       1         1           2          10.0    2.0   vb
7  108105.0 1996-01-10       P 1996-01-11             0.10       0         1           3          10.0    2.0   vb
8  108105.0 1996-01-10       P 1996-01-11             0.10       0         1           4           4.0    2.0   vb
9  108105.0 1996-01-10       P 1998-01-11             0.10       0         1           4           4.0    2.0   vb



Apply delete_zero_bid_filter:

> zb
      secid       date cp_flag     exdate  impl_volatility  volume  best_bid  best_offer  strike_price  mnyns name
3  108105.0 1996-01-06       C 1996-01-11              0.1       1         1           1           0.9    0.9  di4
4  108105.0 1996-01-10       C 1996-01-11              0.1       1         1           2          10.0    2.0   vb
5  108105.0 1996-01-10       P 1996-01-11              0.1       0         1           3          10.0    2.0   vb
6  108105.0 1996-01-10       P 1996-01-11              0.1       0         1           4           4.0    2.0   vb
7  108105.0 1996-01-10       P 1998-01-11              0.1       0         1           4           4.0    2.0   vb


Apply delete_zero_volume_filter:

> zv
      secid       date cp_flag     exdate  impl_volatility  volume  best_bid  best_offer  strike_price  mnyns name
3  108105.0 1996-01-06       C 1996-01-11              0.1       1         1           1           0.9    0.9  di4
4  108105.0 1996-01-10       C 1996-01-11              0.1       1         1           2          10.0    2.0   vb



'''

