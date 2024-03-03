import pandas as pd
from pandas.testing import assert_frame_equal
import load_option_data_01 as l1
import filter_option_data_01 as f1
import config
DATA_DIR = config.DATA_DIR

TEST_START = '1996-01-01'
TEST_END = '2012-01-31'

df_test = l1.load_all_optm_data(DATA_DIR, startDate = TEST_START, endDate = TEST_END )
df_test = f1.getSecPrice(df_test)
df_test = f1.calc_moneyness(df_test)

df_test_ID = f1.delete_identical_filter(df_test)

df_test_ID_But_Price = f1.delete_identical_but_price_filter(df_test_ID)

df_test_Zero_Bid = f1.delete_zero_bid_filter(df_test_ID_But_Price)

df_test_Zero_Vol = f1.delete_zero_volume_filter(df_test_Zero_Bid)

ourValues =  {'id' : 0, "except_price": 10, "bid": 272078, "vol": 2093744, "all": 1044748}
paperValues = {'id' : 1, "except_price": 11, "bid": 272048, "vol": 0, "all": 3138336}

testdict = ourValues

def test_identical_filter():
	""" Test the identical filter
	"""
	d =df_test.shape[0]- df_test_ID.shape[0]
	assert d == testdict['id']

def test_identical_but_price_filter():
	""" Test the identical but price filter
	"""
	d = (df_test_ID.shape[0]- df_test_ID_But_Price.shape[0])
	assert  d == testdict['except_price']

def test_zero_bid_filter(): 
	""" Test the zero bid filter
	"""
	d = (df_test_ID_But_Price.shape[0]- df_test_Zero_Bid.shape[0])
	assert d == testdict['bid']

def test_zero_vol_filter():
	""" Test the zero volume filter
	"""
	d = (df_test_Zero_Bid.shape[0]- df_test_Zero_Vol.shape[0])
	assert  d == testdict['vol']

def test_post_level1(): 
	""" Test the final number of rows
	"""
	d = df_test_Zero_Vol.shape[0] 
	assert d == testdict['all']
