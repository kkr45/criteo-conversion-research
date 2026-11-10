import pandas as pd
from src.data_utils import clean_click_time_features,chronological_split,leakage_audit

def test_guard():
 d=pd.DataFrame({'Sale':[0,1], 'SalesAmountInEuro':[-1,7], 'time_delay_for_conversion':[-1,8], 'product_price':[1,2], 'click_timestamp':[10,11]})
 x=clean_click_time_features(d)
 assert list(x.columns)==['product_price']
 a=leakage_audit(d)
 assert len(a)==4

def test_temporal():
 d=pd.DataFrame({'click_timestamp':[5,1,4,2,3], 'Sale':[0,1,0,0,1]})
 a,b,c=chronological_split(d)
 assert max(a.click_timestamp) < min(b.click_timestamp)
 assert max(b.click_timestamp) < min(c.click_timestamp)
