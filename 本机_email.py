
# conda activate streamlit_test
# cd /d d:\reg_time\reg_time
# streamlit run email.py

# print(schedule.__version__)
import streamlit as st

import datetime 
import pandas as pd 
import tushare as ts
import pandas as pd
import time
import os
os.chdir('d:/reg_time/reg_time')
import xgbdata
import etf_xgbdata 
import numpy as np

import smtplib
from email.mime.text import MIMEText


#全局配置
st.set_page_config(
    page_title="fafafa",    #页面标题
    page_icon=":rainbow:",        #icon:emoji":rainbow:"
    layout="wide",                #页面布局
    initial_sidebar_state="auto"  #侧边栏
)

#行业

def industry():
    r_stock_info=pd.read_excel("./stock_info.xlsx")
    r_stock_info['symbol']=r_stock_info['ts_code'].apply(lambda x :x[:6])
    return r_stock_info
r_stock_info=industry()
#
# if st.button('发发发', key='datab1'):
start = time.perf_counter()
symbollist=['159577', '561060', '510710', '159559', '513400', '159896', '159505', '512150',
            '561550', '513630',
            '600320', '600744', '601686', '002801', '000541', 
            '002771',
            '603231', '603276', '002378', '603291', '001366', 
            '600603',  
            '001332', '601929', '001227', 
            '600206', 
            '002577',  '600818'
            ]

enddate='20240524'
# t_today=datetime.date.today()
# enddate=t_today.strftime("%Y%m%d")

xgbparams=pd.DataFrame()
for symbol in symbollist:
    try:
        if (str(symbol)[0]=="5")|(str(symbol)[0]=="1"):
            xgbsdata=etf_xgbdata.predicttf(symbol,startdate="20230101",enddate=enddate,model_enddate='20240523',trainr=0.8)
        else:
            xgbsdata=xgbdata.predicttf(symbol,startdate="20230101",enddate=enddate,model_enddate='20240523',trainr=0.8)
        xgbparams=pd.concat([xgbparams,xgbsdata],axis=0,ignore_index=True) 
    except:
        print(symbol+'error!')

xgbparams.columns=['symbol_fo','n_pred_fo','acc_fo','tpr_fo','tnr_fo','avg_range']
  
#匹配行业、名称
realtimedata = ts.get_realtime_quotes(symbollist)[['name','code','volume']]
predictdata=(xgbparams.merge(realtimedata,how='left',left_on='symbol_fo',right_on='code',)
             .merge(r_stock_info[['symbol','industry']],how='left',left_on='symbol_fo',right_on='symbol'))

#模型选择
predictdata[['n_pred_fo']]=predictdata[['n_pred_fo']].apply(lambda x :round(pd.to_numeric(x),3))

#pred_singlemodel   
predictdata['tmr_ud_fo']=predictdata.apply(lambda x :1 if x['n_pred_fo']>0.5 else 0,axis=1)

predictdata_show=(predictdata[['symbol_fo','name','tmr_ud_fo','n_pred_fo','industry',
                               'acc_fo','tpr_fo','tnr_fo','avg_range']]
             .sort_values(by=['n_pred_fo'],ascending=False))
#predictdata_show.to_excel("d:/predict"+t_today.strftime("%Y%m%d")+".xlsx")
predictdata_email=predictdata_show.copy()
 
##验证   
import time
import akshare as ak
import datetime 
import pandas as pd 
import tushare as ts
import xgboost as xgb
import random
import numpy as np


# lastday=datetime.date(2024,5,18)
# last_predict=pd.read_excel("d:/predict"+lastday.strftime("%Y%m%d")+".xlsx")
# last_predict['symbol_fo']=last_predict['symbol_fo'].apply(lambda x :str(x).zfill(6))
# # predictdata=last_predict.copy()
##############################################

last_predict=predictdata.copy()

#1/获取需要预测日期的实际数据
symbollist=last_predict['symbol_fo'].tolist()
enddate='20240527'
realtimedata=pd.DataFrame()
for symbol in symbollist:
    if (str(symbol)[0]=="5")|(str(symbol)[0]=="1"):
        historydata = ak.fund_etf_hist_em(symbol=symbol, period="daily", start_date=enddate, end_date=enddate, adjust="")
    else:
        historydata = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=enddate, end_date=enddate, adjust="")
    historydata.rename(columns={"收盘":"price"},inplace=True)
    historydata['code']=symbol
    realtimedata=pd.concat([realtimedata,historydata],axis=0,ignore_index=True)
realtimedata['pre_close']=realtimedata['price']-realtimedata['涨跌额']
realtimedata=realtimedata[['code','price','pre_close']]


# =============================================================================
# 
# =============================================================================
#昨日预测匹配当天实时数据，对比
compared_data=last_predict.merge(realtimedata,how='left',left_on='symbol_fo',right_on='code',)
compared_data['pra_ud']=compared_data.apply(lambda x :1 if x['price']>x['pre_close'] else 0,axis=1)  #pratical_updown  

#分模型查看预测效果
compared_data['jude_fo']=compared_data.apply(lambda x :1 if x['pra_ud']==x['tmr_ud_fo'] else 0,axis=1) 

print("accpre_fo:"+str(compared_data['jude_fo'].sum()/compared_data.shape[0]))#accuracy_predict_four

#调整列顺序'acc','tpr','tnr'
bbb=compared_data.columns.to_list()
invaildcol=['symbol_fo','name','industry','pra_ud','tmr_ud_fo','jude_fo','acc_fo','tpr_fo','tnr_fo']
new_col=[x for x in bbb if x not in invaildcol]
bbb =invaildcol + new_col
compared_data=compared_data.reindex(columns=bbb).sort_values(by=['pra_ud'],ascending=False)
# compared_data.to_excel("d:/compared_data"+enddate+".xlsx")