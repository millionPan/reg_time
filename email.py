
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
# os.chdir('d:/reg_time/reg_time')
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
@st.cache_data
def industry():
    r_stock_info=pd.read_excel("./stock_info.xlsx")
    r_stock_info['symbol']=r_stock_info['ts_code'].apply(lambda x :x[:6])
    return r_stock_info
r_stock_info=industry()
#
if st.button('发发发', key='datab1'):
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
    
    # enddate='20240524'
    t_today=datetime.date.today()
    enddate=t_today.strftime("%Y%m%d")
    
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
    
    
    st.dataframe(predictdata_show)
    elapsed = (time.perf_counter() - start)
    st.write("Time used:",elapsed,"s")
    
    try:
        predictdata.to_excel("d:/predict"+t_today.strftime("%Y%m%d")+".xlsx")
    except:
        pass
    
    filename = "./hhh.xlsx"
    predictdata.to_excel(filename, index=False)
    from email.mime.multipart import MIMEMultipart
    # from email.mime.text import MIMEText
    from email.mime.application import MIMEApplication
    #发送邮件
    html_file=predictdata_email.to_html(index=False)
    
    # 创建邮件对象
    msg = MIMEMultipart()
    #邮件标题
    msg['Subject'] = '测试报告'
    # 编写HTML类型的邮件正文
    msg.attach(MIMEText(html_file, 'html')) 
    
   
    #添加附件
    part = MIMEApplication(open(filename, 'rb').read())
    part.add_header('Content-Disposition', 'attachment', filename=filename)
    msg.attach(part)
    
    # 连接发送邮件
    smtp = smtplib.SMTP()
    smtp.connect('smtp.126.com')
    smtp.login('fszxpan@126.com', 'zxP200206822')
    smtp.sendmail('fszxpan@126.com', '809251311@qq.com', msg.as_string())
    smtp.quit()
    st.success('达达达！')
 
