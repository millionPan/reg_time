# -*- coding: utf-8 -*-
"""
Created on Sat May 18 20:00:54 2024

@author: pan
"""


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
# import os
# os.chdir('d:/reg_time/reg_time')

import smtplib
from email.mime.text import MIMEText


#全局配置
st.set_page_config(
    page_title="fafafa",    #页面标题
    page_icon=":rainbow:",        #icon:emoji":rainbow:"
    layout="wide",                #页面布局
    initial_sidebar_state="auto"  #侧边栏
)
import xgbdata
import etf_xgbdata 

#行业
@st.cache_data
def industry():
    r_stock_info=pd.read_excel("./stock_info.xlsx")
    r_stock_info['symbol']=r_stock_info['ts_code'].apply(lambda x :x[:6])
    return r_stock_info
r_stock_info=industry()

if st.button('发发发', key='datab1'):
    start = time.perf_counter()
    symbollist=['159577', '561060', '588700', '510710', 
                '159889', '561550', '513730', '513630',
                '002771','603373',
                '603231', '603276', '002378', '603291', '600075', 
                '000756', '600618', '600603', '600633', '000561',
                '002245', '001332', '601929', '001227', '603520', 
                 '002073', '600993', '605060', '600206', 
                '002577', '600577', '603097', '600818', '000536'
                ]


    #enddate='20240520'
    t_today=datetime.date.today()
    enddate=t_today.strftime("%Y%m%d")
    
    xgbparams=pd.DataFrame()
    for symbol in symbollist:
        try:
            if (str(symbol)[0]=="5")|(str(symbol)[0]=="1"):
                xgbsdata=etf_xgbdata.predicttf(symbol,startdate="20230101",enddate=enddate,model_enddate='20240516',trainr=0.8)
            else:
                xgbsdata=xgbdata.predicttf(symbol,startdate="20230101",enddate=enddate,model_enddate='20240516',trainr=0.8)
            xgbparams=pd.concat([xgbparams,xgbsdata],axis=0,ignore_index=True) 
        except:
            print(symbol+'error!')
    
    xgbparams.columns=['symbol_fo','n_pred_fo','acc_fo','tpr_fo','tnr_fo','avg_range','symbol_fi','n_pred_fi','acc_fi','tpr_fi','tnr_fi']
      
    #匹配行业、名称
    realtimedata = ts.get_realtime_quotes(symbollist)[['name','code','volume']]
    predictdata=(xgbparams.merge(realtimedata,how='left',left_on='symbol_fo',right_on='code',)
                 .merge(r_stock_info[['symbol','industry']],how='left',left_on='symbol_fo',right_on='symbol'))
    
    #模型选择
    predictdata[['n_pred_fo','n_pred_fi']]=predictdata[['n_pred_fo','n_pred_fi']].apply(lambda x :round(pd.to_numeric(x),3))
    
    #pred_combination
    # predictdata.loc[(predictdata['n_pred_fo']<=0.5)&(predictdata['n_pred_fi']<=0.5)&(predictdata['tnr_fo']>=predictdata['tnr_fi']),'pred_co']=predictdata['n_pred_fo'] 
    # predictdata.loc[(predictdata['n_pred_fo']<=0.5)&(predictdata['n_pred_fi']<=0.5)&(predictdata['tnr_fo']<predictdata['tnr_fi']),'pred_co']=predictdata['n_pred_fi'] 
    
    # predictdata.loc[(predictdata['n_pred_fo']>0.5)&(predictdata['n_pred_fi']>0.5)&(predictdata['tpr_fo']>=predictdata['tpr_fi']),'pred_co']=predictdata['n_pred_fo'] 
    # predictdata.loc[(predictdata['n_pred_fo']>0.5)&(predictdata['n_pred_fi']>0.5)&(predictdata['tpr_fo']<predictdata['tpr_fi']),'pred_co']=predictdata['n_pred_fi'] 
    
    # predictdata.loc[(predictdata['n_pred_fo']<=0.5)&(predictdata['n_pred_fi']>0.5)&(predictdata['tnr_fo']>=predictdata['tpr_fi']),'pred_co']=predictdata['n_pred_fo'] 
    # predictdata.loc[(predictdata['n_pred_fo']<=0.5)&(predictdata['n_pred_fi']>0.5)&(predictdata['tnr_fo']<predictdata['tpr_fi']),'pred_co']=predictdata['n_pred_fi']  
    
    # predictdata.loc[(predictdata['n_pred_fo']>0.5)&(predictdata['n_pred_fi']<=0.5)&(predictdata['tpr_fo']>=predictdata['tnr_fi']),'pred_co']=predictdata['n_pred_fo']  
    # predictdata.loc[(predictdata['n_pred_fo']>0.5)&(predictdata['n_pred_fi']<=0.5)&(predictdata['tpr_fo']<predictdata['tnr_fi']),'pred_co']=predictdata['n_pred_fi']    
    #pred_singlemodel   
    predictdata['pred_sm']=predictdata.apply(lambda x :x['n_pred_fo'] if x['acc_fo']>=x['acc_fi'] else x['n_pred_fi'],axis=1)
    
   
    predictdata['acc']=predictdata.apply(lambda x :x['acc_fo'] if x['acc_fo']>=x['acc_fi'] else x['acc_fi'],axis=1)  
    predictdata['tpr']=predictdata.apply(lambda x :x['tpr_fo'] if x['acc_fo']>=x['acc_fi'] else x['tpr_fi'],axis=1)
    predictdata['tnr']=predictdata.apply(lambda x :x['tnr_fo'] if x['acc_fo']>=x['acc_fi'] else x['tnr_fi'],axis=1)

    predictdata[['pred_sm']]=predictdata[['pred_sm']].apply(lambda x :round(pd.to_numeric(x),3))
    
    # predictdata['tmr_co']=predictdata.apply(lambda x :1 if x['pred_co']>0.5 else 0,axis=1)
    predictdata['tmr_sm']=predictdata.apply(lambda x :1 if x['pred_sm']>0.5 else 0,axis=1)
    
    # predictdata=(predictdata.merge(gc[['symbol','rank','acc','model']],how='left',left_on='symbol_fo',right_on='symbol') 
    #               )
    predictdata_show=(predictdata[['symbol_fo','name','tmr_sm','pred_sm','tpr','tnr','acc','avg_range','industry','acc_fo','tpr_fo','tnr_fo','n_pred_fo','acc_fi','tpr_fi','tnr_fi','n_pred_fi']]
                 .sort_values(by=['pred_sm'],ascending=False))
    
    predictdata_email=(predictdata[['symbol_fo','name','tmr_sm','pred_sm','tpr','tnr','acc','avg_range','industry']]
                 .sort_values(by=['pred_sm'],ascending=False))
    
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
 