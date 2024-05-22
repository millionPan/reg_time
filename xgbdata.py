# -*- coding: utf-8 -*-
"""
Created on Sun May 19 10:27:35 2024

@author: pan
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 22:03:17 2024

@author: pan
"""
###单股分析
import time
import akshare as ak
import datetime 
import pandas as pd 
import tushare as ts
import xgboost as xgb
import random
import numpy as np


def predicttf(symbol,startdate,enddate,model_enddate,trainr):
        # symbol='002577'
        # print('444-----------')
        # startdate='20230101'
        # enddate='20240516'
        # model_enddate='20240508'
        # trainr=0.8
        if (str(symbol)[0]=="5")|(str(symbol)[0]=="1"):
            historydata = ak.fund_etf_hist_em(symbol=symbol, period="daily", start_date=startdate, end_date=enddate, adjust="")
        else:
            historydata = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=startdate, end_date=enddate, adjust="")
        historydata.rename(columns={"收盘":"tclose","开盘":"topen","最高":"thigh","最低":"tlow"},inplace=True)
        historydata['lopen']=historydata['topen'].shift(1)
        historydata['lclose']=historydata['tclose'].shift(1)
        
        historydata['lhigh']=historydata['thigh'].shift(1)
        historydata['llow']=historydata['tlow'].shift(1)
        
        historydata.loc[historydata['涨跌幅']>=0,'rf']=1#今日涨
        historydata.loc[historydata['涨跌幅']<0,'rf']=0#今日跌
        historydata['ycol']=historydata['rf'].shift(-1)#明日涨跌
        historydata['lrf']=historydata['涨跌幅'].shift(1)#last昨日涨跌幅
        historydata['nrf']=historydata['涨跌幅'].shift(-1)#next明日涨跌幅
        
        historydata["换手率_l"]=historydata["换手率"].shift(1)#今开-昨开
        historydata["振幅_l"]=historydata["振幅"].shift(1)#今开-昨收
        
        historydata["oo"]=(historydata['topen']-historydata['lopen'])/historydata['topen']#今开-昨开
        historydata["oc"]=(historydata['topen']-historydata['lclose'])/historydata['topen']#今开-昨收
        historydata["co"]=(historydata['tclose']-historydata['lopen'])/historydata['topen']#今收-昨开
        historydata["cc"]=(historydata['tclose']-historydata['lclose'])/historydata['topen']#今收-昨收
        
        historydata["hh"]=(historydata['thigh']-historydata['lhigh'])/historydata['topen']#今高-昨高
        historydata["hl"]=(historydata['thigh']-historydata['llow'])/historydata['topen']#今高-昨低
        historydata["lh"]=(historydata['tlow']-historydata['lhigh'])/historydata['topen']#今低-昨高
        historydata["ll"]=(historydata['tlow']-historydata['llow'])/historydata['topen']#今低-昨低

        historydata['MA2'] = historydata['topen'].rolling(window=2).mean()-historydata['topen']
        historydata['MA5'] = historydata['topen'].rolling(window=5).mean()-historydata['topen']
        historydata['MA8'] = historydata['topen'].rolling(window=8).mean()-historydata['topen']
        
       
        
        historydata['MA25_diff']=historydata['MA2']-historydata['MA5']
        historydata['MA58_diff']=historydata['MA5']-historydata['MA8']
        
        #555---
        historydata["updiff"]=historydata.apply(lambda x :max(x['topen'],x['tclose'])-max(x['lopen'],x['lclose']),axis=1)
        historydata["downdiff"]=historydata.apply(lambda x :min(x['topen'],x['tclose'])-min(x['lopen'],x['lclose']),axis=1)

        historydata["tco"]=(historydata['tclose']-historydata['topen'])#/historydata['topen']#今收-今开
        historydata["lco"]=(historydata['lclose']-historydata['lopen'])#/historydata['lclose']#昨收-昨开

        historydata["date"] = pd.to_datetime(historydata["日期"])
        historydata.set_index("date", inplace=True, drop=True) # 把index设为索引
        
        #划分训练模型的数据（训练集测试集）-包含模型截止日期和专门用于预测的数据
        #type(historydata['日期'][0])
        #非etf的日期本来就是日期格式，不用转换
        #historydata['日期']=historydata['日期'].apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d').date())
        scaled_data=historydata.loc[lambda x :x['日期']<=datetime.datetime.strptime(model_enddate,'%Y%m%d').date()]
        npridict_data=historydata.loc[lambda x :x['日期']>datetime.datetime.strptime(model_enddate,'%Y%m%d').date()]
        #是否按顺序
        scaled_data=scaled_data.sample(frac=1,random_state=106).reset_index(drop=True)
        # scaled_data=historydata.copy()
        
        ycol='ycol'
        
        #pd增大测试集
        n_train=int(scaled_data.shape[0]*trainr)
        train_XGB= scaled_data.iloc[:n_train].dropna()
        test_XGB = scaled_data.iloc[n_train:].dropna()
        #pd预测最新一天
        # train_XGB= scaled_data[(pd.notna(scaled_data[ycol]))&(pd.notna(scaled_data['oo']))]
        npredict_XGB = npridict_data[pd.isna(npridict_data[ycol])]
        
        
        #close(t+1),'涨跌幅'
        xlist_four=['振幅','换手率','oo','oc','co','cc','hh','hl','lh','ll','MA25_diff','MA58_diff']
        # '振幅_l','换手率_l','oo_l','oc_l','co_l','cc_l','hh_l','hl_l','lh_l','ll_l'])#变量列
        xlist_five=['tco','lco','振幅','换手率','updiff','downdiff','hh','ll']
                    # 'tco_l','lco_l','振幅_l','换手率_l','updiff_l','downdiff_l','hh_l','ll_l']
        xlist=[xlist_four,xlist_five]
        
        #444--
        params_four = {
            'booster':'gbtree',
            'objective':'binary:logistic',  # binary:logistic此处为回归预测，这里如果改成multi:softmax 则可以进行多分类
            'gamma':0.1,
            'max_depth':5,
            'lambda':8,
            'subsample':trainr,
            'colsample_bytree':trainr,
            'min_child_weight':3,
            'verbosity':1,
            'eta':0.1,
            'seed':1000,
            'nthread':4,
        }
        
        params_five = {
            'booster':'gbtree',
            'objective':'binary:logistic',  # binary:logistic此处为回归预测，这里如果改成multi:softmax 则可以进行多分类
            'gamma':0.1,
            'max_depth':4,
            'lambda':8,
            'subsample':trainr,
            'colsample_bytree':trainr,
            'min_child_weight':3,
            'verbosity':1,
            'eta':0.1,
            'seed':1000,
            'nthread':4,}
        
        paramslist=[params_four,params_five]
        
        xgbsdata=pd.DataFrame()
        for model in range(len(paramslist)): 
            # print(model)
            #model=1
            train_XGB_X, train_XGB_Y = train_XGB[xlist[model]],train_XGB.loc[:,ycol]
            test_XGB_X, test_XGB_Y = test_XGB[xlist[model]],test_XGB.loc[:,ycol]
            
            npredict_XGB_X, npredict_XGB_Y = npredict_XGB[xlist[model]],npredict_XGB.loc[:,ycol]
            
            
            #生成数据集格式
            xgb_train = xgb.DMatrix(train_XGB_X,label = train_XGB_Y)
            # xgb_test = xgb.DMatrix(test_XGB_X,label = test_XGB_Y)
            xgb_test = xgb.DMatrix(test_XGB_X)
            
            npredict_test = xgb.DMatrix(npredict_XGB_X) 
            
            num_rounds =150
            #watchlist = [(xgb_test,'eval'),(xgb_train,'train')]
            # watchlist = [(xgb_train,'train')]
            # 
            # print(xgb.__version__)
        
        
        #xgboost模型训练
        
            # print(params)
            #params=paramslist[0]
            model_xgb = xgb.train(paramslist[model],xgb_train,num_rounds)
            
            # %matplotlib qt5
            # xgb.plot_importance(model_xgb)
            
            #对测试集进行预测
            y_pred_xgb = model_xgb.predict(xgb_test)
            
            #将测试集的预测Y转换成数据框，加上时间index
            testy=pd.DataFrame(y_pred_xgb,index=test_XGB_Y.index,columns=['y_pred_xgb_t']).astype(float)
            
            #对明日涨跌进行预测
            npredict_XGB_Y = model_xgb.predict(npredict_test)
            
            #将测试集真实Y和预测Y合并数据框并求准确率
            test_acc=pd.concat([test_XGB_Y,testy],axis=1)
            test_acc.loc[test_acc['y_pred_xgb_t']>0.5,'predict']=1
            test_acc.loc[test_acc['y_pred_xgb_t']<=0.5,'predict']=0
            # 准确率
            TP=len(test_acc.loc[(test_acc['ycol']==1)&(test_acc['predict']==1)])
            TN=len(test_acc.loc[(test_acc['ycol']==0)&(test_acc['predict']==0)])
            FP=len(test_acc.loc[(test_acc['ycol']==0)&(test_acc['predict']==1)])
            FN=len(test_acc.loc[(test_acc['ycol']==1)&(test_acc['predict']==0)])
            acc=round((TP+TN)/(TP+FP+FN+TN), 4)#accuracy
            # fpr=round((FP)/(TP+FP+FN+TN), 4)#false positive rate
            if TP==0:
                tpr=0
            else:
                tpr=round((TP)/(TP+FP), 4)#在所有预测为正类别,实际为正类别比例。
            if TN==0:
                tnr=0
            else:
                tnr=round((TN)/(TN+FN), 4)#在所有预测为负类别,实际为负类别比例。
                
            xgbdata=pd.DataFrame({
            'symbol':symbol,
            'n_pred':npredict_XGB_Y,
            'acc':acc,
            # 'fpr':fpr
            'tpr':tpr,
            'tnr':tnr
            })
            xgbsdata=pd.concat([xgbsdata,xgbdata],axis=1)
            xgbsdata['avg_range']=round(np.mean(abs(historydata['涨跌幅'])),2)
            time.sleep(random.uniform(0.5,1.5))
        return xgbsdata

