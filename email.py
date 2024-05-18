# -*- coding: utf-8 -*-
"""
Created on Sat May 18 20:00:54 2024

@author: pan
"""

import schedule
import time
import yagmail
# print(schedule.__version__)
# 连接服务器
# 用户名、授权码、服务器地址
yag_server = yagmail.SMTP(user='fszxpan@126.com', password='zxP200206822', host='smtp.126.com')

def sendemail():
    # 发送对象列表
    email_to = ['809251311@qq.com']
    email_title = '测试报告'
    email_content = """这是测试报告的具体内容\n
                        这是测试报告的具体内容\n
                        这是测试报告的具体内容\n"""
    # 附件列表
    # email_attachments = ['./attachments/report.png', ]
    
    # 发送邮件
    
    yag_server.send(to=email_to,subject= email_title, contents=email_content)
    # 关闭连接
    yag_server.close()



# 每周一早上8点执行报表生成任务
schedule.every().day.at("22:00").do(sendemail)

while True:
    schedule.run_pending()
    time.sleep(1)