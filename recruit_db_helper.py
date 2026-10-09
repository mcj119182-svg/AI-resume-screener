import pymysql
import os
from dotenv import load_dotenv

# 数据库配置信息抽离出来，方便以后统一修改
load_dotenv()
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': os.getenv('DB_PASSWORD'),       # ⚠️ 检查这里是不是你的密码
    'database': 'recruit_db', # ⚠️ 检查这里是不是你实际要连的库
    'charset': 'utf8mb4'
}

def get_recruit_connection():
    """获取数据库连接的统一入口"""
    return pymysql.connect(**DB_CONFIG)