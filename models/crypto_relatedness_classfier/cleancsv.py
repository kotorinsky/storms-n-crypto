import pandas as pd
import re

# 读取CSV文件
df = pd.read_csv('../../data/crypto-without-unknown.csv')

# 打印列名，以确认正确的列名
print("列名：", df.columns)

# 定义一个函数用于预处理文本
def preprocess_text(text):
    # 确保文本是字符串
    if pd.isnull(text):
        return ""
    # 移除URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    # 移除@提及的文本
    text = re.sub(r'@\S+', '', text)
    return text

# 使用正确的列名'text'来处理数据
df['text'] = df['text'].apply(preprocess_text)

# 显示预处理后的数据
print(df.head())

# 保存处理后的DataFrame回CSV
df.to_csv('../../data/tweets-from-influentials-process.csv', index=False)
