import pandas as pd
df=pd.read_csv('ton_iot_cleaned.csv');
print(df['src_ip'].nunique()); print(df['src_ip'].value_counts().head(15))