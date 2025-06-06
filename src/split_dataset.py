import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('../preprocess/dataset_index_tdnn_gan.csv')

train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)

val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)

train_df.to_csv('train_index.csv', index=False)
val_df.to_csv('val_index.csv', index=False)
test_df.to_csv('test_index.csv', index=False)

print(f"Train: {len(train_df)}\nVal: {len(val_df)}\nTest: {len(test_df)}")
