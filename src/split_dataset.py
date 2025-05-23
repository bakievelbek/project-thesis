import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Загружаем исходный индекс
df = pd.read_csv('../preprocess/dataset_index.csv')

# 2. Сначала делим на train (70%) и temp (30%)
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)

# 3. Теперь temp делим на val (15%) и test (15%)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)

# 4. Сохраняем отдельные индексы
train_df.to_csv('train_index.csv', index=False)
val_df.to_csv('val_index.csv', index=False)
test_df.to_csv('test_index.csv', index=False)

print(f"Train: {len(train_df)}\nVal: {len(val_df)}\nTest: {len(test_df)}")
