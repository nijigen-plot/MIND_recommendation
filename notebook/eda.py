# %%
import ast

import pandas as pd

# %%
news_columns = [
    'news_id',
    'category',
    'subcategory',
    'title',
    'abstract',
    'url',
    'title_entities',
    'abstract_entities'
]
df = pd.read_csv('../data/MINDlarge_train/news.tsv', sep='\t', names=news_columns, header=None)

# %%
df


# %%
# カテゴリは１８種類ある。「この１８種類をあてる」ということをk-NNでやればいいのでは
df.category.unique()
# %%
# サブカテゴリは285種類
df.groupby(['subcategory']).size().sort_values()

# %%
# 一部サブカテゴリの名前はカテゴリ間で重複する
# とりあえずカテゴリをあてる場合、サブカテゴリはリークになるので使わない
df.groupby(['category','subcategory']).size().reset_index().drop_duplicates(['subcategory'])
# %%
df.iloc[0, 6]
# %%
# 要約(abstract)は使えそうと思ったが、若干入ってないのもある。
df.isnull().sum()

# %%
# entitiesのlistにNaNが入っている場合がある。からのListで補完
df['title_entities'] = df['title_entities'].apply(lambda x : [] if pd.isna(x) else ast.literal_eval(x))
# %%
df['abstract_entities'] = df['abstract_entities'].apply(lambda x : [] if pd.isna(x) else ast.literal_eval(x))
# %%
df.dropna()
# %%
import ast

ast.literal_eval(df.iloc[0, 6])
# %%
# %%
test = df['title_entities'].tolist()
for i in test:
    try:
        len(i)
    except:
        print(i)
        break
