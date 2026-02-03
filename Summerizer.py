import pandas as pd
import numpy as np
from itertools import islice

df_all = pd.read_parquet("Test/All-100-Article-preprocessed.parquet")

summerize_df = pd.DataFrame(columns=["Section", "Authors", "Authors number", "No Authors", "Top 5 Authors", "Key Word", "Key Word Number", "Top 5 Key Word", 
                                     "Number of Word", "Lemma Vocabulary", "Lemma number", "Top 5 lemma", "Token Vocabulary", "Token Number", "Top 5 Token", "Staring period", "Ending period"])

for category in df_all["Section"].unique():
    category_group = df_all[df_all["Section"]==category]

    #Authors
    authors_dict = {} 
    authors_list, authors_count = np.unique(np.concatenate(category_group["Authors"].to_list()), return_counts=True)
    author_number = len(authors_list)
    no_author = category_group["Authors"].apply(lambda x: len(x) == 0).sum()
    for author, count in zip(authors_list, authors_count):
        authors_dict[author] = int(count)
    authors_dict = dict(sorted(authors_dict.items(), key=lambda item: item[1], reverse=True))   
    top_5_authors = dict(islice(authors_dict.items(), 5))


    #Key Word
    keyword_dict = {} 
    keyword_list, keyword_count = np.unique(np.concatenate(category_group["Key Word"].to_list()), return_counts=True)
    keyword_number = len(keyword_list)
    for keyword, count in zip(keyword_list, keyword_count):
        keyword_dict[keyword] = int(count)
    keyword_dict = dict(sorted(keyword_dict.items(), key=lambda item: item[1], reverse=True))
    top_5_keyword = dict(islice(keyword_dict.items(), 5)) 
    
    #Word Number
    word_number = category_group["Number of Word"].sum()

    #Lemma
    all_lemma = [token_info[1] for text in category_group["Token"] for token_info in text]
    lemma_list, lemma_count = np.unique(all_lemma, return_counts=True)
    lemma_number = len(lemma_list)

    lemma_dict = {str(lemma): int(count) for lemma, count in zip(lemma_list, lemma_count)}
    lemma_dict = dict(sorted(lemma_dict.items(), key=lambda item: item[1], reverse=True))
    top_5_lemma = dict(islice(lemma_dict.items(), 5))

    #Token
    token_dict = {}
    for text in category_group["Token"]:
        for token_info in text:
            if token_info[0] in token_dict: token_dict[token_info[0]] += 1
            else: token_dict[token_info[0]] = 1

    all_token = [token_info[0] for text in category_group["Token"] for token_info in text]
    token_list = token_dict.keys()
    token_number = len(token_dict.keys())

    token_dict = dict(sorted(token_dict.items(), key=lambda item: item[1], reverse=True))
    top_5_token = dict(islice(token_dict.items(), 5))

    #Date
    date_list = category_group["Date"].tolist()
    date_min = min(date_list)
    date_max = max(date_list)

    summerize_df.loc[len(summerize_df)] = [category, authors_dict, author_number, no_author, top_5_authors, keyword_dict, keyword_number, 
                                           top_5_keyword, word_number, lemma_dict, lemma_number, top_5_lemma, token_dict, token_number, top_5_token, date_min, date_max]

summerize_df.to_csv("Test/All-100-Article-summerized.csv")
print("Done!")