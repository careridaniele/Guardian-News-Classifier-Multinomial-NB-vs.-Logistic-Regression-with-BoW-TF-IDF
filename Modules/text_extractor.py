from Modules.api_request import ApiRequest
import time, hashlib, base64, sys
import numpy as np
import pandas as pd
import pickle

class TextExtractor():
    def __init__(self, api:ApiRequest, result_number: int, request_size:int, interval_request:int=0, id_lenght=8):
        self.__data = []
        self.__new_data = []
        self.__meta_data = {"Article Title":[], "Section": [], "Authors": [], "Key Word": [], "Text": [], 
                            "Language": [], "Number of Word": [], "Url": [], "Date": [], "Article Id": []}
        self.__current_page = 0
        self.__id_lenght = id_lenght
        self.__html_article = []
        self.set_api(api)
        self.set_result_number(result_number)
        self.set_interval_request(interval_request)
        self.set_request_size(request_size)

    def set_api(self, api:ApiRequest):
        self.__api = api
        self.__current_page = 0
        return self
    
    def set_result_number(self, result_number:int):
        self.__result_number = result_number 
        return self

    def set_interval_request(self, interval_request:int):
        self.__interval_request = interval_request
        return self

    def set_request_size(self, request_size:int):
        self.__request_size = request_size 
        return self

    def run(self):
        for _ in range(self.__result_number//self.__request_size):
            self.__current_page += 1
            self.__api.set_headers(article_number=self.__request_size, article_page=self.__current_page)
            self.__api.search()
            result = self.__api.get_result()
            self.__data.extend(result)
            self.__new_data.extend(result)
            self.__bar_loading(self.__current_page/(self.__result_number/self.__request_size)*100, action="Download")
            time.sleep(self.__interval_request)
        print("\n")
        if len(self.__new_data) == self.__result_number: print("Download Success")
        else: print(f"Error during download, {self.__result_number-len(self.__new_data)} article is missing")
        return self

    def __set_id(self, string:str):
        hash_bytes = hashlib.md5(string.encode('utf-8')).digest()
        code_b64 = base64.urlsafe_b64encode(hash_bytes).decode('utf-8').rstrip('=')
        return code_b64[:self.__id_lenght]

    def calch_meta_data(self) -> dict:
        """Returning a dict whit Article Title, Section, Authors, Key Word, Text, Language, Number of Word, Url"""
        if any(self.__new_data):
            for article in self.__new_data:
                if self.__set_id(article["webTitle"]+article["webPublicationDate"]) not in self.__meta_data["Article Id"]:
                    self.__meta_data["Article Title"].append(article["webTitle"])
                    self.__meta_data["Section"].append(article["sectionName"])
                    self.__meta_data["Authors"].append([tag["webTitle"] for tag in article["tags"] if tag["type"]== "contributor"])
                    self.__meta_data["Key Word"].append([tag["webTitle"] for tag in article["tags"] if tag["type"]== "keyword"])
                    self.__meta_data["Text"].append(article["fields"]["bodyText"])
                    self.__meta_data["Language"].append(article["fields"]["lang"])
                    self.__meta_data["Number of Word"].append(int(article["fields"]["wordcount"]))
                    self.__meta_data["Url"].append(article["fields"]["shortUrl"])
                    self.__meta_data["Date"].append(article["webPublicationDate"][:10])
                    self.__meta_data["Article Id"].append(self.__set_id(article["webTitle"]+article["webPublicationDate"]))
                    self.__html_article.append({"Article Id": self.__set_id(article["webTitle"]+article["webPublicationDate"]), "Article Title":article["webTitle"], "Text":article["fields"]["body"]})
            self.__new_data = np.array([])
        return self  
    
    def get_meta_data(self):
        return self.__meta_data

    def save_parquet(self, path, name):
        try:
            print("Saving...")
            pd.DataFrame(self.__meta_data).to_parquet(f"{path}/{name}-metadata.parquet")
            pd.DataFrame(self.__html_article).to_parquet(f"{path}/{name}-html.parquet")
            with open(f'{path}/{name}-rawdata.pickle', 'wb') as file: pickle.dump(self.__data, file, protocol=pickle.HIGHEST_PROTOCOL)
            print("Save Succes")
            return True
        
        except:
            print("Save Parquet: Error during saving, maybe the Directory is not found")
            return False

    def __bar_loading(self, percentage, action):
        bar = "█" * int(30 * percentage // 100) + "-" * (30 - int(30 * percentage // 100))
        sys.stdout.write(f"\r{action}: |{bar}| {percentage:.1f}%")
        sys.stdout.flush()