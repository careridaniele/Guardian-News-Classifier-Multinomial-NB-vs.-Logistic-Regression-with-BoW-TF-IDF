from Modules.theguardian import theguardian_content
import numpy as np

class ApiRequest():
    """
    Class that interrogate the guardian using api to extract result.

    Parameters:
        section: The newspaper section (e.g., 'football', 'politics').
        start_date: Search start date (format YYYY-MM-DD).
        end_date: Search end date.
        author: Filter by a specific author name.
        api_key: Your API key (default is 'test').
        
    Methods:
    -------
        get_metadata()
        get_result()
        get_headers()
    
    """
    
    def __init__(self, section:str=False, start_date:str=False, end_date:str=False, author:str=False, article_number:int=10, article_page:int=1, api_key:str='test', show_request:bool = False, order: str="newest", queary:str=False):
        self.__show_request = show_request
        self.__headers = {}
        self.set_api_key(api_key)
        self.set_headers(section, start_date, end_date, author, article_number, article_page, order, queary)

    def set_headers(self, section:str=False, start_date:str=False, end_date:str=False, author:str=False, article_number:int=10, article_page:int=1, order: str="newest", queary:str=False):
        for item in [("section",section), ("start_date",start_date), ("end_date", end_date), ("author", author), ("show-fields", "all"), ("show-tags","all"), ("page-size", str(article_number)), ("page", str(article_page)), ("order-by", order), ("q", queary)]:
            if item[1]: self.__headers[item[0]] = item[1]

    def set_api_key(self, api_key:str):
        self.__api_key = api_key   

    def set_show_request(self, attr:bool):
        self.__show_request = attr

    def get_headers(self):
        return self.__headers

    def get_result(self):
        return self.__results

    def search(self):
        self.__content = theguardian_content.Content(api=self.__api_key) if not self.__headers else theguardian_content.Content(api=self.__api_key, **self.__headers)
        self.__results = np.array([text for text in self.__content.get_results(self.__content.get_content_response())])
        if self.__show_request:
            for request in self.__results:
                print(f"Getting: {request["apiUrl"]}")
  