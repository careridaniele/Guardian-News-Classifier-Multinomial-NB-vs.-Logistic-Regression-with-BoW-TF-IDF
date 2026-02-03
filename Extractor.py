#https://open-platform.theguardian.com/explore/
#https://www.theguardian.com/europe

#world, sport, food, music, film, artanddesign

from Modules.api_request import ApiRequest
from Modules.text_extractor import TextExtractor

api = ApiRequest(section="world", start_date="01-27-2026")
extractor = TextExtractor(api, result_number=10, request_size=10, interval_request=1)
extractor.run()
extractor.calch_meta_data()
extractor.save_parquet("Test/World", "World-10-Article")