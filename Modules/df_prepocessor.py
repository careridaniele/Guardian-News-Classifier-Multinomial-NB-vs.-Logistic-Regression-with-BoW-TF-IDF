from Modules.text_preprocessor import Text_Preprocessor
import pandas as pd
import sys
import torch

class Dataframe_Preprocessor:

    def __init__(self, df:pd.DataFrame, text_for_process=10, model="en_core_web_trf", use_gpu:bool = False, processor_number:int=-1, batch_size:int=20):
        self.__df = df
        self.__text_for_process = text_for_process
        self.__text_number = len(self.__df)
        self.__text_results = []
        self.__token_results = []
        self.__processed_df = df
        self.__processor = Text_Preprocessor([], model=model, show_process=False, use_gpu=use_gpu, processor_number=processor_number, batch_size=batch_size)

    def run(self):
        print("Start preprocessing")
        for i in range (0, self.__text_number, self.__text_for_process):
            chunk = self.__df["Text"].iloc[i : i + self.__text_for_process].tolist()
            self.__processor.set_text(chunk)
            self.__processor.process()
            self.__text_results.extend(res[0] for res in self.__processor.get_text())
            self.__token_results.extend(res for res in self.__processor.get_token())
            percentage = min(100, ((i + self.__text_for_process) / self.__text_number) * 100)
            self.__bar_loading(percentage, "Processing")
            torch.cuda.empty_cache()
            #print(torch.cuda.memory_summary(device=None, abbreviated=False))
        self.__processed_df["Cleaned Text"] = self.__text_results
        self.__processed_df["Token"] = self.__token_results
        print("\nPreprocessing Done!")

    def get_preprocessed_df(self):
        return self.__processed_df

    def save_parquet(self, path, name):
        try:
            print("Saving...")
            pd.DataFrame(self.__processed_df).to_parquet(f"{path}/{name}-preprocessed.parquet")
            print("Save Succes")
            return True
        
        except:
            print("Save Parquet: Error during saving, maybe the Directory is not found")
            return False

    def __bar_loading(self, percentage, action):
        bar = "█" * int(30 * percentage // 100) + "-" * (30 - int(30 * percentage // 100))
        sys.stdout.write(f"\r{action}: |{bar}| {percentage:.1f}%")
        sys.stdout.flush()

