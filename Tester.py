from Model import Model
from joblib import load
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from sklearn.metrics import classification_report


class Tester:
    def __init__(self, folder:str, custom_test:pd.DataFrame=None):
        self.__path = folder
        self.__custom_test = custom_test

    def exec(self):
        files = glob.glob(os.path.join(self.__path, "*.joblib"))
        self.result = {"model":[], "precision":[], "recall":[], "f1-score":[]}
        if type(self.__custom_test) == pd.DataFrame:
            for file_path in files:
                if "TOKEN" in file_path: text = [[token[0] for token in text] for text in self.__custom_test["Token"]]
                else: text = [[token[1] for token in text] for text in self.__custom_test["Token"]]
                model = load(file_path)
                predicted = model.use(text)
                real = self.__custom_test["Section"]
                res = classification_report(real, predicted, output_dict=True)["macro avg"]
                self.result["model"].append(file_path)
                self.result["precision"].append(res["precision"])
                self.result["recall"].append(res["recall"])
                self.result["f1-score"].append(res["f1-score"])
        
        else:
            for file_path in files:
                model = load(file_path)
                res = model.model_evaluation()["macro avg"]
                self.result["model"].append(file_path)
                self.result["precision"].append(res["precision"])
                self.result["recall"].append(res["recall"])
                self.result["f1-score"].append(res["f1-score"])
        
        self.__result_df = pd.DataFrame(self.result)
        self.__result_df['Nome Modello'] = self.__result_df['model'].apply(self.__clear_name)
        self.__result_df.set_index('Nome Modello', inplace=True)
        self.__result_df = self.__result_df.sort_values('f1-score', ascending=True)

    def __clear_name(self, path:str=""):
        parts = path.split('/')
        nome_file = parts[-1].replace('.joblib', '')
        return f"{nome_file}"

    def save_result(self, path:str="", name:str="model_evaluation"):
        ax = self.__result_df[['precision', 'recall', 'f1-score']].plot(kind='barh', figsize=(12, 18), width=0.85, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        plt.title('Classifica Modelli: Precision, Recall e F1-Score', fontsize=16)
        plt.xlabel('Score', fontsize=12)
        plt.ylabel('Modello (Dataset - Tipo)', fontsize=12)
        plt.xlim(0.85, 1.0)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.legend(title="Metriche", loc="lower right")
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)
        plt.tight_layout()
        plt.savefig(f'{path}/{name}.png', dpi=300, bbox_inches='tight')
        print("Graph Saved!")

    def display_result(self):
        print(self.__result_df)


if __name__ == "__main__":
    df = pd.read_parquet("Test/All-100-Article-preprocessed.parquet")
    tester = Tester("Model")
    tester.exec()
    tester.save_result("/home/admin/Studio/NLP/Progetto")