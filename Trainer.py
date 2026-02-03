import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from joblib import dump
from Model import Model, tokenizer

class Trainer:
    def __init__(self, df:pd.DataFrame=None, x_label:str="Token", y_label:str="Section", train_size:int=-1, test_size:float=0.3, stratify:bool=True, lemma:bool=True, tfidf:bool=True, model:str="mnb", ngram:set=(1,1)):
        self._df = df
        self.__original_training_size = train_size
        self._train_size = int(len(df)/train_size) if train_size>1 else 1
        self._test_size = test_size
        self._startify = stratify
        self._lemma = lemma
        self._tfidf = tfidf
        self._model = model
        self._trained_model = None
        self._ngram = ngram
        self._x_data = df[x_label] 
        self._y_data = df[y_label]
    
    def __prepare_training_data(self):
        if self._lemma: self._x_data = [[token[1] for token in text] for text in self._x_data]
        else: self._x_data = [[token[0] for token in text] for text in self._x_data]
        self._x_data = self._x_data[::self._train_size]
        self._y_data = self._y_data[::self._train_size]
        if self._startify: self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._x_data , self._y_data, test_size=self._test_size, random_state=42, stratify=self._y_data)
        else: self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._x_data , self._y_data, test_size=self._test_size, random_state=42)
    
    def __vectorize_data(self):
        if self._tfidf: 
            self.__vectorizer = TfidfVectorizer(tokenizer=tokenizer, preprocessor=tokenizer, token_pattern=None, lowercase=False, ngram_range=self._ngram)
            self._X_train = self.__vectorizer.fit_transform(self._X_train)
            self._X_test = self.__vectorizer.transform(self._X_test)
        
        else:
            self.__vectorizer = CountVectorizer(tokenizer=tokenizer, preprocessor=tokenizer, token_pattern=None, stop_words=[], lowercase=False, ngram_range=self._ngram)
            self._X_train = self.__vectorizer.fit_transform(self._X_train)
            self._X_test = self.__vectorizer.transform(self._X_test)
    
    def __train(self):
        if self._model == "mnb":
            self._trained_model = MultinomialNB(alpha = 0.01, fit_prior = True)
            self._trained_model.fit(self._X_train, self._y_train)

        elif self._model == "lgr":
            self._trained_model = LogisticRegression(max_iter=1000, random_state=42, C=100, l1_ratio=0)
            self._trained_model.fit(self._X_train, self._y_train)

    def __search_best_model(self):
            if self._model == "mnb":
                self._searching_model = MultinomialNB()
                self.param_grid = {'alpha': [0.01, 0.1, 0.5, 1.0, 10.0], 'fit_prior': [True, False]}
                grid_nb = GridSearchCV(self._searching_model, self.param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                print("Inizio GridSearch per Naive Bayes...")
                grid_nb.fit(self._X_train, self._y_train)
                print(f"Miglior Naive Bayes: {grid_nb.best_params_}")
                print(f"Accuracy Naive Bayes: {grid_nb.best_score_:.4f}")
            
            if self._model == "lgr":
                self._searching_model = LogisticRegression(solver='lbfgs', max_iter=1000) 
                self.param_grid = {'C': [0.1, 1, 10, 100], 'penalty': ['l1', 'l2'], 'class_weight': [None, 'balanced']}
                grid_lr = GridSearchCV(self._searching_model, self.param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                print("Inizio GridSearch per Logistic Regression...")
                grid_lr.fit(self._X_train, self._y_train)
                print(f"Miglior LogReg: {grid_lr.best_params_}")
                print(f"Accuracy LogReg: {grid_lr.best_score_:.4f}")
                self._best_parameter = grid_lr.best_params_

    def __test(self):
        self._y_predicted  = self._trained_model.predict(self._X_test)

    def train(self):
        print("Preparing data...")
        self.__prepare_training_data()
        print("Vectorizing data...")
        self.__vectorize_data()
        print("Training...")
        self.__train()
        print("Testing...")
        self.__test()

    def best_model(self):
        self.__prepare_training_data()
        self.__vectorize_data()
        self.__search_best_model()

    def display_result(self):
        print("Accuracy BoW:", accuracy_score(self._y_test, self._y_predicted))
        print(classification_report(self._y_test, self._y_predicted))
        print(confusion_matrix(self._y_test, self._y_predicted))

    def get_result(self):
        return classification_report(self._y_test, self._y_predicted, output_dict=True)

    def save_model(self, path, name:str=""):
        print("Saving model...")
        model = Model(self._trained_model, [self._X_test, self._y_test], self.__vectorizer)
        if name == "":
            model_name = self._model.upper()
            vectorizer_name = "TFIDF" if self._tfidf else "BGW"
            part_name = "LEMMA" if self._lemma else "TOKEN" 
            size_name = self.__original_training_size
            name = f"{model_name}-{vectorizer_name}-{part_name}-{size_name}"
            dump(model, f'{path}/{name}.joblib', compress=0)

        else:
            dump(model, f'{path}/{name}.joblib', compress=0)

        print(f"Model {path}/{name}.joblib saved!")

    def get_test(self):
        return self._X_test, self._y_test

if __name__ == "__main__":
    df = pd.read_parquet("DataBase/All-60000-Article-preprocessed.parquet")
    model = Trainer(df=df, tfidf=True, model="mnb", train_size=600, lemma=True)
    model.train()
    model.save_model("Model")


