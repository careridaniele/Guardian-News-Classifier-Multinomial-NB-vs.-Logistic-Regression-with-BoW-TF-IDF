from sklearn.metrics import classification_report

def tokenizer(token):
    return token

class Model:
    def __init__(self, model, tester, vectorizer):
        self.__model = model
        self.__tester = tester
        self.__vectorizer = vectorizer

    def get_model(self):
        return self.__model
    
    def get_tester(self):
        return self.__tester
    
    def model_evaluation(self):
        x_test = self.__tester[0]
        y_test = self.__tester[1]
        y_predicted  = self.__model.predict(x_test)
        return classification_report(y_test, y_predicted, output_dict=True)

    def use(self, token_list):
        x_data = self.__vectorizer.transform(token_list)
        y_predicted = self.__model.predict(x_data)
        return y_predicted
