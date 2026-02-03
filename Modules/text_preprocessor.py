import torch as trc
from spacy.language import Language
import spacy
import re

trc.set_num_threads(1)

@Language.component("smart_entity_merger")
def __smart_entity_merger(doc):
    with doc.retokenize() as retokenizer:
        for ent in doc.ents:
            if ent.label_ in ["MONEY", "PERCENT", "QUANTITY", "ORDINAL", "CARDINAL"]:
                retokenizer.merge(ent, attrs={"LEMMA": f"_{ent.label_.lower()}_"})

            elif ent.label_ in ["DATE", "TIME"]:
                retokenizer.merge(ent, attrs={"LEMMA": "_TIME_"})

            elif ent.label_ in ["PERSON", "ORG", "GPE", "EVENT", "FAC", "WORK_OF_ART"]:
                new_lemma = ent.text.replace(" ", "_").lower()
                retokenizer.merge(ent, attrs={"LEMMA": new_lemma})
    return doc

class Text_Preprocessor:
    def __init__(self, text_list, model="en_core_web_trf", show_process:bool = False, use_gpu:bool = False, processor_number:int=-1, batch_size:int=20):
        self.__raw_text = text_list
        self.__clean_text_list = []
        self.__processed_tokens = []
        self.__processed_text = []
        self.__show_process = show_process
        self.__processor_number = processor_number
        self.__batch_size = batch_size
        if use_gpu: spacy.require_gpu()
        if self.__show_process: print(f"Loading model: {model}...")
        self.__nlp = spacy.load(model)
        if "smart_entity_merger" not in self.__nlp.pipe_names: self.__nlp.add_pipe("smart_entity_merger", after="ner")

    def _clean_structure(self):
        if self.__show_process: print("Cleaning Structure (Regex)...")
        cleaned = []
        for text in self.__raw_text:
            text = re.sub(r"http\S+|www\.\S+", "_URL_", text)
            text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "_EMAIL_", text)
            text = re.sub(r'\s+', ' ', text).strip()
            cleaned.append(text)
        self.__clean_text_list = cleaned
        if self.__show_process: print("Text Structure Cleaned!")
        return self

    def _process_spacy(self):
        if self.__show_process: print("Running Spacy Pipeline (Tokenizer -> NER -> Merger -> Lemmatizer)...")
        docs = self.__nlp.pipe(self.__clean_text_list, disable=["parser"], n_process=self.__processor_number, batch_size=self.__batch_size)
        for doc in docs:
            doc_tokens = []
            for token in doc:
                is_valid_word = token.is_alpha and not token.is_stop
                is_entity = token.ent_type_ != ""
                is_exception = token.text in ["_URL_", "_EMAIL_"]
                if is_valid_word or is_entity or is_exception: doc_tokens.append((token.lower_, token.lemma_.lower(), token.pos_, token.ent_type_))
            self.__processed_tokens.append(doc_tokens)
        if self.__show_process: print("Spacy Processing Done!")
        return self.__processed_tokens

    def _rebuild_text(self):
        if self.__show_process: print("Rebuilding Text...")
        for text in self.__processed_tokens:
            lemmas = [token[1] for token in text]
            self.__processed_text.append([" ".join(lemmas)])
        if self.__show_process: print("Text rebuilde!")
        return self

    def process(self):
        self._clean_structure()
        self._process_spacy()
        self._rebuild_text()

    def set_text(self, text_list):
        self.__raw_text = text_list
        self.__clean_text_list = []
        self.__processed_tokens = []
        self.__processed_text = []

    def get_text(self):
        return self.__processed_text

    def get_token(self):
        return self.__processed_tokens




