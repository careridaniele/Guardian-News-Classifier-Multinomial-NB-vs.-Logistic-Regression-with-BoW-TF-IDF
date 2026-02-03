from Modules.df_prepocessor import Dataframe_Preprocessor
import pandas as pd

if __name__ == "__main__":
    created = True
    if not created:
        #df_art = pd.read_parquet("DataBase/Art/Art-10000-Article-metadata.parquet")
        #df_film = pd.read_parquet("DataBase/Film/Film-10000-Article-metadata.parquet")
        #df_food = pd.read_parquet("DataBase/Food/Food-10000-Article-metadata.parquet")
        #df_music = pd.read_parquet("DataBase/Music/Music-10000-Article-metadata.parquet")
        #df_sport = pd.read_parquet("DataBase/Sport/Sport-10000-Article-metadata.parquet")
        df_world = pd.read_parquet("DataBase/World/World-10000-Article-metadata.parquet")

        preprocessor = Dataframe_Preprocessor(df_world, text_for_process=1000, use_gpu=True, processor_number=1, batch_size=10)
        preprocessor.run()
        preprocessor.save_parquet(path="DataBase/World", name="World-10000-Article")

    else:
        df_art = pd.read_parquet("DataBase/Art/Art-10000-Article-preprocessed.parquet")
        df_film = pd.read_parquet("DataBase/Film/Film-10000-Article-preprocessed.parquet")
        df_food = pd.read_parquet("DataBase/Food/Food-10000-Article-preprocessed.parquet")
        df_music = pd.read_parquet("DataBase/Music/Music-10000-Article-preprocessed.parquet")
        df_sport = pd.read_parquet("DataBase/Sport/Sport-10000-Article-preprocessed.parquet")
        df_world = pd.read_parquet("DataBase/World/World-10000-Article-preprocessed.parquet")
        df_list = [df_art, df_film, df_food, df_music, df_sport, df_world]
        df_total = pd.concat(df_list)
        df_total.to_parquet("DataBase/All-100000-Article-preprocessed.parquet")
