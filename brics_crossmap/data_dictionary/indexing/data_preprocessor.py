import pandas as pd


class DataPreprocessor:
    def __init__(self, cfg):
        self.cfg = cfg

    def preprocess_data(self, df):
        # Clean the DataFrame based on config settings
        return df.dropna(subset=[self.cfg.indices.index.collections.embed.id_column])
