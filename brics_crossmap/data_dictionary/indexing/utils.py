from typing import List, Optional
import pandas as pd


def batchify(data, batch_size):
    """Yield successive batch-sized chunks from data."""
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


class CheckVectorStoreForUpdates:
    def __init__(self, data_df, config, chromadb_client):
        self.data_df = data_df  # Directly using the passed DataFrame
        self.cfg = config
        self.chromadb_client = chromadb_client
        self.current_metadata = None
        self.current_collection = None

    def set_current_collection(self, collection_name):
        # Fetch existing metadata from the ChromaDB vector store
        self.current_collection = self.chromadb_client.get_collection(collection_name)
        return self.current_collection

    def get_existing_metadata(self):
        # Fetch existing metadata from the ChromaDB vector store
        self.current_metadata = self.current_collection.get(include=["metadatas"])[
            "metadatas"
        ]
        return self.current_metadata

    def get_list_for_add(self):
        # Determine which variables do not exist in ChromaDB and need to be upserted
        variable_names_df = set(
            self.data_df[self.cfg.indices.index.collections.embed.id_column].tolist()
        )
        existing_variable_names = set(
            meta["variable name"] for meta in self.current_metadata
        )
        new_variables_to_add = variable_names_df - existing_variable_names
        return list(new_variables_to_add)

    def get_list_for_update(self):
        # Determine which variables exist in ChromaDB and need to be updated
        variables_to_update = []
        chromadb_variables = {
            meta["variable name"]: meta["last change date"]
            for meta in self.current_metadata
            if "last change date" in meta and "variable name" in meta
        }

        for _, row in self.data_df.iterrows():
            variable_name = row[self.cfg.indices.index.collections.embed.id_column]
            if variable_name in chromadb_variables:
                if pd.to_datetime(row["last change date"]) > pd.to_datetime(
                    chromadb_variables[variable_name]
                ):
                    variables_to_update.append(variable_name)
        return variables_to_update
