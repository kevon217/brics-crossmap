"""
update_index.py

This script updates the existing semantic search indices within the ChromaDB vector store.
It identifies new variables from a data dictionary that need to be added (or upserted) and existing variables
that need to be updated to ensure the indices reflect the most current data. The script uses the LlamaIndex
framework to embed and index new documents, while existing document updates are handled directly through
the ChromaDB API, as the LlamaIndex framework does not currently support direct updates or upserts.

Prerequisites:
- An initialized and populated ChromaDB vector store.
- A CSV file containing the updated data dictionary.
- Configuration settings in a YAML file specifying the data paths, model parameters, and semantic search settings.

Parameters:
- `filepath_update`: The path to the CSV file containing the updated data dictionary information.
- `embed.columns`: The columns within the data dictionary that need to be embedded and indexed.

Functions:
- `DataPreprocessor`: Processes the raw data dictionary CSV to prepare it for indexing.
- `DocumentCreator`: Generates document objects suitable for embedding and indexing.
- `Indexer`: Orchestrates the addition of new nodes and updates existing ones within the ChromaDB vector store.

Execution:
To execute the script, run the following command:

    python update_index.py

The script will perform the following operations:
- Load and preprocess the updated data dictionary.
- Initialize the connection to the ChromaDB vector store.
- For each specified column in the data dictionary, identify new variables to be added to the vector store and existing variables that need to be updated.
- Embed and index new variables using the LlamaIndex framework.
- Directly update existing variables in the vector store using the ChromaDB API (LlamaIndex wrapper doesn't utilize upsert/update functionality).

The update process is vital to maintaining an accurate and efficient semantic search capability, ensuring that
the vector store remains synchronized with the most recent data dictionary variables.

Note: The script currently assumes that the ChromaDB update method is available and properly configured to handle
the update logic required for maintaining the vector store.
"""

import pandas as pd
import chromadb

from brics_crossmap.data_dictionary.indexing.data_preprocessor import DataPreprocessor
from brics_crossmap.data_dictionary.indexing.document_creator import DocumentCreator
from brics_crossmap.data_dictionary.indexing.indexer import Indexer
from brics_crossmap.data_dictionary.indexing.utils import CheckVectorStoreForUpdates
from brics_crossmap.utils import helper
from brics_crossmap.data_dictionary.indexing import indexing_logger, log


@log(msg="Updating Index")
def main(cfg):
    indexing_logger.info("Starting the index update process.")

    # Load and preprocess the data
    indexing_logger.info("Loading and preprocessing the data dictionary.")
    df = pd.read_csv(cfg.indices.index.filepath_update, dtype="object")
    df_clean = DataPreprocessor(cfg).preprocess_data(df)

    # Initialize the ChromaDB client
    indexing_logger.info("Initializing the ChromaDB client.")
    client = chromadb.PersistentClient(path=cfg.indices.index.storage_path_root)

    # Initialize the document creator and indexer
    document_creator = DocumentCreator(cfg)
    indexer = Indexer(cfg, client)

    # Iterate over each collection to check for updates and add new documents
    for col in cfg.indices.index.collections.embed.columns:
        indexing_logger.info(f"Processing collection: {col}")

        # Initialize the check for updates class for the current collection
        check_updates = CheckVectorStoreForUpdates(df_clean, cfg, client)
        check_updates.set_current_collection(col)

        # Get the existing metadata from the vector store for the current collection
        existing_metadata = check_updates.get_existing_metadata()

        # Generate lists of variables to add and update for the current collection
        variables_to_add = check_updates.get_list_for_add()
        variables_to_update = check_updates.get_list_for_update()

        indexing_logger.info(
            f"Found {len(variables_to_add)} variables to add to the collection '{col}': {variables_to_add}"
        )
        indexing_logger.info(
            f"Found {len(variables_to_update)} variables to update in the collection '{col}': {variables_to_update}"
        )

        # Filter the DataFrame for variables to add and create documents and nodes
        df_to_add = df_clean[
            df_clean[cfg.indices.index.collections.embed.id_column].isin(
                variables_to_add
            )
        ]
        documents_to_add = document_creator.create_documents_by_column(df_to_add, col)
        nodes_to_add = document_creator.create_nodes_from_documents(documents_to_add)
        indexer.add_nodes(nodes_to_add, collection_name=col)

        # Filter the DataFrame for variables to update and create documents and nodes
        df_to_update = df_clean[
            df_clean[cfg.indices.index.collections.embed.id_column].isin(
                variables_to_update
            )
        ]
        documents_to_update = document_creator.create_documents_by_column(
            df_to_update, col
        )
        nodes_to_update = document_creator.create_nodes_from_documents(
            documents_to_update
        )
        indexer.update_nodes(nodes_to_update, collection_name=col)

    indexing_logger.info("Index update process completed successfully.")


if __name__ == "__main__":
    # Load configuration settings
    cfg = helper.compose_config(
        config_path="../configs/",
        config_name="config",
        overrides=[],
    )
    main(cfg)
