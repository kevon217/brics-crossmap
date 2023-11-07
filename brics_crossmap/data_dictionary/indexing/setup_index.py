"""
setup_index.py

This script initializes the semantic search pipeline by creating and indexing documents
from a given data source. It reads from a CSV file, preprocesses the data, creates
documents, embeds them using a transformer model, and then persists these embeddings
into a ChromaDB vector store.

Prerequisites:
- A CSV file with the data to be indexed.
- Configurations set in a YAML file specifying model parameters, vector store settings,
  and the location of the input data.

Parameters:
- `filepath_input`: The path to the CSV file containing the data.
- `collections`: A dictionary within the config specifying the fields to be indexed.

Functions:
- `DataPreprocessor`: Cleans and prepares the data for indexing.
- `DocumentCreator`: Creates document objects from data rows.
- `Indexer`: Handles the indexing of documents into ChromaDB.

Execution:
Run the script from the command line using:

    python setup_index.py

Ensure that the configurations in your YAML file are set correctly before execution.
"""
import pandas as pd
import chromadb

from brics_crossmap.data_dictionary.indexing.data_preprocessor import DataPreprocessor
from brics_crossmap.data_dictionary.indexing.document_creator import DocumentCreator
from brics_crossmap.data_dictionary.indexing.indexer import Indexer
from brics_crossmap.utils import helper
from brics_crossmap.data_dictionary.indexing import indexing_logger, log


@log(msg="Setting Up Index")
def main(cfg):
    indexing_logger.info("Starting index setup process.")

    # Load and preprocess the data
    indexing_logger.info("Loading and preprocessing data dictionary.")
    df = pd.read_csv(cfg.indices.index.filepath_input, dtype="object")
    df_clean = DataPreprocessor(cfg).preprocess_data(df)

    # Initialize the document creator and the ChromaDB client
    document_creator = DocumentCreator(cfg)
    client = chromadb.PersistentClient(path=cfg.indices.index.storage_path_root)
    indexing_logger.info("ChromaDB client initialized.")

    indexer = Indexer(cfg, client)

    # Iterate over each specified column and create and index documents
    for col in cfg.indices.index.collections.embed.columns:
        indexing_logger.info(f"Processing column: {col}")
        documents = document_creator.create_documents_by_column(df_clean, col)
        nodes = document_creator.create_nodes_from_documents(documents)
        indexing_logger.info(f"Adding nodes to the collection: {col}")
        indexer.add_nodes(nodes, collection_name=col)

    # Save the updated configuration to file
    helper.save_config(
        cfg,
        cfg.indices.index.storage_path_root,
        "config_chromdb_llamaindex.yaml",
    )
    indexing_logger.info("Index setup completed and configuration saved.")


if __name__ == "__main__":
    # Load configuration settings from a YAML file
    indexing_logger.info("Loading configuration settings.")
    cfg = helper.compose_config(
        config_path="../configs/",
        config_name="config",
        overrides=[],
    )
    main(cfg)
