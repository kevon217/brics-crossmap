# brics_crossmap

The **brics_crossmap** tool is a Python-based utility for semantically mapping individual metadata fields of a user's variables to corresponding metadata fields in BRICS data elements. It uses language model embeddings to encode the semantics of each individually specified metadata field and facilitates the one-to-one field mapping process through a vector database search and reranking pipeline. The tool includes features for setting up an initial index with the embedded data elements and provides functionality to update this index as new data becomes available or existing data is modified.

## Table of Contents
- [Core Features](#core-features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Contact](#contact)

## Core Features:
- **Index Setup**: Initializes and builds a searchable index from a given data dictionary, preparing it for semantic search operations.
- **Semantic Search**: Utilizes embedded language models to perform semantic queries, matching variable titles with high precision. Additionally, it employs a cross-encoder reranking mechanism to refine search results and enhance match accuracy.
- **Index Updating**: Keeps the index current with a streamlined process for updating existing entries and adding new ones as the data dictionary evolves.

## Installation
To set up the BRICS Crossmap Tool on your local machine, you can use either `virtualenv` or `Poetry` for managing your Python environment and dependencies. Follow these steps:

```bash
# Clone the repository
git clone https://github.com/kevon217/brics-crossmap

# Navigate to the project directory
cd brics-crossmap
```

### Using virtualenv
```bash
# If using virtualenv (recommended for general use)
virtualenv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### Using Poetry
```bash
# If using Poetry (recommended for consistent development environments)
poetry install

# Activate the virtual environment created by Poetry
poetry shell
```

Ensure you have Python 3.8+ installed on your system, and you are using the correct version of `pip` associated with Python 3.8+ when using `virtualenv`, or the correct Python version set in `pyproject.toml` if using Poetry.

## Configuration
The configuration of the tool is managed by YAML files organized as follows:

```plaintext
configs/
├── config.yaml                      # Main configuration file for global settings
├── indices/                         # Contains configurations for indexing
│   └── fitbir/                      # Specific configuration for the FITBIR index
│       └── data_dictionary/
│           ├── llamaindex_chromadb_PubMedBERT.yaml  # Configuration for LlamaIndex with PubMedBERT embeddings
│           └── test.yaml                            # Test configurations for development purposes
└── semantic_search/
    └── crossmap_dd.yaml             # Configuration for crossmapping data dictionaries
```

Modify these files according to your requirements before running the scripts for indexing, crossmapping, or updating the vector database.


### `config.yaml`
```yaml
defaults:
  - semantic_search: crossmap_dd
  - indices: /fitbir/data_dictionary/llamaindex_chromadb_PubMedBERT.yaml # Path to index configuration
```
### `llamaindex_chromadb_PubMedBERT.yaml`
```yaml
index:
  index_id: 'fitbir_data_dictionary' # Unique identifier for the index
  summary: "FITBIR Data Dictionary Embeddings:" # Description of the index
  filepath_input: 'path/to/dataElementExport.csv' # Input CSV file for indexing
  filepath_update: 'path/to/dataElementExport_updates.csv' # path for update CSV file when running update_index.py
  storage_path_root: 'path/to/storage/fitbir/' # Root path for storage
  collections:
    embed:
      id_column: 'variable name' # Column used as unique identifier for entries
      columns: &columns # List of columns to embed
        - 'title'
        - 'definition'
      model_name: 'embedding-model-name' # Name of the embedding model
      model_kwargs: # Additional arguments for the embedding model
        batch_size: 500
        device: 'cpu'
        normalize_embeddings: True
    max_batch_size: 500 # Maximum size for processing batches (NOTE: chromadb has a batch size limit)
    distance_metric: {"hnsw:space": "cosine"} # Metric used for vector comparisons
    metadata_columns: # List of metadata columns associated with each entry
      - 'variable name'
      - 'title'
      # ... other columns ...
```
### `crossmap_dd.yaml`
```yaml
data_dictionary:
  filepath_input:  'path/to/your/input.csv' # Source CSV file for crossmapping
  directory_output: 'path/to/your/output/directory' # Directory to save output files
  embed:
    id_column: 'variable name' # Unique identifier column in the data dictionary
    columns: # Columns used for crossmapping
      - 'title'
      - 'definition'
  metadata_columns: # Metadata columns to include in the output
    - 'variable name'
    - 'title'
    - 'definition'
    # ... other columns ...

query:
  storage_path_root: 'path/to/vector/database' # Path to the vector database (chroma.sqlite3)
  queries:
    title_title: [title, title] # Mapping of queries to collection names
    definition_definition: [definition, definition]
  similarity_top_k: 10 # Number of top similar results to retrieve
  rerank:
    cross_encoder:
      model_name: "cross-encoder/stsb-distilroberta-base" # Model for reranking
      top_n: 10 # Top N results to consider during reranking
  include: ['documents','metadatas','ids'] # Additional data to include in the results
```

## Usage
After installation and proper yaml configuration, you can run the following scripts (**NOTE**: *you'll have to specify directory/file paths in yaml config files as modules currently don't support interactive directory/file selection*):
```bash
# Set up the initial index
python data_dictionary/indexing/setup_index.py

# Run a batch crossmapping
python data_dictionary/batch_crossmap_dd.py

# Update the vector database with new changes
python data_dictionary/indexing/update_index.py
```

## Contact

This pipeline was created by [Kevin Armengol](mailto:kevin.armengol@gmail.com) but any future modifications or enhancements will be performed by [Maria Bagonis](mailto:maria.bagonis@nih.gov) and [Olga Vovk](mailto:olga.vovk@nih.gov)
