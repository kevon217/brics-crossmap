import os
from dotenv import load_dotenv
import openai

from brics_crossmap.utils import helper
from brics_crossmap.loaders.dictionary_loader import DictionaryLoader
from brics_crossmap.document_creators.dictionary_document_creator import (
    DictionaryDocumentCreator,
)
from brics_crossmap.node_parsers.dictionary_node_parser import DictionaryNodeParser
from brics_crossmap.index_managers.dictionary_index_manager import (
    DictionaryIndexCreator,
)
from brics_crossmap.query_engines.dictionary_query_engine import DictionaryQueryEngine

# from brics_crossmap.service_contexts.factories.llm_service_factory import ServiceContextFactory


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")

cfg = helper.compose_config(
    config_path="../configs/",
    config_name="config_crossmap",
    overrides=[],
)


# # Step 1: Load Data
# dictionary_loader = DictionaryLoader(cfg.loaders.dictionary_loader)
# dictionary_loader.load_studies()
# # df_dictionary = dictionary_loader.df_dictionary

# # Step 2: Preprocess


# # Step 3: Create Documents
# dictionary_doc_creator = DictionaryDocumentCreator(cfg.document_creators.dictionary_document)
# dictionary_docs = dictionary_doc_creator.create_documents(dictionary_loader.df_dictionary)


# # Step 4: Parse Documents into Nodes
# dictionary_node_parser = DictionaryNodeParser(cfg.node_parsers.dictionary_nodes)
# dictionary_nodes = dictionary_node_parser.parse_nodes_from_documents(dictionary_docs)

# # Step 5: Create Llama Index from Nodes
# dictionary_index_creator = DictionaryIndexCreator(cfg.index_managers.dictionary_index, dictionary_nodes)
# dictionary_index = dictionary_index_creator.create_index()

# Step 6: Initialize Query Engine
engine_default = DictionaryQueryEngine.from_defaults(config=cfg)
engine_custom = DictionaryQueryEngine(config=cfg)
engine_custom.create_query_engine(top_k=10, response_mode="compact")


test_default_1 = engine_default.query(
    "Tell me which studies involve veteran populations."
)
test_default_1.source_nodes
test_custom_1 = engine_custom.query(
    "Tell me which studies involve veteran populations."
)
test_custom_1.source_nodes
engine_default.display_results(test_default_1)
engine_custom.display_results(test_custom_1)
