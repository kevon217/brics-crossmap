import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm
import numpy as np
import pickle
from sentence_transformers.cross_encoder import CrossEncoder
from scrap.setup_chromadb.fitbir.data_dictionary.utils import (
    normalize_embeddings,
    embed_texts_in_batches,
)
from brics_crossmap.utils import helper
from brics_crossmap.data_dictionary.crossmap import crossmap_logger, log, copy_log
from pathlib import Path


cfg = helper.compose_config(
    config_path="../configs/",
    config_name="config",
    overrides=[],
)


@log(msg="Running Curation")
def run_curation(cfg, **kwargs):
    # LOAD DATA DICTIONARY
    fp_cur = "C:/Users/armengolkm/Desktop/Full Pipeline Test v1.1.0/LLM_tests/DE_Step-1_Hydra-search (1)/DE_Step-1_curation_keepCol.csv"
    df_cur = pd.read_csv(fp_cur, dtype="object")

    df_cur.loc[
        df_cur["pipeline_name"]
        == "hybrid_semantic_search (custom=title_def, alpha=[1.0, 0.5, 0.0])",
        "pipeline_name",
    ] = df_cur.loc[
        df_cur["pipeline_name"]
        == "hybrid_semantic_search (custom=title_def, alpha=[1.0, 0.5, 0.0])",
        "pipeline_name_alpha",
    ]  # TODO: temp fix

    cols_include = [
        "variable name",
        "title",
        "definition",
        "search_ID",
        "pipeline_name",
        "query_term_1",
        "recCount",
        "CandidateScore",
        "data element concept identifiers",
        "data element concept names",
        "data element concept definitions",
        "overall_count",
        "average_score",
        "title_str_rank",
        "title_str_score",
        "definition_def_rank",
        "definition_def_score",
        "overall_rank",
        "keep",
    ]

    df_cur = df_cur[cols_include]
    df_cur.rename({"overall_rank": "semantic_search_rank"}, axis=1, inplace=True)
    df_cur["keep_reason"] = np.nan

    # SET OUTPUT DIRECTORY
    output_dir = helper.create_folder(Path(Path(fp_cur).parent, "curation"))

    df_cur.to_csv(Path(output_dir, "curation_semantic-search.csv"), index=False)

    # LOAD CROSS ENCODER MODEL
    model_name = "cross-encoder/stsb-distilroberta-base"
    crossencoder = CrossEncoder(model_name)

    # EMBED + SEMANTIC SEARCH + CROSS ENCODER RERANKING
    dfs_to_concat = []  # list to store processed DataFrame chunks

    # Group by variable name, pipeline name, search_ID
    groups = df_cur.groupby(["variable name", "pipeline_name", "search_ID"])
    query_col = "query_term_1"
    result_col = "data element concept names"

    for name, group in groups:
        query = group[query_col].iloc[0]
        cross_encoder_input = [[query, result] for result in group[result_col]]
        cross_encoder_score = crossencoder.predict(cross_encoder_input)
        group[f"cross_encoder_score"] = cross_encoder_score
        dfs_to_concat.append(group)  # add the processed group to the list

    # Combine all chunks into a single DataFrame
    result_df = pd.concat(dfs_to_concat)
    result_df = result_df.sort_values(
        by=["variable name", "pipeline_name", "search_ID", "cross_encoder_score"],
        ascending=[True, True, True, False],
    )
    result_df.drop_duplicates(inplace=True)
    result_df.to_csv(
        Path(output_dir, "curation_semantic-search_crossencoder.csv"), index=False
    )

    # Save the updated variables DataFrame
    df_curation.to_csv(Path(output_dir, "crossmapping.csv"), index=False)

    # SAVE CONFIG
    helper.save_config(cfg, output_dir, "config_faiss_index.yaml")

    return df_curation, cfg


if __name__ == "__main__":
    df_curation, cfg = run_curation(cfg)
