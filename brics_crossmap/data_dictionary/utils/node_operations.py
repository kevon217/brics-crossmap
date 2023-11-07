from typing import List, Optional
import pandas as pd

from llama_index import QueryBundle
from llama_index.indices.postprocessor.types import BaseNodePostprocessor
from llama_index.schema import NodeWithScore


class DummyNodePostprocessor:
    def postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:
        # subtracts 1 from the score
        for n in nodes:
            n.score = 1 - n.score

        return nodes


def node_results_to_dataframe(results):
    source_nodes = results.source_nodes
    results = []
    for node in source_nodes:
        result_dict = {}
        node_dict = node.dict()
        node_score = {"score": node_dict["score"]}
        node_info = node_dict["node"]["relationships"]["1"]
        node_id = {"node_id": node_info["node_id"]}
        node_meta = node_info["metadata"]
        result_dict = {**node_id, **node_meta, **node_score}
        results.append(result_dict)
    df_results = pd.DataFrame(results)
    return df_results
