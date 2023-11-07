from llama_index.node_parser import SimpleNodeParser
from llama_index import Document


class DocumentCreator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.parser = SimpleNodeParser.from_defaults(
            include_metadata=True, chunk_size=512, chunk_overlap=0
        )

    def create_documents_by_column(self, df, column):
        documents = []
        metadata_cols = self.cfg.indices.index.collections.metadata_columns

        for _, row in df.iterrows():
            doc = row[column]
            meta = {val: row[val] for val in metadata_cols}
            documents.append(
                Document(
                    text=doc,
                    metadata=meta,
                    excluded_embed_metadata_keys=list(meta.keys()),
                    excluded_llm_metadata_keys=list(meta.keys()),
                    text_template="{content}",
                )
            )
        return documents

    def create_nodes_from_documents(self, documents):
        nodes = self.parser.get_nodes_from_documents(documents, show_progress=True)
        for n in nodes:
            n.id_ = n.metadata[self.cfg.indices.index.collections.embed.id_column]
        return nodes
