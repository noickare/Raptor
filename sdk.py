from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tiktoken
import umap
from bs4 import BeautifulSoup as Soup
from flask import Flask, jsonify, request
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_hub import hub
from numpy.typing import NDArray
from sklearn.mixture import GaussianMixture

# prep


class Preparaton:
    def __init__(self, urls_with_depths: list[tuple[str, int]]):
        self.urls_with_depths = urls_with_depths

    def num_tokens_from_string(self, string: str, encoding_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def get_document(self, url: str, max_depth: int) -> list:
        loader = RecursiveUrlLoader(
            url=url,
            max_depth=max_depth,
            extractor=lambda x: Soup(x, "html.parser").text,
        )
        docs = loader.load()
        return docs

    def get_docs(self) -> list:
        docs = []

        for url, depth in self.urls_with_depths:
            fetched_docs = self.get_document(url, depth)
            docs.extend(fetched_docs)

        return docs

    def get_concatenated_content(self):
        docs = self.get_docs()
        docs_texts = [d.page_content for d in docs]
        counts = [self.num_tokens_from_string(d, "cl100k_base") for d in docs_texts]
        d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
        d_reversed = list(reversed(d_sorted))
        concatenated_content = "\n\n\n --- \n\n\n".join(
            [doc.page_content for doc in d_reversed]
        )
        return concatenated_content

    def text_split(self):
        chunk_size_tok = 2000
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size_tok, chunk_overlap=0
        )

        concatenated_content = self.get_concatenated_content()

        texts_split = text_splitter.split_text(concatenated_content)

        return texts_split

    def global_cluster_embeddings(
        self,
        embeddings: np.ndarray,
        dim: int,
        n_neighbors: Optional[int] = None,
        metric: str = "cosine",
    ) -> np.ndarray:
        if n_neighbors is None:
            n_neighbors = int((len(embeddings) - 1) ** 0.5)
        result = umap.UMAP(
            n_neighbors=n_neighbors, n_components=dim, metric=metric
        ).fit_transform(embeddings)

        return result

    def local_cluster_embeddings(
        self,
        embeddings: np.ndarray,
        dim: int,
        num_neighbors: int = 10,
        metric: str = "cosine",
    ) -> np.ndarray:
        return umap.UMAP(
            n_neighbors=num_neighbors, n_components=dim, metric=metric
        ).fit_transform(embeddings)

    def get_optimal_clusters(
        self, embeddings: np.ndarray, max_clusters: int = 50, random_state: int = 224
    ) -> int:
        max_clusters = min(max_clusters, len(embeddings))
        n_clusters = np.arange(1, max_clusters)
        bics = []
        for n in n_clusters:
            gm = GaussianMixture(n_components=n, random_state=random_state)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
        return n_clusters[np.argmin(bics)]

    def GMM_cluster(
        self, embeddings: np.ndarray, threshold: float, random_state: int = 0
    ):
        n_clusters = self.get_optimal_clusters(embeddings)
        gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
        gm.fit(embeddings)
        probs = gm.predict_proba(embeddings)
        labels = [np.where(prob > threshold)[0] for prob in probs]
        return labels, n_clusters

    def perform_clustering(
        self,
        embeddings: np.ndarray,
        dim: int,
        threshold: float,
    ) -> List[np.ndarray]:
        if len(embeddings) <= dim + 1:
            # Avoid clustering when there's insufficient data
            return [np.array([0]) for _ in range(len(embeddings))]

        # Global dimensionality reduction
        reduced_embeddings_global = self.global_cluster_embeddings(embeddings, dim)
        # Global clustering
        global_clusters, n_global_clusters = self.GMM_cluster(
            reduced_embeddings_global, threshold
        )

        all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
        total_clusters = 0

        # Iterate through each global cluster to perform local clustering
        for i in range(n_global_clusters):
            # Extract embeddings belonging to the current global cluster
            global_cluster_embeddings_ = embeddings[
                np.array([i in gc for gc in global_clusters])
            ]

            if len(global_cluster_embeddings_) == 0:
                continue
            if len(global_cluster_embeddings_) <= dim + 1:
                # Handle small clusters with direct assignment
                local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
                n_local_clusters = 1
            else:
                # Local dimensionality reduction and clustering
                reduced_embeddings_local = self.local_cluster_embeddings(
                    global_cluster_embeddings_, dim
                )
                local_clusters, n_local_clusters = self.GMM_cluster(
                    reduced_embeddings_local, threshold
                )

            # Assign local cluster IDs, adjusting for total clusters already processed
            for j in range(n_local_clusters):
                local_cluster_embeddings_ = global_cluster_embeddings_[
                    np.array([j in lc for lc in local_clusters])
                ]
                indices = np.where(
                    (embeddings == local_cluster_embeddings_[:, None]).all(-1)
                )[1]
                for idx in indices:
                    all_local_clusters[idx] = np.append(
                        all_local_clusters[idx], j + total_clusters
                    )

            total_clusters += n_local_clusters

        return all_local_clusters


class Embed:
    def __init__(self, texts: List[str]):
        self.texts = texts

    def get_embedding_model(self):
        # can be passed via constructor
        embd = OpenAIEmbeddings()
        return embd

    def embed(self):
        embd = self.get_embedding_model()
        text_embeddings = embd.embed_documents(self.texts)
        text_embeddings_np = np.array(text_embeddings)
        return text_embeddings_np

    def embed_cluster_texts(self, clustering_func):
        text_embeddings_np = self.embed()  # Generate embeddings
        cluster_labels = clustering_func(
            text_embeddings_np, 10, 0.1
        )  # Perform clustering on the embeddings
        df = pd.DataFrame()  # Initialize a DataFrame to store the results
        df["text"] = self.texts  # Store original texts
        df["embd"] = list(
            text_embeddings_np
        )  # Store embeddings as a list in the DataFrame
        df["cluster"] = cluster_labels  # Store cluster labels
        return df

    def fmt_txt(self, df: pd.DataFrame) -> str:
        unique_txt = df["text"].tolist()
        return "--- --- \n --- --- ".join(unique_txt)

    def embed_cluster_summarize_texts(
        self, texts: List[str], level: int, model: Any
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        # Embed and cluster the texts, resulting in a DataFrame with 'text', 'embd', and 'cluster' columns
        df_clusters = self.embed_cluster_texts(texts)

        # Prepare to expand the DataFrame for easier manipulation of clusters
        expanded_list = []

        # Expand DataFrame entries to document-cluster pairings for straightforward processing
        for index, row in df_clusters.iterrows():
            for cluster in row["cluster"]:
                expanded_list.append(
                    {"text": row["text"], "embd": row["embd"], "cluster": cluster}
                )

        # Create a new DataFrame from the expanded list
        expanded_df = pd.DataFrame(expanded_list)

        # Retrieve unique cluster identifiers for processing
        all_clusters = expanded_df["cluster"].unique()

        print(f"--Generated {len(all_clusters)} clusters--")

        # Summarization
        template = """Here is a sub-set of LangChain Expression Language doc. 
        
        LangChain Expression Language provides a way to compose chain in LangChain.
        
        Give a detailed summary of the documentation provided.
        
        Documentation:
        {context}
        """

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model | StrOutputParser()

        # Format text within each cluster for summarization
        summaries = []
        for i in all_clusters:
            df_cluster = expanded_df[expanded_df["cluster"] == i]
            formatted_txt = self.fmt_txt(df_cluster)
            summaries.append(chain.invoke({"context": formatted_txt}))

        # Create a DataFrame to store summaries with their corresponding cluster and level
        df_summary = pd.DataFrame(
            {
                "summaries": summaries,
                "level": [level] * len(summaries),
                "cluster": list(all_clusters),
            }
        )

        return df_clusters, df_summary

    def recursive_embed_cluster_summarize(
        self,
        texts: List[str],
        model: Any,
        level: int = 1,
        n_levels: int = 3,
    ) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Recursively embeds, clusters, and summarizes texts up to a specified level or until
        the number of unique clusters becomes 1, storing the results at each level.

        Parameters:
        - texts: List[str], texts to be processed.
        - level: int, current recursion level (starts at 1).
        - n_levels: int, maximum depth of recursion.

        Returns:
        - Dict[int, Tuple[pd.DataFrame, pd.DataFrame]], a dictionary where keys are the recursion
        levels and values are tuples containing the clusters DataFrame and summaries DataFrame at that level.
        """
        results = {}  # Dictionary to store results at each level

        # Perform embedding, clustering, and summarization for the current level
        df_clusters, df_summary = self.embed_cluster_summarize_texts(
            texts, level, model
        )

        # Store the results of the current level
        results[level] = (df_clusters, df_summary)

        # Determine if further recursion is possible and meaningful
        unique_clusters = df_summary["cluster"].nunique()
        if level < n_levels and unique_clusters > 1:
            # Use summaries as the input texts for the next level of recursion
            new_texts = df_summary["summaries"].tolist()
            next_level_results = self.recursive_embed_cluster_summarize(
                new_texts, level + 1, n_levels
            )

            # Merge the results from the next level into the current results dictionary
            results.update(next_level_results)

        return results


# class Retriever():
