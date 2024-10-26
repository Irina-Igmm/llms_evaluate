import re
from pathlib import Path
from typing import Dict, Any, List
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import Chroma
from langchain.chains.qa_with_sources.base import BaseQAWithSourcesChain
from langchain.embeddings import OpenAIEmbeddings

import os
import mlflow
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY environment variable must be set"

mlflow.set_experiment("Test Embeddings evaluation with gpt-4o-mini")


def extract_urls_from_markdown(markdown_path: str) -> list:
    """Extrait les URLs d'un fichier Markdown existant."""
    path = Path(markdown_path)
    if not path.exists():
        raise FileNotFoundError(f"Le fichier {markdown_path} n'existe pas")

    content = path.read_text(encoding="utf-8")
    patterns = [
        r"\[([^\]]+)\]\(([^)]+)\)",
        r"<([^>]+)>",
        r"(?<![\(\[])\b(https?:\/\/[^\s\)]+)\b(?![\)\]])",
    ]

    urls = []
    for pattern in patterns:
        matches = re.finditer(pattern, content)
        for match in matches:
            url = match.group(2) if len(match.groups()) > 1 else match.group(1)
            if url.startswith(("http://", "https://")):
                urls.append(url)
    return list(set(urls))


def initialize_rag_system():
    """Crée la base de connaissances initiale avec sources."""
    try:
        urls = extract_urls_from_markdown("../urls/sources.md")
        for url in urls:
            print(f"URL trouvée: {url}")
        assert urls, "Aucun URL trouvé dans le fichier sources.md"

        loader = WebBaseLoader(urls)
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(
            documents=texts, embedding=embeddings, persist_directory="./data/chroma_db"
        )
        vectorstore.persist()
        return True

    except Exception as e:
        print(f"Erreur lors de l'initialisation: {e}")
        return False


class RAGWithSources:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            persist_directory="./files_test/data/chroma_db",
            embedding_function=self.embeddings,
        )
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
        )

    def query(self, question: str) -> Dict[str, Any]:
        """
        Effectue une requête et retourne la réponse avec les documents sources.
        """
        # Obtenir la réponse et les sources via la chaîne QA
        qa_response = self.qa_chain({"question": question})

        # Rechercher les documents les plus pertinents
        relevant_docs = self.vectorstore.similarity_search(question, k=4)

        return {
            "question": question,
            "answer": qa_response["answer"],
            "sources": qa_response["sources"],
            "relevant_documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": getattr(doc, "score", None),  # Si disponible
                }
                for doc in relevant_docs
            ],
        }

    def evaluate_embeddings(self, vectorstore, eval_data):
        # Configuration du retrieveur avec paramètres optimisés
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 5,
                "fetch_k": 20,
                "maximal_marginal_relevance": True,
                "distance_metric": "cos",
            }
        )

        def retrieve_doc_ids(question: str) -> List[str]:
            try:
                docs = retriever.get_relevant_documents(question)
                # S'assurer que chaque document a une source dans ses métadonnées
                return [
                    doc.metadata.get("source", "")
                    for doc in docs
                    if "source" in doc.metadata
                ]
            except Exception as e:
                print(
                    f"Erreur lors de la récupération des documents pour la question: {question}"
                )
                print(f"Erreur: {e}")
                return []

        def retriever_model_function(question_df: pd.DataFrame) -> pd.Series:
            return question_df["question"].apply(retrieve_doc_ids)

        # Vérification et préparation des données d'évaluation
        if not isinstance(eval_data, pd.DataFrame):
            raise ValueError("eval_data doit être un DataFrame pandas")

        if "question" not in eval_data.columns or "source" not in eval_data.columns:
            raise ValueError(
                "eval_data doit contenir les colonnes 'question' et 'source'"
            )

        # S'assurer que les sources sont dans un format compatible
        eval_data["source"] = eval_data["source"].apply(
            lambda x: [x] if isinstance(x, str) else x
        )

        # Configuration de l'évaluation MLflow
        with mlflow.start_run():
            try:
                return mlflow.evaluate(
                    model=retriever_model_function,
                    data=eval_data,
                    model_type="retriever",
                    targets="source",
                    evaluators="default",  # Simplifié pour éviter les erreurs potentielles
                    extra_metrics=[
                        mlflow.metrics.precision_at_k(k=3),
                        mlflow.metrics.recall_at_k(k=3),
                        mlflow.metrics.ndcg_at_k(k=3),
                    ],
                    evaluator_config={
                        "col_mapping": {"inputs": "question", "targets": "source"},
                        "max_depth": 5,
                        "relevance_method": "binary",
                    },
                )
            except Exception as e:
                print(f"Erreur lors de l'évaluation MLflow: {e}")
                raise


def format_response(response: Dict[str, Any]) -> str:
    """Format the response for display."""

    output = [f"Question: {response['question']}"]
    output.append(f"Answer: {response['answer']}")
    output.append("Sources cited:")
    output.extend(response["sources"])

    output.append("Relevant documents used:")
    for document in response["relevant_documents"]:
        output.append(f"Document:")
        output.append(f"Content: {document['content'][:200]}...")
        output.append(f"Source: {document['metadata'].get('source', 'Not specified')}")
        if document["score"]:
            output.append(f"Relevance score: {document['score']:.4f}")
        output.append("-" * 80)

    return "\n".join(output)


def evaluate_rag_system():
    # Initialisation du système RAG
    rag_system = RAGWithSources()

    # Création des données d'évaluation
    # Utilisation des URLs depuis le fichier sources.md
    urls_list = extract_urls_from_markdown("./urls/sources.md")

    # Création du DataFrame d'évaluation avec des questions test
    eval_data = pd.DataFrame(
        {
            "question": [
                "Qu'est-ce que la surapprentissage du modèle IA et comment l'éviter ?",
                "Donne-moi un exemple de données synthétiques textuelles.",
                "Quels sont les avantages et inconvénients des réseaux de neurones récurrents ?",
            ],
            "source": urls_list,
        }
    )

    # Appel de la fonction d'évaluation en utilisant les composants de RAGWithSources
    evaluation_results = rag_system.evaluate_embeddings(
        vectorstore=rag_system.vectorstore,  # Chroma vectorstore déjà initialisé
        eval_data=eval_data,
    )

    # Affichage des résultats
    print("\nRésultats de l'évaluation:")
    eval_results_df = evaluation_results.tables["eval_results_table"]

    # Sauvegarde des résultats (optionnel)
    eval_results_df.to_csv("./outpus/evaluation_embeddings.csv", index=False)

    return evaluation_results


if __name__ == "__main__":
    # Décommentez pour réinitialiser la base
    # initialize_rag_system()

    # Création du système de requêtes
    rag_system = RAGWithSources()

    # Exemple de requêtes
    # questions = [
    #     "Comment identifier la surapprentissage du modèle IA et comment l'éviter ?",
    #     "Quelle est la discrepancere entre l'apprentissage supervisé et non supervisé ?",
    # ]

    # for question in questions:
    #     response = rag_system.query(question)
    #     print(format_response(response))
    #     print("\n" + "=" * 100 + "\n")

    results = evaluate_rag_system()
