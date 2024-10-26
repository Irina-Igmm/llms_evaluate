from langchain.document_loaders import TextLoader, WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
import os
from dotenv import load_dotenv

load_dotenv()

# Run a quick validation that we have an entry for the OPEN_API_KEY within environment variables
assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY environment variable must be set"


class RAGSystem:
    def __init__(self, file_paths=None, urls=None, chunk_size=1000, chunk_overlap=0):
        self.file_paths = file_paths or []
        self.urls = urls or []
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def load_documents(self):
        documents = []

        # Charger les fichiers locaux
        for file_path in self.file_paths:
            try:
                loader = TextLoader(file_path)
                documents.extend(loader.load())
            except Exception as e:
                print(f"Erreur lors du chargement du fichier {file_path}: {e}")

        # Charger les URLs
        if self.urls:
            try:
                web_loader = WebBaseLoader(self.urls)
                documents.extend(web_loader.load())
            except Exception as e:
                print(f"Erreur lors du chargement des URLs: {e}")

        return documents

    def process_documents(self):
        # Chargement des documents
        documents = self.load_documents()

        # Découpage en chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        texts = text_splitter.split_documents(documents)

        # Création de la base de données vectorielle
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory="./data/chroma_db",
        )

        # Persister la base de données
        vectorstore.persist()
        return vectorstore

    def conversation(self, question, vectorstore=None):
        # Charger la base de données existante
        qa = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type="stuff", retriever=vectorstore.as_retriever()
        )
        result = qa.run(question)
        return result

    def query(self, question, k=4):
        # Charger la base de données existante
        vectorstore = Chroma(
            persist_directory="./data/chroma_db", embedding_function=self.embeddings
        )

        # Rechercher les documents pertinents
        docs = vectorstore.similarity_search(question, k=k)
        return docs


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration
    files = ["./inputs/knowledge_base.txt"]
    urls = [""]

    # Initialisation du système RAG
    rag = RAGSystem(file_paths=files, urls=urls, chunk_size=1000, chunk_overlap=0)

    # Construction de la base de connaissances
    vectorstore = rag.process_documents()

    # Exemple de requête
    question = "Quelle est la politique de confidentialité?"
    relevant_docs = rag.query(question)

    # Afficher les résultats
    for i, doc in enumerate(relevant_docs, 1):
        print(f"\nDocument {i}:")
        print(doc.page_content)
