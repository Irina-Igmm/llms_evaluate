import re
from pathlib import Path
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
import os
from dotenv import load_dotenv

load_dotenv()
assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY environment variable must be set"

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
    """Crée la base de connaissances initiale. À exécuter une seule fois ou lors des mises à jour."""
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
            documents=texts,
            embedding=embeddings,
            persist_directory="./data/chroma_db"
        )
        vectorstore.persist()
        return True
        
    except Exception as e:
        print(f"Erreur lors de l'initialisation: {e}")
        return False

def load_qa_system():
    """Charge le système QA à partir de la base persistante. Utiliser cette fonction pour les requêtes."""
    embeddings = OpenAIEmbeddings()
    # Charge la base existante sans recréer les embeddings
    vectorstore = Chroma(
        persist_directory="./files_test/data/chroma_db",
        embedding_function=embeddings
    )
    
    llm = ChatOpenAI(model="gpt-4-mini", temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return qa

# Script pour l'initialisation (à exécuter une seule fois)
if __name__ == "__main__":
    # Décommentez la ligne suivante uniquement pour réinitialiser la base
    # initialize_rag_system()
    
    # Utilisation normale pour les requêtes
    qa_system = load_qa_system()
    
    # Exemple de plusieurs questions sans recharger la base
    questions = [
        "Qu'est-ce que l'apprentissage supervisé ?",
        "Quelle est la différence entre l'apprentissage supervisé et non supervisé ?",
        "Qu'est-ce qu'un réseau de neurones ?"
    ]
    
    for question in questions:
        res = qa_system.run(question)
        print(f"\nQuestion: {question}")
        print(f"Réponse: {res}\n")