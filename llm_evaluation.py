import os
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple

import mlflow
import pandas as pd
import json
from mlflow.exceptions import MlflowException
from openai import OpenAI
from anthropic import Anthropic
import cohere
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from dotenv import load_dotenv

load_dotenv()

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class LLMPredictor(ABC):
    @abstractmethod
    def __call__(self, messages: List[Dict[str, str]]) -> str:
        pass

class OpenAIPredictor(LLMPredictor):
    def __init__(self, api_key: str, model_name: str = "gpt-4-0125-preview"):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def __call__(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        return response.choices[0].message.content

class ClaudePredictor(LLMPredictor):
    def __init__(self, api_key: str, model_name: str = "claude-3-haiku-20240307"):
        self.client = Anthropic(api_key=api_key)
        self.model_name = model_name

    def __call__(self, messages: List[Dict[str, str]]) -> str:
        system_message = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_message = next((m["content"] for m in messages if m["role"] == "user"), "")
        
        response = self.client.messages.create(
            model=self.model_name,
            system=system_message,
            messages=[{"role": "user", "content": user_message}]
        )
        return response.content[0].text

class CoherePredictor(LLMPredictor):
    def __init__(self, api_key: str, model_name: str = "command-light"):
        self.client = cohere.Client(api_key)
        self.model_name = model_name

    def __call__(self, messages: List[Dict[str, str]]) -> str:
        system_message = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_message = next((m["content"] for m in messages if m["role"] == "user"), "")
        
        response = self.client.chat(
            message=user_message,
            preamble=system_message,
            model=self.model_name
        )
        return response.text

class MistralPredictor(LLMPredictor):
    def __init__(self, api_key: str, model_name: str = "mistral-tiny"):
        self.client = MistralClient(api_key=api_key)
        self.model_name = model_name

    def __call__(self, messages: List[Dict[str, str]]) -> str:
        chat_messages = [ChatMessage(role=m["role"], content=m["content"]) for m in messages]
        response = self.client.chat(
            model=self.model_name,
            messages=chat_messages
        )
        return response.choices[0].message.content

def get_api_keys() -> Dict[str, str]:
    """Récupère toutes les clés API depuis les variables d'environnement."""
    keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "cohere": os.getenv("COHERE_API_KEY"),
        "mistral": os.getenv("MISTRAL_API_KEY")
    }
    
    missing_keys = [name for name, key in keys.items() if not key]
    if missing_keys:
        raise ValueError(f"Clés API manquantes : {', '.join(missing_keys)}")
    
    return keys

def create_predictors(api_keys: Dict[str, str]) -> List[Tuple[str, LLMPredictor]]:
    """Crée les prédicteurs pour chaque modèle."""
    return [
        ("openai-gpt4o", OpenAIPredictor(api_keys["openai"], "gpt-4o")),
        ("claude-3-haiku", ClaudePredictor(api_keys["anthropic"], "claude-3-haiku-20240307")),
        ("cohere-command", CoherePredictor(api_keys["cohere"], "command-light")),
        ("mistral-tiny", MistralPredictor(api_keys["mistral"], "mistral-tiny"))
    ]

def create_evaluation_dataframe() -> pd.DataFrame:
    """Crée le DataFrame pour l'évaluation."""
    return pd.DataFrame({
        "inputs": [
            "Quelles sont les étapes pour obtenir un permis de construire ?",
            "Comment estimer le coût d'une rénovation complète ?",
            "Quels matériaux sont recommandés pour une isolation thermique efficace ?",
        ],
        "ground_truth": [
            "Pour obtenir un permis de construire, il faut préparer et déposer un dossier de demande à la mairie, attendre l'instruction qui dure généralement 2 à 3 mois. La décision de la mairie doit ensuite être reçue et l'autorisation affichée sur le terrain concerné.",
            "Pour estimer le coût d'une rénovation complète, il faut évaluer la surface à rénover et déterminer l'ampleur des travaux. Il est recommandé d'obtenir plusieurs devis d'artisans, calculer le coût des matériaux, et ajouter une marge de 10 à 15% pour les imprévus.",
            "Les matériaux recommandés pour une isolation thermique efficace comprennent la laine de verre, la laine de roche et la ouate de cellulose. Le polystyrène expansé ou extrudé, le polyuréthane et le liège sont également des options efficaces.",
        ]
    })

def setup_mlflow_experiment(experiment_name: str = "multi_llm_evaluation"):
    """Configure l'expérience MLflow."""
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
    except MlflowException as e:
        logger.error(f"Erreur lors de la configuration de l'expérience MLflow: {e}")
        raise

def evaluate_model(predictor: LLMPredictor, model_name: str, eval_df: pd.DataFrame):
    """Évalue un modèle spécifique et enregistre les résultats."""
    with mlflow.start_run(nested=True, run_name=model_name):
        model = mlflow.pyfunc.log_model(
            artifact_path=f"model-{model_name}",
            python_model=predictor,
            artifacts={},
            input_example=[
                {"role": "system", "content": "Répondez en deux phrases maximum."},
                {"role": "user", "content": "Exemple de question?"},
            ]
        )

        results = mlflow.evaluate(
            model.model_uri,
            eval_df,
            targets="ground_truth",
            model_type="question-answering",
            evaluators="default",
        )

        for metric_name, metric_value in results.metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            logger.info(f"{model_name} - {metric_name}: {metric_value}")

        results.tables["eval_results_table"].to_csv(f"{model_name}_eval_results.csv", index=False)

        return results.metrics

def run_model_evaluations():
    """Exécute l'évaluation pour tous les modèles."""
    try:
        api_keys = get_api_keys()
        predictors = create_predictors(api_keys)
        eval_df = create_evaluation_dataframe()
        setup_mlflow_experiment()

        all_metrics = {}
        with mlflow.start_run(run_name="multi_model_comparison"):
            for model_name, predictor in predictors:
                logger.info(f"Évaluation du modèle : {model_name}")
                try:
                    metrics = evaluate_model(predictor, model_name, eval_df)
                    all_metrics[model_name] = metrics
                except Exception as e:
                    logger.error(f"Erreur lors de l'évaluation de {model_name}: {str(e)}")

        return all_metrics

    except Exception as e:
        logger.error(f"Erreur inattendue: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        metrics = run_model_evaluations()
        logger.info("Évaluation terminée avec succès")       
        # Créer un dictionnaire pour stocker les résultats de tous les modèles
        all_results = {}
        
        for model_name, model_metrics in metrics.items():
            logger.info(f"Résultats pour {model_name}:")
            all_results[model_name] = {}
            for metric_name, value in model_metrics.items():
                logger.info(f"  {metric_name}: {value}")
                all_results[model_name][metric_name] = value
        
        # Écrire les résultats dans un fichier JSON
        with open('model_evaluation_results.json', 'w') as f:
            json.dump(all_results, f, indent=4)
        
        logger.info("Les résultats ont été enregistrés dans 'model_evaluation_results.json'")
    
    except Exception as e:
        logger.error(f"L'évaluation a échoué: {str(e)}")