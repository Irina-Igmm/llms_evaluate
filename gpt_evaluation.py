import os
import logging
from typing import Optional, List, Dict, Any

import mlflow
from openai import OpenAI
import pandas as pd
from mlflow.exceptions import MlflowException

from dotenv import load_dotenv

load_dotenv()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_api_key() -> Optional[str]:
    """Récupère la clé API OpenAI depuis les variables d'environnement."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "La clé API OpenAI n'est pas définie. Définissez la variable d'environnement OPENAI_API_KEY."
        )
    return api_key


def create_evaluation_dataframe() -> pd.DataFrame:
    """Crée le DataFrame pour l'évaluation."""
    return pd.DataFrame(
        {
            "inputs": [
                "Quelles sont les étapes pour obtenir un permis de construire ?",
                "Comment estimer le coût d'une rénovation complète ?",
                "Quels matériaux sont recommandés pour une isolation thermique efficace ?",
            ],
            "ground_truth": [
                "Pour obtenir un permis de construire, il faut préparer et déposer un dossier de demande à la mairie, attendre l'instruction qui dure généralement 2 à 3 mois. La décision de la mairie doit ensuite être reçue et l'autorisation affichée sur le terrain concerné.",
                "Pour estimer le coût d'une rénovation complète, il faut évaluer la surface à rénover et déterminer l'ampleur des travaux. Il est recommandé d'obtenir plusieurs devis d'artisans, calculer le coût des matériaux, et ajouter une marge de 10 à 15% pour les imprévus.",
                "Les matériaux recommandés pour une isolation thermique efficace comprennent la laine de verre, la laine de roche et la ouate de cellulose. Le polystyrène expansé ou extrudé, le polyuréthane et le liège sont également des options efficaces.",
            ],
        }
    )


def setup_mlflow_experiment(experiment_name: str = "openai_qa_evaluation"):
    """Configure l'expérience MLflow."""
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
    except MlflowException as e:
        logger.error(f"Erreur lors de la configuration de l'expérience MLflow: {e}")
        raise


class OpenAIPredictor:
    def __init__(self, client: OpenAI, model_name: str = "gpt-4o"):
        self.client = client
        self.model_name = model_name

    def __call__(self, messages: List[Dict[str, str]]) -> Any:
        """Appelle l'API OpenAI avec les messages donnés."""
        response = self.client.chat.completions.create(
            model=self.model_name, messages=messages
        )
        return response.choices[0].message.content


def run_model_evaluation():
    """Exécute l'évaluation du modèle."""
    try:
        api_key = get_api_key()
        client = OpenAI(api_key=api_key)
        eval_df = create_evaluation_dataframe()
        setup_mlflow_experiment()

        predictor = OpenAIPredictor(client, model_name="gpt-3.5-turbo")

        with mlflow.start_run() as run:
            system_prompt = "Répondez à la question suivante en deux phrases maximum."

            # Définition et enregistrement du modèle
            model = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=predictor,
                artifacts={},
                input_example=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Exemple de question?"},
                ],
            )

            # Évaluation du modèle
            results = mlflow.evaluate(
                model.model_uri,
                eval_df,
                targets="ground_truth",
                model_type="question-answering",
                evaluators="default",
            )

            # Enregistrement des métriques
            for metric_name, metric_value in results.metrics.items():
                mlflow.log_metric(metric_name, metric_value)
                logger.info(f"{metric_name}: {metric_value}")

            return results.metrics

    except ValueError as e:
        logger.error(f"Erreur de configuration: {e}")
        raise
    except MlflowException as e:
        logger.error(f"Erreur MLflow: {e}")
        raise
    except Exception as e:
        logger.error(f"Erreur inattendue: {e}")
        raise


if __name__ == "__main__":
    try:
        metrics = run_model_evaluation()
        logger.info("Évaluation terminée avec succès")
    except Exception as e:
        logger.error(f"L'évaluation a échoué: {e}")
