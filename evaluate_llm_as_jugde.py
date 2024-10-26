import mlflow
import os

from dotenv import load_dotenv

load_dotenv()

# Run a quick validation that we have an entry for the OPEN_API_KEY within environment variables

assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY environment variable must be set"

import openai
import pandas as pd

from show_finale_metrics import create_gauge_chart


system_prompt = "Translate the following sentences into Spanish"
mlflow.set_experiment("Test Translation Experiment")
basic_translation_model = mlflow.openai.log_model(
    model="gpt-4o-mini",
    task=openai.chat.completions,
    artifact_path="model",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "{user_input}"},
    ],
)

model = mlflow.pyfunc.load_model(basic_translation_model.model_uri)

res = model.predict("Hello, how are you?")
print(f"Output :{res}")

eval_data = pd.DataFrame(
    {
        "llm_inputs": [
            "I'm over the moon about the news!",
            "Spill the beans.",
            "Bite the bullet.",
            "Better late than never.",

        ]
    }
)

# Define the custom metric
cultural_sensitivity = mlflow.metrics.genai.make_genai_metric(
    name="cultural_sensitivity",
    definition="Assesses how well the translation preserves cultural nuances and idioms.",
    grading_prompt="Score from 1-5, where 1 is culturally insensitive and 5 is highly culturally aware.",
    examples=[
        mlflow.metrics.genai.EvaluationExample(
            input="Break a leg!",
            output="¡Rómpete una pierna!",
            score=2,
            justification="This is a literal translation that doesn't capture the idiomatic meaning."
        ),
        mlflow.metrics.genai.EvaluationExample(
            input="Break a leg!",
            output="¡Mucha mierda!",
            score=5,
            justification="This translation uses the equivalent Spanish theater idiom, showing high cultural awareness."
        ),
        mlflow.metrics.genai.EvaluationExample(
            input="It's raining cats and dogs.",
            output="Está lloviendo gatos y perros.",
            score=1,
            justification="This literal translation does not convey the idiomatic meaning of heavy rain."
        ),
        mlflow.metrics.genai.EvaluationExample(
            input="It's raining cats and dogs.",
            output="Está lloviendo a cántaros.",
            score=5,
            justification="This translation uses a Spanish idiom that accurately conveys the meaning of heavy rain."
        ),
        mlflow.metrics.genai.EvaluationExample(
            input="Kick the bucket.",
            output="Patear el balde.",
            score=1,
            justification="This literal translation fails to convey the idiomatic meaning of dying."
        ),
        mlflow.metrics.genai.EvaluationExample(
            input="Kick the bucket.",
            output="Estirar la pata.",
            score=5,
            justification="This translation uses the equivalent Spanish idiom for dying, showing high cultural awareness."
        ),
        mlflow.metrics.genai.EvaluationExample(
            input="Once in a blue moon.",
            output="Una vez en una luna azul.",
            score=2,
            justification="This literal translation does not capture the rarity implied by the idiom."
        ),
        mlflow.metrics.genai.EvaluationExample(
            input="Once in a blue moon.",
            output="De vez en cuando.",
            score=4,
            justification="This translation captures the infrequency but lacks the idiomatic color of the original."
        ),
        mlflow.metrics.genai.EvaluationExample(
            input="The ball is in your court.",
            output="La pelota está en tu cancha.",
            score=3,
            justification="This translation is understandable but somewhat lacks the idiomatic nuance of making a decision."
        ),
        mlflow.metrics.genai.EvaluationExample(
            input="The ball is in your court.",
            output="Te toca a ti.",
            score=5,
            justification="This translation accurately conveys the idiomatic meaning of it being someone else's turn to act."
        )
    ],
    model="openai:/gpt-4",
    parameters={"temperature": 0.0},
)

# Check if there is an active run
if mlflow.active_run():
    mlflow.end_run()
# Log and evaluate the model
with mlflow.start_run() as run:
    results = mlflow.evaluate(
        basic_translation_model.model_uri,
        data=eval_data,
        model_type="text",
        evaluators="default",
        extra_metrics=[cultural_sensitivity],
        evaluator_config={
        "col_mapping": {
            "inputs": "llm_inputs",
           }}
   )

mlflow.end_run()
results.tables["eval_results_table"].to_csv("./outpus/eval_results_table-gpt4-32k.csv")


cultural_sensitivity_score = results.metrics['cultural_sensitivity/v1/mean']
print(f"Cultural Sensitivity Score: {cultural_sensitivity_score}")

toxicity_score = results.metrics['toxicity/v1/mean']
# Calculate non-toxicity score
non_toxicity_score = "{:.2f}".format((1 - toxicity_score) * 100)
print(f"Non-Toxicity Score: {non_toxicity_score}%")


