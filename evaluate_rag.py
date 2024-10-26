import pandas as pd
import mlflow
import openai

import os

from dotenv import load_dotenv
from show_finale_metrics import create_gauge_chart


load_dotenv()

# Run a quick validation that we have an entry for the OPEN_API_KEY within environment variables

assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY environment variable must be set"


# Using custom function
def my_llm(inputs):
    answers = []
    system_prompt = "Please answer the following question in formal language based on the context provided."
    for index, row in inputs.iterrows():
        print("INPUTS:", row)
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{row}"},
            ],
            temperature=0,
        )
        answers.append(completion.choices[0].message.content)

    return answers


# Prepare evaluation data
eval_data = pd.DataFrame(
    {
        "llm_inputs": [
            """Question: What is the company's policy on employee training?
context: "Our company offers various training programs to support employee development. Employees are required to complete at least one training course per year related to their role. Additional training opportunities are available based on performance reviews." """,
            """Question: What is the company's policy on sick leave?
context: "Employees are entitled to 10 days of paid sick leave per year. Sick leave can be used for personal illness or to care for an immediate family member. A doctor's note is required for sick leave exceeding three consecutive days." """,
            """Question: How does the company handle performance reviews?
context: "Performance reviews are conducted annually. Employees are evaluated based on their job performance, goal achievement, and overall contribution to the team. Feedback is provided, and development plans are created to support employee growth." """,
        ]
    }
)

# exemples pour cette mesure de fidélité
examples = [
    mlflow.metrics.genai.EvaluationExample(
        input="""Question: What is the company's policy on remote work?
context: "Our company supports a flexible working environment. Employees can work remotely up to three days a week, provided they maintain productivity and attend all mandatory meetings." """,
        output="Employees can work remotely up to three days a week if they maintain productivity and attend mandatory meetings.",
        score=5,
        justification="The answer is accurate and directly related to the question and context provided.",
    ),
    mlflow.metrics.genai.EvaluationExample(
        input="""Question: What is the company's policy on remote work?
context: "Our company supports a flexible working environment. Employees can work remotely up to three days a week, provided they maintain productivity and attend all mandatory meetings." """,
        output="Employees are allowed to work remotely as long as they want.",
        score=2,
        justification="The answer is somewhat related but incorrect because it does not mention the three-day limit.",
    ),
    mlflow.metrics.genai.EvaluationExample(
        input="""Question: What is the company's policy on remote work?
context: "Our company supports a flexible working environment. Employees can work remotely up to three days a week, provided they maintain productivity and attend all mandatory meetings." """,
        output="Our company supports flexible work arrangements.",
        score=3,
        justification="The answer is related to the context but does not specifically answer the question about the remote work policy.",
    ),
    mlflow.metrics.genai.EvaluationExample(
        input="""Question: What is the company's annual leave policy?
context: "Employees are entitled to 20 days of paid annual leave per year. Leave must be approved by the employee's direct supervisor and should be planned in advance to ensure minimal disruption to work." """,
        output="Employees are entitled to 20 days of paid annual leave per year, which must be approved by their supervisor.",
        score=5,
        justification="The answer is accurate and directly related to the question and context provided.",
    ),
]

#  Define the custom metric
faithfulness = mlflow.metrics.genai.make_genai_metric(
    name="faithfulness",
    definition="Assesses how well the answer relates to the question and provided context.",
    grading_prompt="Score from 1-5, where 1 is not related at all and 5 is highly relevant and accurate.",
    examples=examples,
)


with mlflow.start_run() as run:
    results = mlflow.evaluate(
        my_llm,
        eval_data,
        model_type="text",
        evaluators="default",
        extra_metrics=[faithfulness],
        evaluator_config={
            "col_mapping": {
                "inputs": "llm_inputs",
            }
        },
    )
mlflow.end_run()
