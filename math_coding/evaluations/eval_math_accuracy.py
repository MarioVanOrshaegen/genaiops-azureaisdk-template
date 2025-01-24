
import os

from azure.ai.evaluation import F1ScoreEvaluator, evaluate
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

from math_coding.orchestration.python.pure_python_flow import get_math_response

load_dotenv()

def eval_run_eval(
        name,
        data_path,
        column_mapping,
        output_path
        ):
    """
    Evaluate the model using the given data and column mapping.
    """
    print(os.environ["CONNECTION_STRING"])
    project = AIProjectClient.from_connection_string(
        conn_str=f"{os.environ['CONNECTION_STRING']}",
        credential=DefaultAzureCredential()
    )
    print(project.scope)
    f1score = F1ScoreEvaluator()
    result = evaluate(
        data=data_path,
        target=get_math_response,
        evaluation_name="evaluate_math_responses",
        evaluators={
            "f1_score": f1score,
        },
        evaluator_config={
            "default": {
                "column_mapping": dict(column_mapping)
            }
        },
        azure_ai_project=project.scope,
        output_path=f"{output_path}/{name}.json"
    )
    return result
