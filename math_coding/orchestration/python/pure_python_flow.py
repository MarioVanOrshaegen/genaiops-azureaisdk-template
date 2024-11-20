import ast
import json
import os
import sys
from io import StringIO

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.prompts import PromptTemplate
from azure.core.credentials import AzureKeyCredential


def infinite_loop_check(code_snippet):
    """Check if the code snippet has an infinite loop"""
    tree = ast.parse(code_snippet)
    for node in ast.walk(tree):
        if isinstance(node, ast.While):
            if not node.orelse:
                return True
    return False


def syntax_error_check(code_snippet):
    """Check if the code snippet has a syntax error"""
    try:
        ast.parse(code_snippet)
    except SyntaxError:
        return True
    return False


def error_fix(code_snippet):
    """Fix the code snippet by adding a break statement to the infinite loop"""
    tree = ast.parse(code_snippet)
    for node in ast.walk(tree):
        if isinstance(node, ast.While):
            if not node.orelse:
                node.orelse = [ast.Pass()]
    return ast.unparse(tree)


def code_refine(original_code: str) -> str:
    """Refine the code snippet by fixing infinite loops and syntax errors"""
    try:
        original_code = json.loads(original_code)["code"]
        fixed_code = None

        if infinite_loop_check(original_code):
            fixed_code = error_fix(original_code)
        else:
            fixed_code = original_code

        if syntax_error_check(fixed_code):
            fixed_code = error_fix(fixed_code)

        return fixed_code
    except json.JSONDecodeError:
        return "JSONDecodeError"
    except Exception as e:
        return "Unknown Error:" + str(e)


def func_exe(code_snippet: str):
    """Execute the code snippet and return the result"""
    if code_snippet == "JSONDecodeError" or code_snippet.startswith("Unknown Error:"):
        return code_snippet

    # Define the result variable before executing the code snippet
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()

    # Execute the code snippet
    try:
        exec(code_snippet.lstrip())
    except Exception as e:
        sys.stdout = old_stdout
        return str(e)

    sys.stdout = old_stdout
    return redirected_output.getvalue().strip()


def get_math_response(question):
    """Get the response for the math question"""
    try:
        endpoint = os.environ["AZURE_AI_CHAT_ENDPOINT"]
        key = os.environ["AZURE_AI_CHAT_KEY"]
        prompty_file = os.environ["PROMPTY_FILE"]
    except KeyError:
        print("Missing environment variable 'AZURE_AI_CHAT_ENDPOINT' or 'AZURE_AI_CHAT_KEY'")
        print("Set them before running this sample.")
        exit()

    path = f"./{prompty_file}"
    prompt_template = PromptTemplate.from_prompty(file_path=path)
    
    messages = prompt_template.create_messages(question=question)
    client = ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    code = client.complete(
        messages=messages,
        model=prompt_template.model_name,
        **prompt_template.parameters,
    )

    code_refined = code_refine(code.choices[0].message.content)
    result = func_exe(code_refined)
    return {"response": result}


if __name__ == "__main__":
    """Test the math response"""
    question = "what is 10 + 20?"
    result = get_math_response(question)
    print(result)
