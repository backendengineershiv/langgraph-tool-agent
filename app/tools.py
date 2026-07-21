from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a basic arithmetic expression, e.g. '15% of 340' -> '15/100*340'.
    Use this whenever the user asks for a calculation instead of guessing the answer.
    """
    try:
        allowed = set("0123456789+-*/(). ")
        if not set(expression).issubset(allowed):
            return "Error: expression contains characters I'm not allowed to evaluate."
        return str(eval(expression))
    except Exception as e:
        return f"Error evaluating expression: {e}"


TOOLS = [calculator]
