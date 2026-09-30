import os
import json
import ast
import re
from typing import Dict, Any, Union
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage

from src.models.predict import predict_churn_risk
from src.rag.rag_pipeline import query_retention_playbooks

load_dotenv()


# ==========================================
# Helper: Parse String to Dictionary
# ==========================================

def parse_customer_input(input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Converts various user string formats into a standardized customer profile dictionary."""
    if isinstance(input_data, dict):
        return input_data

    if not isinstance(input_data, str):
        raise ValueError("Input data must be a string or dictionary.")

    # Attempt 1: Direct JSON parse
    try:
        return json.loads(input_data)
    except Exception:
        pass

    # Attempt 2: Python dict literal
    try:
        return ast.literal_eval(input_data)
    except Exception:
        pass

    # Attempt 3: Regex fallback for Key=Value or Key: Value pairs
    pattern = r"([A-Za-z0-9_]+)\s*[:=]\s*['\"]?([^,'\"]+)['\"]?"
    matches = re.findall(pattern, input_data)

    if matches:
        parsed_dict = {}
        for key, val in matches:
            val = val.strip()
            if val.isdigit():
                val = int(val)
            else:
                try:
                    val = float(val)
                except ValueError:
                    pass
            parsed_dict[key] = val
        return parsed_dict

    raise ValueError(f"Could not parse customer profile string: {input_data}")


# ==========================================
# 1. Tools Definition
# ==========================================

@tool
def churn_prediction_tool(customer_profile: Union[str, Dict[str, Any]]) -> str:
    """
    Predicts the churn risk probability for a customer.
    Input MUST contain customer attributes such as Tenure, Contract, MonthlyCharges, PaymentMethod, InternetService.
    Can be passed as a dictionary or a key-value string.
    """
    try:
        profile_dict = parse_customer_input(customer_profile)
        churn_prob = predict_churn_risk(profile_dict)
        return f"Predicted Churn Risk Probability: {round(churn_prob * 100, 2)}%"
    except Exception as e:
        return f"Error executing churn_prediction_tool: {str(e)}"


@tool
def playbook_retrieval_tool(query: str) -> str:
    """
    Searches retention playbooks for strategies, scripts, discounts, and customer success guidelines.
    """
    try:
        results = query_retention_playbooks(query, k=3)
        if not results:
            return "NO_PLAYBOOK_FOUND: No relevant retention playbooks found for this query."

        return "\n\n".join([
            f"--- Source: {res['source']} ---\n{res['content']}" for res in results
        ])
    except Exception as e:
        return f"Error querying retention playbooks: {str(e)}"


# ==========================================
# 2. Agent System Prompt
# ==========================================

SYSTEM_PROMPT = """You are an AI Customer Retention Specialist for a telecommunications company.

Your goal is to assist retention agents by analyzing customer profiles and finding retention strategies from company playbooks.

You have access to two tools:
1. `churn_prediction_tool`: Calculates the customer's churn risk probability.
2. `playbook_retrieval_tool`: Retrieves retention strategies and agent talk tracks.

INSTRUCTIONS:
- Whenever customer details are mentioned (e.g. Tenure, Contract, MonthlyCharges, PaymentMethod), ALWAYS call `churn_prediction_tool`.
- Call `playbook_retrieval_tool` to search for relevant strategies or script offers.
- Synthesize all tool output into a helpful response containing: Risk Diagnosis, Retention Strategies, and Agent Talk Track.
- If a query is completely unrelated to customer retention or telecommunications (e.g. cooking, employee leave, unrelated policies), state that the topic is outside the scope of company retention playbooks.
"""


# ==========================================
# 3. RetentionAgent Execution Runner
# ==========================================

class RetentionAgent:
    def __init__(self):
        self.tools = {
            "churn_prediction_tool": churn_prediction_tool,
            "playbook_retrieval_tool": playbook_retrieval_tool,
        }

        llm = ChatGroq(
            model_name="openai/gpt-oss-20b",
            temperature=0.0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )

        self.llm_with_tools = llm.bind_tools([churn_prediction_tool, playbook_retrieval_tool])

    def invoke(self, user_input: str) -> str:
        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", user_input),
        ]

        ai_msg = self.llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        max_iterations = 5
        iteration = 0

        while ai_msg.tool_calls and iteration < max_iterations:
            iteration += 1
            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call["name"].lower()
                selected_tool = self.tools.get(tool_name)

                if selected_tool:
                    tool_output = selected_tool.invoke(tool_call["args"])
                else:
                    tool_output = f"Tool '{tool_name}' not recognized."

                messages.append(
                    ToolMessage(
                        content=str(tool_output),
                        tool_call_id=tool_call["id"],
                    )
                )

            ai_msg = self.llm_with_tools.invoke(messages)
            messages.append(ai_msg)

        return ai_msg.content


def get_retention_agent():
    return RetentionAgent()