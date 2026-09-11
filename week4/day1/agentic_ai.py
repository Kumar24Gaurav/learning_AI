import ast
import json
import operator
import os
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from tavily.tavily import TavilyClient


load_dotenv()

groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def calculate(expression: str) -> int | float:
	"""Evaluate a basic arithmetic expression without executing arbitrary code."""
	operators: dict[type[ast.operator], Any] = {
		ast.Add: operator.add,
		ast.Sub: operator.sub,
		ast.Mult: operator.mul,
		ast.Div: operator.truediv,
		ast.FloorDiv: operator.floordiv,
		ast.Mod: operator.mod,
		ast.Pow: operator.pow,
		ast.USub: operator.neg,
		ast.UAdd: operator.pos,
	}

	def evaluate(node: ast.AST) -> int | float:
		if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
			return node.value
		if isinstance(node, ast.BinOp) and type(node.op) in operators:
			left = evaluate(node.left)
			right = evaluate(node.right)
			return operators[type(node.op)](left, right)
		if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
			return operators[type(node.op)](evaluate(node.operand))
		raise ValueError("Only numbers and basic arithmetic operators are supported")

	parsed_expression = ast.parse(expression, mode="eval")
	return evaluate(parsed_expression.body)


def web_search(query: str) -> str:
	"""Search the web with Tavily and return its answer with source details."""
	result = tavily.search(query=query, search_depth="advanced", max_results=5)
	answer = result.get("answer")
	if answer:
		return answer

	sources = result.get("results", [])
	if not sources:
		return "Tavily did not return an answer or search results."

	return "\n\n".join(
		f"{source.get('title', 'Untitled')}: {source.get('content', '')}"
		for source in sources
	)


tools = [
	{
		"type": "function",
		"function": {
			"name": "web_search",
			"description": "Search the web for current or factual information.",
			"parameters": {
				"type": "object",
				"properties": {
					"query": {
						"type": "string",
						"description": "The question or search query to look up.",
					}
				},
				"required": ["query"],
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "calculate",
			"description": "Calculate a mathematical expression such as 2 * 2.",
			"parameters": {
				"type": "object",
				"properties": {
					"expression": {
						"type": "string",
						"description": "A basic arithmetic expression to evaluate.",
					}
				},
				"required": ["expression"],
			},
		},
	},
]


available_tools = {
	"web_search": web_search,
	"calculate": calculate,
}


def ask_agent(user_input: str) -> str:
	"""Let Groq choose tools, execute them, and return the final answer."""
	messages: list[dict[str, Any]] = [
		{
			"role": "system",
			"content": (
				"You are a helpful assistant. Use web_search for current or external "
				"information and calculate for arithmetic. Explain the result clearly."
			),
		},
		{"role": "user", "content": user_input},
	]

	while True:
		completion = groq.chat.completions.create(
			model="openai/gpt-oss-120b",
			messages=messages,
			tools=tools,
			tool_choice="auto",
		)
		assistant_message = completion.choices[0].message

		if not assistant_message.tool_calls:
			return assistant_message.content or "I could not generate an answer."

		messages.append(
			{
				"role": "assistant",
				"content": assistant_message.content,
				"tool_calls": [
					{
						"id": tool_call.id,
						"type": "function",
						"function": {
							"name": tool_call.function.name,
							"arguments": tool_call.function.arguments,
						},
					}
					for tool_call in assistant_message.tool_calls
				],
			}
		)

		for tool_call in assistant_message.tool_calls:
			tool_name = tool_call.function.name
			tool = available_tools.get(tool_name)
			if tool is None:
				tool_result = f"Unknown tool: {tool_name}"
			else:
				try:
					arguments = json.loads(tool_call.function.arguments)
					tool_result = str(tool(**arguments))
				except (ValueError, TypeError, json.JSONDecodeError) as error:
					tool_result = f"Tool error: {error}"

			messages.append(
				{
					"role": "tool",
					"tool_call_id": tool_call.id,
					"content": tool_result,
				}
			)


if __name__ == "__main__":
	user_input = input("Ask me anything: ")
	print(ask_agent(user_input))
