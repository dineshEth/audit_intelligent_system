PLANNER_PROMPT = """You are a planning agent for an offline audit assistant.
Break the task into ordered subtasks and return JSON with:
{"tasks": [{"name": "...", "goal": "..."}], "document_type": "..."}
User request:
{query}
Document types:
{doc_types}
"""

SUMMARY_PROMPT = """You are an audit assistant working fully offline.
Using the provided context and analysis, write a concise audit summary.
Mention numbers only if present in the context.
Context:
{context}

Analysis:
{analysis}

User question:
{query}
"""

TRANSACTION_LABEL_PROMPT = """Classify the bank transaction into exactly one category.
Allowed categories:
FOOD, TRAVEL, UTILITIES, TRANSFER, CASH, INCOME, SUBSCRIPTION, SHOPPING, HEALTH, FEES, ENTERTAINMENT, HOUSING, TAX, LOAN, EDUCATION, INSURANCE, UNCATEGORIZED

Transaction:
Description: {description}
Debit: {debit}
Credit: {credit}

Return only the category label.
"""

QA_GENERATION_PROMPT = """Generate 10 to 20 question-answer pairs from the document context.
Return JSON as a list of objects with keys: question, answer.
Context:
{context}
"""
