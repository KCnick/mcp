import os
from dotenv import load_dotenv

load_dotenv()

MCP_SERVER_URL = "https://vipfapwm3x.us-east-1.awsapprunner.com/mcp"
MCP_SERVER_NAME = "aws-service"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"

PAGE_TITLE = "Computer Products Support Chatbot"
PAGE_ICON = "🖥️"

SYSTEM_PROMPT = """You are a helpful customer support assistant for a company that sells computer products including monitors, printers, keyboards, mice, and other peripherals.

Your role is to:
1. Answer questions about products
2. Help with order inquiries
3. Provide technical support
4. Use available tools to look up information when needed

Be friendly, professional, and concise. If you need to use a tool, explain what you're doing.
"""

MAX_HISTORY = 10  # Keep last 10 messages in context