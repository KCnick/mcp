"""
Streamlit Customer Support Chatbot Application
"""
import streamlit as st
import asyncio
from typing import List
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
import config
from mcp_client import setup_mcp_client

# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 2rem;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 2rem;
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
    .tool-badge {
        background-color: #e8f4f8;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
        margin: 0.25rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_chatbot():
    """Initialize the chatbot with MCP tools and LLM"""
    try:
        # Check for API key
        if not config.GROQ_API_KEY:
            st.error("GROQ_API_KEY not found. Please set it in your environment variables.")
            return None, None, None
        
        # Setup MCP client and tools asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        with st.spinner("🔧 Connecting to MCP servers and loading tools..."):
            mcp_client, tools = loop.run_until_complete(
                setup_mcp_client(config.MCP_SERVER_URL, config.MCP_SERVER_NAME)
            )
        
        if not tools:
            st.warning("⚠️ No tools were loaded from the MCP server. Check your server connection.")
            return None, None, None
        
        st.success(f"✅ Successfully connected! Loaded {len(tools)} tools.")
        
        # Initialize Groq LLM
        llm = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model=config.GROQ_MODEL,
            temperature=0.7,
        )
        
        # Create agent using initialize_agent (compatible with older LangChain versions)
        agent_executor = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,  # Don't show internal reasoning
            handle_parsing_errors=True,
            max_iterations=10,  # Increased from 5 to 10
            early_stopping_method="generate",  # Return best answer even if not complete
            return_intermediate_steps=False,  # Don't return intermediate steps
            agent_kwargs={
                "prefix": config.SYSTEM_PROMPT,
                "format_instructions": (
                    "Always provide your final answer directly to the user in a friendly, "
                    "conversational manner. Do not expose your internal reasoning, thoughts, "
                    "or tool usage. Present information as if you naturally know it. "
                    "Format your responses in clear, easy-to-read paragraphs or bullet points when appropriate."
                ),
                "suffix": (
                    "\n\nRemember: Give a direct, helpful answer. If you need to use tools, use them efficiently. "
                    "If you have enough information, provide your answer immediately."
                )
            }
        )
        
        return agent_executor, tools, mcp_client
        
    except Exception as e:
        st.error(f"❌ Error initializing chatbot: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None


def display_chat_history():
    """Display chat message history"""
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(content)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content)


def format_chat_history(messages):
    """Format chat history as a string for the agent"""
    history = []
    for msg in messages[1:-1]:  # Exclude initial greeting and current message
        if msg["role"] == "user":
            history.append(f"Human: {msg['content']}")
        else:
            history.append(f"AI: {msg['content']}")
    
    # Keep only recent history
    if len(history) > config.MAX_HISTORY * 2:
        history = history[-(config.MAX_HISTORY * 2):]
    
    return "\n".join(history) if history else ""


def clean_agent_response(response_text: str) -> str:
    """Clean agent response to remove internal reasoning artifacts"""
    # Remove common agent reasoning patterns
    patterns_to_remove = [
        "Thought:",
        "Action:",
        "Action Input:",
        "Observation:",
        "Final Answer:",
        "Question:",
        "{{{{",
        "}}}}",
        '"action"',
        '"action_input"',
        "consider previous and subsequent steps"
    ]
    
    cleaned = response_text
    
    # Remove lines containing these patterns
    lines = cleaned.split('\n')
    filtered_lines = []
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Skip empty lines
        if not line_lower:
            continue
            
        # Skip lines that contain reasoning artifacts
        if any(pattern.lower() in line_lower for pattern in patterns_to_remove):
            continue
            
        # Skip lines with JSON-like structures
        if '"action"' in line or '"action_input"' in line:
            continue
            
        # Skip lines that look like internal formatting
        if line_lower.startswith("i know what to respond"):
            continue
            
        filtered_lines.append(line)
    
    cleaned = '\n'.join(filtered_lines).strip()
    
    # If cleaning removed everything, return a default message
    if not cleaned or len(cleaned) < 10:
        return "I've processed your request. Is there anything specific you'd like to know more about?"
    
    return cleaned


def main():
    """Main application function"""
    
    # Header
    st.markdown('<div class="main-header">🖥️ Computer Products Support Chatbot</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! 👋 I'm your customer support assistant. I can help you with:\n\n"
                          "- 🖥️ Product information (monitors, printers, keyboards, etc.)\n"
                          "- 📦 Order status and tracking\n"
                          "- 🔧 Technical support\n"
                          "- 🔄 Returns and warranties\n\n"
                          "How can I assist you today?"
            }
        ]
    
    if "agent_executor" not in st.session_state:
        agent_executor, tools, mcp_client = initialize_chatbot()
        st.session_state.agent_executor = agent_executor
        st.session_state.tools = tools
        st.session_state.mcp_client = mcp_client
    
    # Sidebar
    with st.sidebar:
        st.subheader("💬 Chat Controls")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = [st.session_state.messages[0]]  # Keep only greeting
            st.rerun()
        
        st.markdown("---")
        
        # Display connection status
        st.subheader("📡 Connection Status")
        if st.session_state.mcp_client:
            st.success("✅ Connected to MCP Server")
            st.caption(f"Server: {config.MCP_SERVER_NAME}")
        else:
            st.error("❌ Not connected")
        
        st.markdown("---")
        
        # Display available tools
        if st.session_state.tools:
            st.subheader(f"🛠️ Available Tools ({len(st.session_state.tools)})")
            for tool in st.session_state.tools:
                with st.expander(f"📌 {tool.name}"):
                    st.write(tool.description)
        else:
            st.warning("No tools available")
        
        st.markdown("---")
        
        # Model info
        st.subheader("ℹ️ Model Information")
        st.caption(f"**LLM:** {config.GROQ_MODEL}")
        st.caption(f"**Provider:** Groq")
        
        st.markdown("---")
        st.caption("Powered by Groq & MCP")
    
    # Check if agent is initialized
    if st.session_state.agent_executor is None:
        st.error("⚠️ Failed to initialize the chatbot. Please check your configuration and try again.")
        
        with st.expander("🔍 Troubleshooting"):
            st.markdown("""
            **Common issues:**
            1. Check that GROQ_API_KEY is set in your environment
            2. Verify MCP server URL is accessible
            3. Ensure all dependencies are installed
            4. Check the error messages above for details
            """)
        st.stop()
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    if user_input := st.chat_input("Type your message here..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        
        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🤔 Thinking..."):
                try:
                    # Format chat history
                    chat_history_str = format_chat_history(st.session_state.messages)
                    
                    # Prepare input with history context
                    full_input = user_input
                    if chat_history_str:
                        full_input = f"Chat History:\n{chat_history_str}\n\nCurrent Question: {user_input}"
                    
                    # Get response from agent
                    response = st.session_state.agent_executor.invoke({
                        "input": full_input
                    })
                    
                    # Extract and clean the output
                    raw_output = response.get("output", "I apologize, but I couldn't generate a response.")
                    assistant_message = clean_agent_response(raw_output)
                    
                    # Display the cleaned message
                    st.markdown(assistant_message)
                    st.session_state.messages.append({"role": "assistant", "content": assistant_message})
                    
                except Exception as e:
                    error_message = f"I apologize, but I encountered an error while processing your request. Please try again or rephrase your question."
                    st.error(f"Error details: {str(e)}")
                    st.session_state.messages.append({"role": "assistant", "content": error_message})


if __name__ == "__main__":
    main()