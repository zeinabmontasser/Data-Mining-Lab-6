from typing import TypedDict, Annotated
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_name: str
    conversation_count: int

@tool

def get_current_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Get the current time. Returns the current date and time.
    
    Args:
        format: The format string for the time (default: "%Y-%m-%d %H:%M:%S")
    
    Returns:
        Current date and time as a formatted string
    """
    return datetime.now().strftime(format)

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.
    
    Args:
        expression: A mathematical expression (e.g., "2 + 2", "10 * 5", "100 / 4")
    
    Returns:
        The result of the calculation as a string
    """
    try:
        safe_dict = {
            "abs": abs, "round": round, "min": min, "max": max,
            "pow": pow, "sqrt": lambda x: x ** 0.5
        }
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return str(result)
    except Exception as e:
        return f"Error calculating: {str(e)}"

@tool
def get_greeting(name: str) -> str:
    """Generate a personalized greeting for a user.
    
    Args:
        name: The name of the person to greet
    
    Returns:
        A friendly greeting message
    """
    hour = datetime.now().hour
    if hour < 12:
        time_of_day = "Good morning"
    elif hour < 17:
        time_of_day = "Good afternoon"
    else:
        time_of_day = "Good evening"
    return f"{time_of_day}, {name}! Welcome to our conversation."

tools = [get_current_time, calculate, get_greeting]
llm_with_tools = llm.bind_tools(tools)

def chatbot_node(state: State):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: State) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

tool_node = ToolNode(tools=tools)

graph = StateGraph(State)
graph.add_node("chatbot", chatbot_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chatbot")
graph.add_conditional_edges(
    "chatbot",
    should_continue,
    {"tools": "tools", END: END}
)
graph.add_edge("tools", "chatbot")

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    print("=" * 50)
    print("LangGraph Custom Chatbot - Graph-Based Workflow")
    print("=" * 50)
    print("\nAvailable tools:")
    print("- Time: Ask about the current time")
    print("- Calculator: Do math calculations (e.g., '2+2', '10*5')")
    print("- Greeter: Get personalized greetings")
    print("\nType 'quit' to exit\n")
    
    config = {"configurable": {"thread_id": "conversation_1"}}
    
    initial_state = {
        "messages": [],
        "user_name": "User",
        "conversation_count": 0
    }
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        print("\n--- Graph Execution ---")
        try:
            result = app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config
            )
            
            for message in result["messages"]:
                if isinstance(message, AIMessage):
                    if message.content:
                        print(f"Assistant: {message.content}")
                    if message.tool_calls:
                        for tc in message.tool_calls:
                            print(f"Assistant called tool: {tc['name']}")
                    if hasattr(message, 'tool_response') and message.tool_response:
                        print(f"Tool result: {message.tool_response}")
        except Exception as e:
            print(f"Error: {e}")
        print()
