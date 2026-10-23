import os, json,sqlite3
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
import base64
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage ,BaseMessage
from sentence_transformers import SentenceTransformer
import chromadb
import urllib.parse
import requests
from fastapi import WebSocket,FastAPI
from bs4 import BeautifulSoup

# 🧱 LangGraph ke standard tools ko import kiya
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from typing import Annotated 
load_dotenv()


embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")


if not os.path.exists("users.db"):

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            firstname TEXT NOT NULL,
            otp INTEGER DEFAULT 0,
            verified TEXT DEFAULT "False"
        )
    """)

    conn.commit()
    conn.close()



@tool
def search_job(query:str) -> str:
    """
    Real-time RAG Tool (Bing Powered). Bing se query ke top documents scrape karta hai,
    phir local Llama model (Ollama) ke through response generate karke deta hai.
    """

    url = f"https://www.bing.com/search?q={query}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }
    snippets = []

    try:
        response = requests.get(url, headers=headers, timeout=6)
        soup = BeautifulSoup(response.text, "html.parser")
        # Bing ke standard search result cards 'li.b_algo' hote hain
        for card in soup.select("li.b_algo"):
            # Description container nikalna
            snippet_element = card.select_one(".b_caption p") or card.select_one("p.b_algoSlug")
            if snippet_element:
                text = snippet_element.get_text().strip()
                if text and text not in snippets:
                    snippets.append(text)
                    

    except Exception as e:

        scraped_context = f"Bing scraping link broken temporarily: {str(e)}"


    embeddings_numbers = embeddings_model.encode(snippets).tolist()

    #load vector db libaray

    c= chromadb.Client()
    create_storage = c.get_or_create_collection("mycollection")
    create_storage.add(documents=snippets,embeddings=embeddings_numbers,ids=[str(i) for i in range(len(snippets))])



    #convert queery to embeeding

    query_number = embeddings_model.encode(query)
    result = create_storage.query(
        query_embeddings=[query_number.tolist()],
        n_results=1
    )



    system_prompt = (
        "Aap ek intellectual AI Assistant hain. Niche diye gaye 'Scraped Context' "
        "ko achhe se samajhein aur uske basis par user ki query ka sahi aur "
        "up-to-date answer generate karein."
    )
    
    full_prompt = (
        f"{system_prompt}\n\n"
        f"Scraped Context:\n{result['documents']}\n\n"
        f"Query: {query}\n\n"
        f"Answer:"
    )
    
    # Local Ollama endpoint setup
    ollama_url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3",
        "prompt": full_prompt,
        "stream": True
    }

    try:
        llama_response = requests.post(ollama_url, stream=True,json=payload, timeout=180)
        
        if llama_response.status_code == 200:
            ai_reply=""
            # ai_reply = llama_response.json().get("response", "Empty text generated.")

            for chunk in llama_response.iter_lines():
                if chunk:
                    data = json.loads(chunk)
                    ai_reply += data.get("response", "")
                    if data.get("done"):
                        break

            return f"[AI]: {ai_reply}"
        else:
            return f"[AI Error]: Local model server active but responded with status code {llama_response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return "[AI Error]: Local Ollama infrastructure engine offline hai. Please check terminal."
    except Exception as e:
        return f"[AI Error]: Pipeline logic failure: {str(e)}"



model = ChatOllama(
    model="llama3.1:8b",  
    max_tokens=400,
    temperature=0,
    request_timeout=120

)

toollist=[search_job]
llm_with_tools = model.bind_tools(toollist)

resp = {}


class FinanceState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]





def starting_point(state: FinanceState):

    prompt = SystemMessage(content="""
    You are a tool router.

    You have 1 tools:
    1. search_job

    IMPORTANT WORKFLOW:

    - After search_job completes, STOP.
    - Always inspect the previous tool result before selecting the next tool.

    IMPORTANT:
    The result returned by each tool must remain exactly as returned by that tool.
    Do not rewrite or summarize tool results.
    """)

    response = llm_with_tools.invoke(
        [prompt] + state["messages"]
    )

    return {"messages": [response]}

def check_tool_available(state: FinanceState):
    # Agar tool call hai toh tools par bhejo, nahi toh END karo
    return "tools_path" if state["messages"][-1].tool_calls else "end_path"


workflow = StateGraph(FinanceState)

workflow.add_node("starting_point", starting_point)

workflow.add_node("tool_list", ToolNode(toollist))

workflow.add_edge(START, "starting_point")
workflow.add_edge("tool_list", "starting_point")

workflow.add_conditional_edges(
    "starting_point", 
    check_tool_available, 
    {"tools_path": "tool_list", "end_path": END}
)


lang_app = workflow.compile()

app = FastAPI()


@app.websocket("/ws/ai")
async def output(websocket:WebSocket):
    await websocket.accept()
    while True:
        user_text = await websocket.receive_text()
        print(user_text)

        state_object:FinanceState={"messages":[]}
        state_object['messages'].append(HumanMessage(content=user_text))
        final_state = lang_app.invoke(state_object)
        response = ""
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, ToolMessage):
                response=msg.content
                break
        await websocket.send_text(response)
