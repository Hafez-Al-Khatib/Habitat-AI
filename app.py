import os
import google.generativeai as genai
import gradio as gr 
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from operator import itemgetter

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

# Functions
def record_customer_interest(email: str, name: str, message: str) -> str:
    """
    Use this tool to book a consultation or start a new project for a user.
    It captures the user's name, email, and project details to log them as a new lead. [cite: 5, 30]
    """
    log_entry = f"NEW PROJECT LEAD:\n Name: {name}\n Email: {email}\n PROJECT DETAILS: {message}\n---\n"
    print(log_entry)
    with open("leads.log", "a") as f:
        f.write(log_entry)
    return "Excellent! I've logged your project request. A design consultant from Habitat AI will contact you at your email within 24 hours to discuss the next steps."

def record_feedback(question: str):
    """
    Use this tool when a user asks about a service or location that is not in the knowledge base.
    It records the user's specific question as feedback for the team to review. [cite: 6, 30]
    """
    log_entry = f"USER FEEDBACK / UNSUPPORTED REQUEST:\n  Question: {question}\n---\n"
    print(log_entry)
    with open("feedback.log", "a") as f:
        f.write(log_entry)
    return "Great question. I don't have information on that in my current knowledge base, but I've passed your request along to the Habitat AI team for review."

# RAG
try:
    pdf_loader = PyPDFLoader("me\\about_business.pdf")
    pdf_docs = pdf_loader.load()
except Exception as e:
    print(f"Error in knowledge base loading: {e}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

chunks = text_splitter.split_documents(pdf_docs)

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

system_prompt = """
You are ArchieBot, the AI Design Assistant for Habitat AI.
Your persona is professional, creative, and highly helpful, like a real-world architect's assistant.

**Your Goal:**
Your primary purpose is to inform users about Habitat AI's services and to convert interested users into project leads by booking a consultation.

**Your Knowledge:**
Your expertise comes exclusively from the CONTEXT provided below. Do not invent services, locations, or pricing.

**Core Instructions:**
1.  **Engage and Inform:** Greet users warmly and answer their questions about our services using the provided CONTEXT.
2.  **Capture Leads:** This is your most important task. If a user expresses any intent to start a project (e.g., "I want to redo my living room," "How much would this cost?," "How do I start?"), you MUST guide them toward booking a consultation by calling the `record_customer_interest` tool. You must ask for their name, email, and a brief description of their project to use the tool.
3.  **Handle Unknowns:** If a user asks a question and the answer is not in the CONTEXT, you MUST call the `record_feedback` tool with their question. Politely inform them that you've logged their request.
"""

# The main prompt template for the RAG chain
rag_prompt_template = f"""
{system_prompt}

CONTEXT:
{{context}}

CHAT HISTORY:
{{chat_history}}

USER QUESTION: {{question}}
"""

rag_prompt = PromptTemplate.from_template(rag_prompt_template)

rag_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", api_key=GOOGLE_API_KEY)
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

rag_chain = (
    {"context": itemgetter("question") | retriever | format_docs, "question": itemgetter("question"), "chat_history": itemgetter("chat_history")}
    | rag_prompt
    | rag_llm
    | StrOutputParser()
)

tool_llm = genai.GenerativeModel(
    model_name='gemini-2.0-flash-exp',
    tools=[record_customer_interest, record_feedback]
)

def chat_response(message, history):
    chat = tool_llm.start_chat(history=[])
    formatted_history = "\n".join([f"Human: {h[0]}\nAI: {h[1]}" for h in history])
    response = chat.send_message(message)
    try:
        function_call = response.candidates[0].content.parts[0].function_call
        tool_name = function_call.name
        tool_args = function_call.args

        print(f"Tool call: {tool_name} with args {tool_args}")

        if tool_name == "record_customer_interest":
            return record_customer_interest(tool_args['email'], tool_args['name'], tool_args['message'])
        elif tool_name == "record_feedback":
            return record_feedback(tool_args['question'])
        else:
            return rag_chain.invoke({"question": message, "chat_history": formatted_history})
    except (AttributeError, IndexError):
        print("No tool call detected.")
        return rag_chain.invoke({"question": message, "chat_history": formatted_history})

    
        
print("Launching Gradio interface...")
interface = gr.ChatInterface(
    fn=chat_response,
    title="Habitat AI 🏠",
    description="I'm ArchieBot, your AI Design Assistant. Ask me about our services or how to start your dream project!",
    examples=[
        ["What services do you offer?"],
        ["I want to renovate my kitchen, how do I start?"],
        ["Do you have construction teams in London?"]
    ]
)
interface.launch(share=True, debug=True)
