import os
import pickle
from dotenv import load_dotenv
import chainlit as cl
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_community.vectorstores import FAISS
from langgraph.graph.graph import CompiledGraph
from crawl_test import LOADED_CITIES, VECTOR_DB, query_db
from tools import scrape_properties_by_city, calculate_mortgage, fetch_properties_by_city
from utils import load_and_process_markdown,get_vector_db
import glob

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Constants
MARKDOWN_BASE_DIR = "./scraped_output_*"  # Pattern to match all scraped output directories
FAISS_INDEX_PATH = "./faiss_index"  # Path to save/load FAISS index
FAISS_METADATA_PATH = "./faiss_metadata.pkl"  # Path to save/load metadata
MODEL_NAME = "gemini-2.0-flash"
EMBEDDING_MODEL = "models/embedding-001"
MEMORY_SIZE = 10  # Number of conversation turns to remember

# Initialize components
memory = ConversationBufferWindowMemory(k=MEMORY_SIZE, return_messages=True, memory_key="chat_history")
chain = None

LLM = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=api_key,
    convert_system_message_to_human=True
)

# Initialize embeddings
EMBEDDDING = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=api_key
)

# Initialize FAISS vector database (will be loaded/created later)
VECTOR_DB = None

AGENT = create_react_agent(
    model=LLM,
    tools = [scrape_properties_by_city, fetch_properties_by_city, calculate_mortgage],
    prompt="""You are a helpful real estate assistant that provides information about properties based on real-time data. Use relevant tools to help answer the user queries.""",
    checkpointer=InMemorySaver(),
    store=InMemoryStore()
)

def save_faiss_index():
    """Save FAISS index and metadata to disk"""
    global VECTOR_DB
    try:
        if VECTOR_DB is not None:
            VECTOR_DB.save_local(FAISS_INDEX_PATH)
            
            # Save loaded cities metadata
            with open(FAISS_METADATA_PATH, 'wb') as f:
                pickle.dump(LOADED_CITIES, f)
            
            print(f"FAISS index saved to {FAISS_INDEX_PATH}")
            print(f"Metadata saved to {FAISS_METADATA_PATH}")
    except Exception as e:
        print(f"Error saving FAISS index: {str(e)}")

def load_faiss_index():
    """Load FAISS index and metadata from disk"""
    global VECTOR_DB, LOADED_CITIES
    try:
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_METADATA_PATH):
            # Load FAISS index
            VECTOR_DB = FAISS.load_local(
                FAISS_INDEX_PATH, 
                EMBEDDDING,
                allow_dangerous_deserialization=True
            )
            
            # Load metadata
            with open(FAISS_METADATA_PATH, 'rb') as f:
                LOADED_CITIES.extend(pickle.load(f))
            
            print(f"FAISS index loaded from {FAISS_INDEX_PATH}")
            print(f"Loaded cities: {LOADED_CITIES}")
            return True
    except Exception as e:
        print(f"Error loading FAISS index: {str(e)}")
    
    return False

async def load_existing_vector_db():
    """Load existing markdown files from all scraped directories into the vector database"""
    global VECTOR_DB
    
    try:
        # First try to load existing FAISS index
        if load_faiss_index():
            print("Existing FAISS index loaded successfully!")
            return
        
        # If no existing index, create new one
        print("No existing FAISS index found. Creating new one...")
        
        # Find all scraped output directories
        scraped_dirs = glob.glob(MARKDOWN_BASE_DIR)
        
        if not scraped_dirs:
            print("No existing scraped directories found.")
            return
        
        all_chunks = []
        total_chunks = 0
        
        for dir_path in scraped_dirs:
            if os.path.isdir(dir_path):
                # Extract city name from directory name (e.g., "scraped_output_mumbai" -> "mumbai")
                city_name = os.path.basename(dir_path).replace("scraped_output_", "")
                
                # Check if there are markdown files in this directory
                md_files = glob.glob(os.path.join(dir_path, "*.md"))
                if md_files:
                    print(f"Loading markdown files from {dir_path} for city: {city_name}")
                    
                    # Load and process markdown files from this directory
                    chunks = await load_and_process_markdown(dir_path)
                    
                    if chunks:
                        # Add city metadata to chunks
                        for chunk in chunks:
                            chunk.metadata['city'] = city_name
                        
                        all_chunks.extend(chunks)
                        total_chunks += len(chunks)
                        
                        # Add city to loaded cities list if not already present
                        if city_name not in LOADED_CITIES:
                            LOADED_CITIES.append(city_name)
                        
                        print(f"Added {len(chunks)} chunks for {city_name}")
        
        # Create FAISS vector database if we have chunks
        if all_chunks:
            VECTOR_DB = FAISS.from_documents(all_chunks, EMBEDDDING)
            
            # Save the index immediately
            save_faiss_index()
            
            print(f"Created FAISS index with {total_chunks} chunks from {len(scraped_dirs)} directories")
            print(f"Available cities: {LOADED_CITIES}")
        else:
            print("No chunks to create FAISS index")
        
    except Exception as e:
        print(f"Error loading existing markdown files: {str(e)}")

@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session"""
    global VECTOR_DB
    
    # Send a welcome message
    welcome_message = cl.Message(content="Welcome to Real Estate Assistant! I'm initializing the system...")
    await welcome_message.send()
    
    # Initialize the LLM
    try:
        # Add to user session
        cl.user_session.set("llm", LLM)
        
        # Get the shared vector DB instance
        VECTOR_DB = get_vector_db()
        
        # Load existing vector database from scraped files
        with cl.Step("Loading existing property data...") as step:
            await load_existing_vector_db()
            if LOADED_CITIES:
                step.output = f"FAISS vector database loaded successfully! Available cities: {', '.join(LOADED_CITIES)}"
            else:
                step.output = "No existing property data found. You can scrape new cities using the scrape tool."

        
        # Create chain
        with cl.Step("Building conversation chain...") as step:
            step.output = "Agent set-up successfully!"
            cl.user_session.set("agent", AGENT)
        
        # Update the welcome message correctly
        if LOADED_CITIES:
            ready_message = cl.Message(
                content=f"Real Estate Assistant is ready! Available cities: {', '.join(LOADED_CITIES)}. How can I help you today? You can ask about properties or mortgage calculations."
            )
        else:
            ready_message = cl.Message(
                content="Real Estate Assistant is ready! No property data loaded yet. You can ask me to scrape properties for a specific city first."
            )
        await ready_message.send()
        
        
    except Exception as e:
        error_message = cl.Message(content=f"Error initializing the system: {str(e)}")
        await error_message.send()

@cl.on_message
async def on_message(message: cl.Message):
    """Process user messages"""
    agent: CompiledGraph = cl.user_session.get("agent")
    
    if not agent:
        error_message = cl.Message(content="I'm still initializing. Please wait a moment...")
        await error_message.send()
        return
    try:
        # Send the message to thinking state
        msg = cl.Message(content="Thinking...")
        await msg.send()
        
        # Get the response from the chain
        response = ""
        async for step in agent.astream(input={"messages": [message.content]}, config={"configurable": {"thread_id": "1"}}, stream_mode="values"):
            response = step["messages"][-1].content
            step["messages"][-1].pretty_print()
        
        # Update the message with the response - fix the update method
        await msg.remove()
        response_message = cl.Message(content=response)
        await response_message.send()
            
    except Exception as e:
        error_message = cl.Message(content=f"Error: {str(e)}")
        await error_message.send()

@cl.on_settings_update
async def setup_agent(settings):
    """Update settings if needed"""
    print(f"Settings updated: {settings}")

# Cleanup function to save FAISS index on exit
import atexit
atexit.register(save_faiss_index)