import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import HumanMessage
from dotenv import load_dotenv
from utils import get_vector_db
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
LOADED_CITIES = []
EMBEDDING_MODEL = "models/embedding-001"

# Initialize embeddings
EMBEDDDING = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=api_key
)

# Get the shared FAISS vector database instance
VECTOR_DB = get_vector_db()

async def extract_city_from_query(query):
    """
    Extract city name from user query using Google Gemini LLM through LangChain
    Examples:
    "I am looking for property in bangalore" -> "bangalore"
    "Show me apartments in Mumbai" -> "mumbai"
    """
    try:
        # Initialize Google Gemini chat model through LangChain
        # Make sure you have set GOOGLE_API_KEY in your environment variables
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        
        # Create a prompt to extract the city
        prompt = f"""
        Extract the Indian city name from the following query. 
        Return ONLY the city name in lowercase without any additional text or punctuation.
        If multiple cities are mentioned, return the main city where the user wants to search for property.
        If no city is mentioned, return 'mumbai' as the default.
        If 'bengaluru' is mentioned, return 'bangalore' instead.
        
        Query: {query}
        """
        
        # Query the LLM
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        city = response.content.strip().lower()
        
        # Handle potential additional text in the response
        # Remove any non-alphanumeric characters for safety
        city = ''.join(c for c in city if c.isalnum() or c.isspace()).strip()
        
        # Handle common cases
        if city == "bengaluru":
            city = "bangalore"
        
        # Provide feedback
        if city != "mumbai" or "mumbai" in query.lower():
            print(f"Detected city: {city}")
        else:
            print("No city clearly detected. Using default: mumbai")
        
        return city
    
    except Exception as e:
        print(f"Error using LLM for city extraction: {e}")
        print("Falling back to default city: mumbai")
        return "mumbai"

def query_db(prompt: str, k: int = 5):
    """Query the FAISS vector database"""
    try:
        return VECTOR_DB.similarity_search(prompt, k=k)
    except Exception as e:
        print(f"Error querying vector database: {str(e)}")
        return []

def get_vector_db_info():
    """Get information about the current vector database"""
    try:
        if VECTOR_DB and hasattr(VECTOR_DB, 'index'):
            return {
                "total_documents": VECTOR_DB.index.ntotal,
                "dimension": VECTOR_DB.index.d,
                "loaded_cities": LOADED_CITIES
            }
        else:
            return {"error": "Vector database not initialized"}
    except Exception as e:
        return {"error": str(e)}

def search_by_city(prompt: str, city_name: str, k: int = 5):
    """Search for properties in a specific city"""
    try:
        results = VECTOR_DB.similarity_search(
            prompt, 
            k=k,
            filter=lambda metadata: metadata.get('city', '').lower() == city_name.lower()
        )
        return results
    except Exception as e:
        print(f"Error searching by city: {str(e)}")
        return []

