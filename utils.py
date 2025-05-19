import os
import glob
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize embeddings
EMBEDDING_MODEL = "models/embedding-001"
EMBEDDDING = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=api_key
)

# Global FAISS vector database instance
_VECTOR_DB = None

def get_vector_db():
    """Get the global FAISS vector database instance"""
    global _VECTOR_DB
    if _VECTOR_DB is None:
        # Create empty FAISS index if none exists
        try:
            # Try to load existing index
            if os.path.exists("./faiss_index"):
                _VECTOR_DB = FAISS.load_local(
                    "./faiss_index", 
                    EMBEDDDING,
                    allow_dangerous_deserialization=True
                )
                print("Loaded existing FAISS index")
            else:
                # Create empty FAISS database
                # We need at least one document to create the initial index
                from langchain.schema import Document
                dummy_doc = Document(page_content="Initial document", metadata={})
                _VECTOR_DB = FAISS.from_documents([dummy_doc], EMBEDDDING)
                print("Created new FAISS index")
        except Exception as e:
            print(f"Error initializing FAISS: {e}")
            # Fallback: create with dummy document
            from langchain.schema import Document
            dummy_doc = Document(page_content="Initial document", metadata={})
            _VECTOR_DB = FAISS.from_documents([dummy_doc], EMBEDDDING)
    
    return _VECTOR_DB

async def load_and_process_markdown(dir: str):
    """Load markdown files and process them into chunks"""
    try:
        # Check if directory exists
        if not os.path.exists(dir):
            print(f"Directory {dir} does not exist.")
            return []
        
        # Check if directory has any markdown files
        md_files = glob.glob(os.path.join(dir, "*.md"))
        if not md_files:
            print(f"No markdown files found in {dir}")
            return []
        
        # Load markdown files
        loader = DirectoryLoader(
            dir,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}  # Specify encoding to handle special characters
        )
        
        documents = loader.load()
        print(f"Loaded {len(documents)} markdown files from {dir}")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=300,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks from the markdown files")
        
        return chunks
        
    except Exception as e:
        print(f"Error loading and processing markdown files from {dir}: {str(e)}")
        return []

def check_scraped_directories():
    """Check which scraped directories exist and return available cities"""
    scraped_dirs = glob.glob("./scraped_output_*")
    available_cities = []
    
    for dir_path in scraped_dirs:
        if os.path.isdir(dir_path):
            # Extract city name from directory name
            city_name = os.path.basename(dir_path).replace("scraped_output_", "")
            
            # Check if directory has markdown files
            md_files = glob.glob(os.path.join(dir_path, "*.md"))
            if md_files:
                available_cities.append(city_name)
    
    return available_cities

def get_scraped_stats():
    """Get statistics about scraped data"""
    scraped_dirs = glob.glob("./scraped_output_*")
    stats = {}
    
    for dir_path in scraped_dirs:
        if os.path.isdir(dir_path):
            city_name = os.path.basename(dir_path).replace("scraped_output_", "")
            md_files = glob.glob(os.path.join(dir_path, "*.md"))
            stats[city_name] = {
                "directory": dir_path,
                "markdown_files": len(md_files),
                "files": [os.path.basename(f) for f in md_files]
            }
    
    return stats

def save_vector_db(vector_db, index_path="./faiss_index"):
    """Save FAISS vector database to disk"""
    try:
        vector_db.save_local(index_path)
        print(f"FAISS index saved to {index_path}")
    except Exception as e:
        print(f"Error saving FAISS index: {str(e)}")

def load_vector_db(index_path="./faiss_index"):
    """Load FAISS vector database from disk"""
    try:
        if os.path.exists(index_path):
            vector_db = FAISS.load_local(
                index_path, 
                EMBEDDDING,
                allow_dangerous_deserialization=True
            )
            print(f"FAISS index loaded from {index_path}")
            return vector_db
    except Exception as e:
        print(f"Error loading FAISS index: {str(e)}")
    
    return None

def merge_vector_dbs(primary_db, secondary_db):
    """Merge two FAISS vector databases"""
    try:
        primary_db.merge_from(secondary_db)
        print("Successfully merged vector databases")
        return primary_db
    except Exception as e:
        print(f"Error merging vector databases: {str(e)}")
        return primary_db

async def add_documents_to_vector_db(documents, vector_db=None):
    """Add documents to the FAISS vector database"""
    if vector_db is None:
        vector_db = get_vector_db()
    
    try:
        if documents:
            # Add documents to existing FAISS index
            vector_db.add_documents(documents)
            print(f"Added {len(documents)} documents to FAISS index")
            
            # Save the updated index
            save_vector_db(vector_db)
            
        return vector_db
    except Exception as e:
        print(f"Error adding documents to vector database: {str(e)}")
        return vector_db

def search_vector_db(query, vector_db=None, k=5):
    """Search the FAISS vector database"""
    if vector_db is None:
        vector_db = get_vector_db()
    
    try:
        results = vector_db.similarity_search(query, k=k)
        return results
    except Exception as e:
        print(f"Error searching vector database: {str(e)}")
        return []

def get_vector_db_stats(vector_db=None):
    """Get statistics about the FAISS vector database"""
    if vector_db is None:
        vector_db = get_vector_db()
    
    try:
        # Get the number of vectors in the database
        if hasattr(vector_db, 'index') and hasattr(vector_db.index, 'ntotal'):
            total_vectors = vector_db.index.ntotal
        else:
            total_vectors = "Unknown"
        
        return {
            "total_vectors": total_vectors,
            "index_type": type(vector_db.index).__name__ if hasattr(vector_db, 'index') else "Unknown"
        }
    except Exception as e:
        print(f"Error getting vector database stats: {str(e)}")
        return {"error": str(e)}