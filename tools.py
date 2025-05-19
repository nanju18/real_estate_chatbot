from langchain.tools import tool
from crawl_test import LOADED_CITIES, VECTOR_DB, query_db
import os
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from utils import load_and_process_markdown, add_documents_to_vector_db

@tool
async def fetch_properties_by_city(prompt: str, city_name: str):
    """
    Description: This tool helps you find relevant properties from a VectorDB if already scraped.

    Args:
        prompt: str: A modified prompt of what the user needs.
        city_name: str: The name of the city the user wants to find properties.
    """
    city_name = city_name.lower()

    if city_name not in LOADED_CITIES:
        return f"Properties in {city_name} is not available yet. Please use scrape_properties_by_city tool to scrape the cities first and then try using this tool again."
    else:
        docs = query_db(prompt=prompt)

        msg = f"The retrieved documents for the city - {city_name.title()} are as follows -\n\n"

        for doc in docs:
            msg += doc.page_content + "\n\n"

        return msg
    
@tool
def calculate_mortgage(principal: float, interest_rate: float, loan_term_months: int, down_payment: int=0):
    """
    Description: This tool helps you to calculate the mortgage of the property.
    Args:
        principal: float: The total amount of the property.
        interest_rate: float: The annual interest rate (in percentage).
        loan_term_months: int: The loan term in months.
        down_payment: int: The down payment amount (default is 0).
    """
    # Calculate principal after down payment
    loan_amount = principal - down_payment
    monthly_rate = interest_rate / 100 / 12
    if monthly_rate == 0:  
        monthly_payment = loan_amount / loan_term_months
    else:
        monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** loan_term_months) / ((1 + monthly_rate) ** loan_term_months - 1)

    total_payment = monthly_payment * loan_term_months
    total_interest = total_payment - loan_amount
    
    return {
        "monthly_payment": round(monthly_payment, 2),
        "total_payment": round(total_payment, 2),
        "total_interest": round(total_interest, 2),
        "loan_amount": round(loan_amount, 2)
    }

@tool
async def scrape_properties_by_city(city_name: str):
    """
    Description: This tool helps you scrape properties for a particular city if it is not already available.

    Args:
        city_name: str: The name of the city.
    """
    global VECTOR_DB
    
    city_name = city_name.lower()

    print(f"Searching for properties in {city_name.title()}...")
    
    # Create the output directory if it doesn't exist
    output_dir = f"scraped_output_{city_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Configure content pruning
    prune_filter = PruningContentFilter(
        threshold=0.45,          # Prune aggressively if needed
        threshold_type="dynamic", # Use dynamic thresholding
        #min_word_threshold=5     # Skip very short nodes
    )

    # Plug the filter into the Markdown generator
    md_generator = DefaultMarkdownGenerator(content_filter=prune_filter)

    # Set up crawler configuration with the generator
    config = CrawlerRunConfig(markdown_generator=md_generator)

    # Build the URL with the extracted city name
    base_url = f"https://www.magicbricks.com/property-for-sale/residential-real-estate?bedroom=2,3&proptype=Multistorey-Apartment,Builder-Floor-Apartment,Penthouse,Studio-Apartment,Residential-House,Villa&cityName={city_name}"
    
    scraped_files = []
    async with AsyncWebCrawler() as crawler:
        for page in range(1, 6):  # Pages 1 to 5
            url = base_url
            if page > 1:
                url = f"{base_url}&page={page}"
            
            print(f"Scraping page {page} for {city_name}...")

            result = await crawler.arun(url=url, config=config)

            if result.success and result.markdown.fit_markdown.strip():
                file_path = os.path.join(output_dir, f"{city_name}_page{page}.md")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"# {city_name.title()} Properties - Page {page}\n\n")
                    f.write(result.markdown.fit_markdown)
                print(f"✅ Saved: {file_path}")
                scraped_files.append(file_path)
            else:
                print(f"⚠️ No content or error on page {page}: {result.error_message if result else 'Unknown error'}")

    # Only add to loaded cities and vector DB if we successfully scraped files
    if scraped_files:
        # Add to loaded cities if not already present
        if city_name not in LOADED_CITIES:
            LOADED_CITIES.append(city_name)

        # Load and process the scraped markdown files
        chunks = await load_and_process_markdown(f"./{output_dir}")
        
        if chunks:
            # Add city metadata to chunks
            for chunk in chunks:
                chunk.metadata['city'] = city_name
            
            # Add documents to FAISS vector database
            VECTOR_DB = await add_documents_to_vector_db(chunks, VECTOR_DB)
            
            print(f"Added {len(chunks)} chunks to FAISS vector database for {city_name}")

        return f"✅ All scraping complete for {city_name}. Files saved in '{output_dir}/' and {len(chunks)} chunks added to vector database."
    else:
        return f"❌ Failed to scrape any content for {city_name}. Please try again later."