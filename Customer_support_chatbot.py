import nltk
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.document_loaders import SeleniumURLLoader
from langchain import PromptTemplate

# Ensure NLTK data is downloaded
nltk.download('punkt')

# URLs to scrape content from
urls = [
    'https://beebom.com/what-is-nft-explained/',
    'https://beebom.com/how-delete-spotify-account/',
    'https://beebom.com/how-download-gif-twitter/',
    'https://beebom.com/how-use-chatgpt-linux-terminal/',
    'https://beebom.com/how-delete-spotify-account/',
    'https://beebom.com/how-save-instagram-story-with-music/',
    'https://beebom.com/how-install-pip-windows/',
    'https://beebom.com/how-check-disk-usage-linux/'
]

# Load documents using SeleniumURLLoader
loader = SeleniumURLLoader(urls=urls)
docs = loader.load()  # Assuming this returns a list of Document objects or similar structures

# Initialize OpenAIEmbeddings for generating embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Initialize Deep Lake with your organization ID and dataset name
my_activeloop_org_id = "mightyaviator"  # Replace with your actual Activeloop organization ID
my_activeloop_dataset_name = "langchain_course_customer_support"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# Assuming docs are already in the correct format, directly add them to Deep Lake
db.add_documents(docs)  # This assumes docs is a list of Document objects or similar structures directly usable by Deep Lake

# Example query to test the system
query = "how to check disk usage in linux?"
docs = db.similarity_search(query)
if docs:
    print(docs[0].page_content)  # Adjusted to use dot notation for Document objects
else:
    print("No documents found for the query.")

# Craft a prompt for GPT-3
template = """You are an exceptional customer support chatbot that gently answer questions.

You know the following context information.

{chunks_formatted}

Answer to the following question from a customer. Use only information from the previous context information. Do not invent stuff.

Question: {query}

Answer:"""

prompt = PromptTemplate(
    input_variables=["chunks_formatted", "query"],
    template=template,
)

# Retrieve relevant chunks for the query and truncate to fit within token limits
retrieved_chunks = [doc.page_content for doc in docs]  # Adjusted to use dot notation
max_tokens = 4096 - len(query) - 256  # Subtracting the query and a buffer for the completion
chunks_formatted = "\n\n".join(retrieved_chunks[:max_tokens])  # Simplified truncation, consider more sophisticated summarization

# Ensure the prompt is within the token limit
if len(chunks_formatted) > max_tokens:
    chunks_formatted = chunks_formatted[:max_tokens]  # Simplified truncation, adjust as necessary

prompt_formatted = prompt.format(chunks_formatted=chunks_formatted, query=query)

# Generate answer with GPT-3
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
answer = llm(prompt_formatted)
print(answer)
