import nltk
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.document_loaders import SeleniumURLLoader
from langchain import PromptTemplate

nltk.download('punkt')

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

loader = SeleniumURLLoader(urls=urls)
docs = loader.load()

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

my_activeloop_org_id = "mightyaviator"
my_activeloop_dataset_name = "langchain_course_customer_support"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

db.add_documents(docs)

query = "how to check disk usage in linux?"
docs = db.similarity_search(query)
if docs:
    print(docs[0].page_content)
else:
    print("No documents found for the query.")

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

retrieved_chunks = [doc.page_content for doc in docs]
max_tokens = 4096 - len(query) - 256
chunks_formatted = "\n\n".join(retrieved_chunks[:max_tokens])

if len(chunks_formatted) > max_tokens:
    chunks_formatted = chunks_formatted[:max_tokens]

prompt_formatted = prompt.format(chunks_formatted=chunks_formatted, query=query)

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
answer = llm(prompt_formatted)
print(answer)
