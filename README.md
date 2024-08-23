
# OmniAssist

OmniAssist is a sophisticated customer support chatbot application designed to assist users with their queries by leveraging advanced natural language processing (NLP) techniques and web scraping. Built using Streamlit for the frontend interface, this chatbot integrates with OpenAI's GPT-4 Turbo model for generating human-like responses and utilizes DeepLake for document storage and retrieval. It aims to provide accurate and helpful answers to customer inquiries by searching through a database of documents scraped from specified URLs.



## Features

- Dynamic Document Retrieval: Automatically fetches and processes documents from specified URLs, ensuring the chatbot has access to the most current information.

- Customizable Knowledge Base: Allows customization through the specification of URLs in a text file, enabling the chatbot to adapt to various domains and information sources.

- Easily Tailored for Any Industry: This chatbot can be customized to fit any business or website. It's like having a personal assistant that speaks your company's language, ready to help customers with whatever they need.


## Features

- Dynamic Document Retrieval: Automatically fetches and processes documents from specified URLs, ensuring the chatbot has access to the most current information.

- Customizable Knowledge Base: Allows customization through the specification of URLs in a text file, enabling the chatbot to adapt to various domains and information sources.

- Easily Tailored for Any Industry: This chatbot can be customized to fit any business or website. It's like having a personal assistant that speaks your company's language, ready to help customers with whatever they need.


## Getting Started

Clone the project

```bash
  git clone https://link-to-project
```

Install the necessary packages

```bash
  pip install -r requirements.txt
```

Download required NLTK data:

```bash
  python -m nltk.downloader punkt
```

Running the Chatbot

```bash
  streamlit run app.py
```


## API Reference

Required API Keys :

- OpenAI API Key
- Deeplake API Key 

