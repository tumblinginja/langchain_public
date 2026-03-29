import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain   # ✅ new import
from langchain.docstore.document import Document

# Step 1: Instantiate LLM and Pinecone
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo",
    temperature=0.9
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Step 2: Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Step 3: Check if the index already has your data
index_name = str(input('Which index did you store your data?\n'))

if index_name not in pc.list_indexes().names():
    create_new_index_question = str.upper(input("Index doesn't exist. Do you want to create a new one? Y/N\n"))
    if create_new_index_question == 'Y':
        print(f'Creating index {index_name}')
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ) 
        )
        print('Index created!')
else:
    print(f'Index {index_name} already exists!')

index = pc.Index(index_name)
stats = index.describe_index_stats()
document_name = str(input('Which document are you referring to?\n'))

if stats.total_vector_count == 0:
    # First run → upload documents
    print("Index is empty. Uploading embeddings...")

    with open(document_name + '.txt', encoding="utf-8") as f:
        text_document = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # larger chunk size is often better for semantic meaning
        chunk_overlap=50,
        length_function=len
    )

    # Add metadata for better traceability
    chunks = text_splitter.create_documents(
        [text_document],
        metadatas=[{"source": document_name}]
    )
    print(f"Created {len(chunks)} chunks")

    vector_store = PineconeVectorStore.from_documents(
        chunks, embedding=embeddings, index_name=index_name
    )
else:
    # Subsequent runs → skip upload
    print(f"Index already has {stats.total_vector_count} vectors. Skipping upload.")
    vector_store = PineconeVectorStore(
        index=index, embedding=embeddings, text_key="text"
    )

# Step 4: Choose between Q&A or Summarization
mode = str.lower(input("Do you want 'qa' or 'summary' mode?\n"))

if mode == "qa":
    from langchain.prompts import ChatPromptTemplate
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Use the provided context to answer the question.

        - If the answer cannot be found in the context, say "I don't know."

        Context:
        {context}

        Question:
        {input}

        Answer:"""
    )

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, combine_docs_chain)

    while True:
        query = input("Ask a question: ")
        if query.lower() in ["exit", "quit"]:
            break

        result = chain.invoke({"input": query})
        print("\nAnswer:", result["answer"], "\n")

elif mode == "summary":
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Fetch top docs for summarization
    docs = retriever.invoke(" ")   # blank query just retrieves top docs

    # Load summarize chain (map_reduce handles long docs better)
    summarize_chain = load_summarize_chain(llm, chain_type="map_reduce")

    summary = summarize_chain.invoke(docs)
    print("\nDocument Summary:\n", summary['output_text'], "\n")

else:
    print("Invalid choice. Please restart and pick either 'qa' or 'summary'.")
