from pytubefix import YouTube
import os
import whisper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

def download_audio(youtube_url, output_path="."):
    try:
        yt = YouTube(youtube_url)
        audio_stream = yt.streams.filter(only_audio=True).first()

        if not audio_stream:
            print("No audio-only stream found.")
            return None

        print(f"Downloading audio for: {yt.title}")
        output_file = audio_stream.download(output_path=output_path)

        base, ext = os.path.splitext(output_file)
        new_file = base + '.mp3'
        # Check if the new file already exists
        if os.path.exists(new_file):
            os.remove(new_file)
        os.rename(output_file, new_file)

        print(f"Download complete. File saved as: {new_file}")
        return new_file

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def transcribe_audio(audio_path):
    print("Loading Whisper model...")
    model = whisper.load_model("base")

    print(f"Transcribing {audio_path}...")
    result = model.transcribe(audio=audio_path)

    print("Transcription complete.")
    return result['text']

def create_vector_store(text_data):
    # 1. Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text_data)

    # 2. Create embeddings
    embeddings = OpenAIEmbeddings()

    # 3. Create a vector store (FAISS) from the chunks
    print("Creating vector store...")
    vector_store = FAISS.from_texts(chunks, embeddings)

    print("Vector store created successfully.")
    return vector_store

def ask_question(vector_store, query):
    print(f"Searching for relevant documents for the query: '{query}'")
    # Find relevant documents (chunks) in the vector store
    docs = vector_store.similarity_search(query)

    if not docs:
        return "I couldn't find any relevant information in the video to answer that question."

    print("Loading OpenAI LLM and QA Chain...")
    # llm = OpenAI(temperature=0)
    llm = Ollama(model="gpt-oss:20b")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        verbose=True,
    )
    #chain = load_qa_chain(llm, chain_type="stuff")
    print("Generating answer...")

    result = qa_chain({"question": query})
    return result['answer']

    # Run the chain with the documents and the question
    # with get_openai_callback() as cb:
        # response = chain.run(input_documents=docs, question=query)
        # print("\n--- Token Usage ---")
        # print(cb)
        # print("-------------------\n")
    # return response

def create_conversational_chain(vector_store, model_choice):
    """
    Creates and returns a conversational retrieval chain with memory.
    """
    print(f"Initializing conversational chain with model: {model_choice}")

    llm = None
    if model_choice == "Ollama":
        # Initialize the local Ollama model
        llm = Ollama(model="llama3")  # Or any other model you have, like "gpt-oss:20b"

    elif model_choice == "OpenAI":
        # The API key should be set as an environment variable before calling this function
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key is not set in environment variables.")
        # Initialize the OpenAI model
        llm = ChatOpenAI(temperature=0, model="gpt-5-mini",api_key=os.getenv("OPENAI_API_KEY"))

    else:
        raise ValueError(f"Unsupported model choice: {model_choice}. Please choose 'Ollama' or 'OpenAI'.")

    # Define the memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    # Create the chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        verbose=True  # Set to True to see the chain's thinking process
    )

    return qa_chain

def ask_question_with_history(chain, query):
    """
    Asks a question using the existing conversational chain.
    """
    print(f"Submitting query to the chain: '{query}'")

    # The chain automatically handles retrieving docs, adding to memory, and calling the LLM
    result = chain({"question": query})

    # The result is a dictionary. The answer is in the 'answer' key.
    return result['answer']