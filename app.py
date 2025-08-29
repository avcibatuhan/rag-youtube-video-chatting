import streamlit as st
from rag import download_audio, transcribe_audio, create_vector_store,create_conversational_chain,ask_question_with_history

st.title("YouTube Video Q&A 💬")

with st.sidebar:
    st.header("Configuration")
    model_choice = st.selectbox(
        "Choose your LLM:",
        ("Ollama", "OpenAI"),
        key="model_choice"
    )

if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'chat_history_display' not in st.session_state:
    st.session_state.chat_history_display = []

youtube_url = st.text_input("Enter YouTube URL:")

if st.button("Process Video"):
    with st.spinner("Downloading, transcribing, and indexing... This may take a moment."):
        # Call your backend functions here
        audioPath = download_audio(youtube_url)
        if audioPath is not None:
            transcribedText = transcribe_audio(audioPath)
            if transcribedText is not None and len(transcribedText) > 0:
                vector_store = create_vector_store(transcribedText)
                # st.session_state.vector_store = vector_store # Store in session state
                st.session_state.qa_chain = create_conversational_chain(vector_store,model_choice)
                st.session_state.chat_history_display = []  # Clear history for new video
                st.success("Video processed! You can now ask questions.")
            else:
                st.warning("Video text couldn't be transcribed properly.")
        else:
            st.error("Video couldn't be downloaded.")

# Display chat history
for message in st.session_state.chat_history_display:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
user_query = st.chat_input("Ask a question about the video...")

if user_query:
    if st.session_state.qa_chain:
        # Add user message to display history
        st.session_state.chat_history_display.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Get the bot's answer
        with st.spinner("Thinking..."):
            response = ask_question_with_history(st.session_state.qa_chain, user_query)
            st.session_state.chat_history_display.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
    else:
        st.error("You must process a video first before asking questions.")