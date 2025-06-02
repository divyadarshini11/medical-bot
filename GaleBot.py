import os
import streamlit as st

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Load environment variables
from dotenv import load_dotenv, find_dotenv
load_dotenv(dotenv_path="D:/MEDICAL_CHATBOT/data/.env", override=True)

DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id, HF_TOKEN):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512
    )
    return llm


def main():
    st.title("Ask GaleBot!")

    # Chat history and new chat feature
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_sessions' not in st.session_state:
        st.session_state.chat_sessions = []
    if 'current_session' not in st.session_state:
        st.session_state.current_session = 0

    # Sidebar for chat history and new chat
    with st.sidebar:
        st.header("Chat Sessions")
        if st.button("New Chat"):
            # Save current session to chat_sessions
            if st.session_state.messages:
                st.session_state.chat_sessions.append(st.session_state.messages)
            st.session_state.messages = []
            st.session_state.current_session = len(st.session_state.chat_sessions)
        # List previous chat sessions
        for idx, session in enumerate(st.session_state.chat_sessions):
            # Extract the first user message as the session title
            if session:
                first_user_message = next((m['content'] for m in session if m['role'] == 'user'), None)
                if first_user_message:
                    # Use the first 3 words as a keyword summary
                    session_title = ' '.join(first_user_message.split()[:3]) + ('...' if len(first_user_message.split()) > 3 else '')
                    session_help = first_user_message
                else:
                    session_title = f"Session {idx+1}"
                    session_help = "No user question."
            else:
                session_title = f"Session {idx+1}"
                session_help = "No user question."
            if st.button(session_title, help=session_help):
                st.session_state.messages = session.copy()
                st.session_state.current_session = idx

    # Display chat history
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """
        
        HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN=os.environ.get("HF_TOKEN")

        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain=RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response=qa_chain.invoke({'query':prompt})

            result=response["result"]
            source_documents=response["source_documents"]
            result_to_show=result+"\nSource Docs:\n"+str(source_documents)
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()