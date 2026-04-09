import os
import requests
import sqlite3

import streamlit as st
import firebase_admin
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv
load_dotenv()

# --- Firebase Configuration ---
FIREBASE_WEB_API_KEY = "AIzaSyCGjILcHdc0WC7ixc3GLrK_-nO0ASc5hAU" # Your new API Key
SIGNUP_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_WEB_API_KEY}"
LOGIN_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_WEB_API_KEY}"
UPDATE_USER_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:update?key={FIREBASE_WEB_API_KEY}"

DB_FILE = "data/users.db"

def init_db():
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            name TEXT,
            phone TEXT,
            branch TEXT,
            reg_num TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            user_msg TEXT,
            bot_msg TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(email) REFERENCES users(email)
        )
    ''')
    conn.commit()
    conn.close()

# Initialize the database on startup
init_db()

def save_chat_message(email, user_msg, bot_msg):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO chats (email, user_msg, bot_msg)
            VALUES (?, ?, ?)
        ''', (email, user_msg, bot_msg))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving chat: {e}")

def get_chat_history(email):
    history = []
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT user_msg, bot_msg, timestamp FROM chats WHERE email = ? ORDER BY timestamp DESC", (email,))
        rows = cursor.fetchall()
        for row in rows:
            history.append({
                "user": row[0],
                "bot": row[1],
                "time": row[2]
            })
        conn.close()
    except Exception as e:
        print(f"Error fetching chat history: {e}")
    return history

def load_user_data(email):
    if not os.path.exists(DB_FILE):
        return {}
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT name, phone, branch, reg_num FROM users WHERE email = ?", (email,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "name": row[0],
                "phone": row[1],
                "branch": row[2],
                "reg_num": row[3]
            }
    except Exception as e:
        print(f"Error loading user data: {e}")
    return {}

def save_user_data(email, user_details):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO users (email, name, phone, branch, reg_num)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            email, 
            user_details.get("name"), 
            user_details.get("phone"), 
            user_details.get("branch"), 
            user_details.get("reg_num")
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving user data: {e}")


DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

@st.cache_resource
def get_rag_chain():
    """
    Creates and returns the RAG chain.
    The chain is cached to avoid re-creation on every user message.
    """
    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            st.error("Failed to load the vector store. Please run `create_memory_for_llm.py` first.")
            return None

        GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
        GROQ_MODEL_NAME = "llama-3.1-8b-instant"
        llm = ChatGroq(model=GROQ_MODEL_NAME, temperature=0.5, max_tokens=512, api_key=GROQ_API_KEY)

        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
        rag_chain = create_retrieval_chain(vectorstore.as_retriever(search_kwargs={'k': 3}), combine_docs_chain)
        return rag_chain
    except Exception as e:
        st.error(f"Failed to create RAG chain: {str(e)}")
        return None


def chatbot_interface():
    """
    This function contains the main chatbot UI and logic.
    It will only be called after the user has successfully logged in.
    """
    # Get user's display name, default to a generic welcome if not available
    user_info = st.session_state.get('user_info', {})
    display_name = user_info.get('displayName', 'User')
    email = user_info.get('email', '')

    # --- Header ---
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("CampConnect!")
        st.markdown(f"Welcome, {display_name}! Ask me anything about the campus.")

    with col2:
        with st.expander("👤 User Profile"):
            if email:
                profile_data = load_user_data(email)
                st.markdown(f"**Name:** {profile_data.get('name', display_name)}")
                st.markdown(f"**Email:** {email}")
                st.markdown(f"**Phone:** {profile_data.get('phone', 'N/A')}")
                st.markdown(f"**Branch:** {profile_data.get('branch', 'N/A')}")
                st.markdown(f"**Reg No:** {profile_data.get('reg_num', 'N/A')}")
            else:
                st.write("Profile data not available.")
            
            st.divider()
            if st.button("Logout"):
                st.session_state['user'] = None
                st.session_state['user_info'] = None # Clear user info on logout
                st.session_state['viewing_history'] = False
                st.rerun()

    # --- Sidebar (Chat History) ---
    with st.sidebar:
        st.header("Chat Controls")
        if st.button("View Chat History"):
            st.session_state['viewing_history'] = True
            st.rerun()

    # --- Main Area ---
    if st.session_state.get('viewing_history', False):
        st.title("Your Chat History")
        if st.button("Back to Chat"):
            st.session_state['viewing_history'] = False
            st.rerun()

        history = get_chat_history(email)
        if not history:
            st.info("No chat history found.")
        else:
            for chat in history:
                with st.expander(f"{chat['time']} - {chat['user'][:50]}..."):
                    st.write(f"**You:** {chat['user']}")
                    st.write(f"**Bot:** {chat['bot']}")
    else:
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            st.chat_message(message['role']).markdown(message['content'])

        user_prompt = st.chat_input("What do you want to know about the campus?")

        if user_prompt:
            st.chat_message('user').markdown(user_prompt)
            st.session_state.messages.append({'role':'user', 'content': user_prompt})
                    
            try: 
                rag_chain = get_rag_chain()
                if rag_chain is None:
                    return

                response=rag_chain.invoke({'input': user_prompt})
                result=response["answer"]
                st.chat_message('assistant').markdown(result)
                st.session_state.messages.append({'role':'assistant', 'content': result})
                
                if email:
                    save_chat_message(email, user_prompt, result)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

def main():
    """ 
    Main function to handle user authentication and display the appropriate interface.
    """
    if 'user' not in st.session_state:
        st.session_state['user'] = None
        st.session_state['user_info'] = None

    if not all([FIREBASE_WEB_API_KEY, SIGNUP_URL, LOGIN_URL, UPDATE_USER_URL]):
        st.error("Firebase configuration is missing. Please set FIREBASE_WEB_API_KEY in your environment variables.")
        return


    if st.session_state['user']:
        chatbot_interface()
    else:
        st.title("Welcome to CampConnect")
        choice = st.selectbox("Login/Signup", ["Login", "Sign Up"], key="login_choice")

        if choice == "Login":
            email = st.text_input("Email Address")
            password = st.text_input("Password", type="password")
            handle_login(email, password)
            st.divider()
            if st.button("Continue as Guest (Direct Login)"):
                st.session_state['user'] = {'email': 'guest@campconnect.local'}
                st.session_state['user_info'] = {'displayName': 'Guest User', 'email': 'guest@campconnect.local'}
                st.rerun()
        else:
            name = st.text_input("Full Name")
            phone = st.text_input("Phone Number")
            password = st.text_input("Password", type="password")
            email = st.text_input("Email Address")
            branch = st.text_input("Branch")
            reg_num = st.text_input("Registration Number")
            handle_signup(email, password, name, phone, branch, reg_num)

def handle_login(email, password):
    if st.button("Login"):
        if not email or not password:
            st.warning("Please enter both email and password.")
            return
        
        payload = {"email": email, "password": password, "returnSecureToken": True}

        try:
            response = requests.post(LOGIN_URL, json=payload)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            user_data = response.json()

            st.session_state['user'] = user_data
            # The displayName may not be in the login response, but we can store what we have.
            st.session_state['user_info'] = {'displayName': user_data.get('displayName', email), 'email': user_data.get('email')}
            st.success("Logged In Successfully!")
            st.rerun()
        except requests.exceptions.HTTPError:
            st.error("Failed to login: Invalid email or password.")

def handle_signup(email, password, name, phone, branch, reg_num):
    if st.button("Create my account"):
        if not all([email, password, name, phone, branch, reg_num]):
            st.warning("Please fill out all fields.")
            return

        try:
            # 1. Create user via REST API
            signup_payload = {"email": email, "password": password, "returnSecureToken": True}
            signup_response = requests.post(SIGNUP_URL, json=signup_payload)
            signup_response.raise_for_status()
            signup_data = signup_response.json()
            id_token = signup_data['idToken']

            # 2. Update the user's profile to set the display name
            update_payload = {
                "idToken": id_token,
                "displayName": name,
                "returnSecureToken": False
            }
            update_response = requests.post(UPDATE_USER_URL, json=update_payload)
            update_response.raise_for_status()

            # Save extra user data
            user_details = {
                "name": name,
                "email": email,
                "phone": phone,
                "branch": branch,
                "reg_num": reg_num
            }
            save_user_data(email, user_details)

            # Automatically log the user in after successful signup
            st.session_state['user'] = signup_data
            st.session_state['user_info'] = {'displayName': name, 'email': email}
            st.success(f"Welcome, {name}! Your account has been created and you are now logged in.")
            st.rerun()
            
        except requests.exceptions.HTTPError as e:
            error_message = "An unknown error occurred."
            if e.response and e.response.json():
                error_message = e.response.json().get('error', {}).get('message', error_message)
            st.error(f"Failed to create account: {error_message}")
        except requests.exceptions.RequestException as e:
            st.error(f"A network error occurred: {e}")


if __name__ == "__main__":
    main()