import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
import os
from datetime import datetime
import torch
import re

# Set page config
st.set_page_config(page_title="PDF Question Answering System", page_icon="📚", layout="wide")

# Custom CSS for chat interface and sidebar
st.markdown("""
<style>
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}
.chat-message.user {
    background-color: #2b313e;
    margin-left: 20%;
}
.chat-message.bot {
    background-color: #475063;
    margin-right: 20%;
}
.chat-message .avatar {
    width: 20%;
}
.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
    color: #fff;
}
.sidebar .pdf-item {
    padding: 0.5rem;
    margin: 0.2rem 0;
    border-radius: 0.3rem;
    cursor: pointer;
    transition: background-color 0.3s;
}
.sidebar .pdf-item:hover {
    background-color: #2b313e;
}
.sidebar .pdf-item.active {
    background-color: #475063;
}
</style>
""", unsafe_allow_html=True)

# Initialize embeddings with multilingual model
embeddings = SentenceTransformerEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

# Text processing function for Vietnamese
def process_vietnamese_text(text):
    # Remove extra whitespace and normalize spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep Vietnamese diacritics and common punctuation
    text = re.sub(r'[^\w\s\u00C0-\u1EF9.,;:!?()\-]', ' ', text)
    
    # Fix spacing around punctuation
    text = re.sub(r'\s*([.,;:!?])\s*', r'\1 ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Fix spacing around parentheses
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    
    # Remove spaces before punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing spaces
    text = text.strip()
    
    return text

# Custom prompt template with RAG
template = """Bạn là một trợ lý AI thông minh, được thiết kế để trả lời câu hỏi dựa trên nội dung của tài liệu PDF. 
Hãy tuân thủ các nguyên tắc sau:
1. Chỉ trả lời dựa trên thông tin có trong các đoạn văn bản được cung cấp
2. Nếu không tìm thấy thông tin trong các đoạn văn bản, hãy nói rõ rằng bạn không có thông tin
3. Trả lời bằng tiếng Việt nếu câu hỏi bằng tiếng Việt
4. Trả lời ngắn gọn, súc tích và dễ hiểu
5. Nếu câu hỏi không liên quan đến nội dung PDF, hãy nhắc người dùng tập trung vào nội dung PDF

Các đoạn văn bản liên quan từ tài liệu:
{context}

Lịch sử hội thoại:
{chat_history}

Câu hỏi: {question}

Hãy trả lời dựa trên các đoạn văn bản trên. Nếu thông tin không đủ, hãy nói rõ.
Trả lời: """

PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "chat_history", "question"]
)

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None
if "pdfs" not in st.session_state:
    st.session_state.pdfs = {}

# Load existing PDFs and their vector stores
def load_existing_pdfs():
    # Create pdfs directory if it doesn't exist
    if not os.path.exists("pdfs"):
        os.makedirs("pdfs")
    
    # Create chroma_db directory if it doesn't exist
    if not os.path.exists("chroma_db"):
        os.makedirs("chroma_db")
    
    # Load existing PDFs
    for pdf_name in os.listdir("pdfs"):
        if pdf_name.endswith(".pdf"):
            pdf_path = os.path.join("pdfs", pdf_name)
            chroma_path = os.path.join("chroma_db", pdf_name)
            
            # Check if vector store exists
            if os.path.exists(chroma_path):
                try:
                    # Load existing vector store
                    vectorstore = Chroma(
                        persist_directory=chroma_path,
                        embedding_function=embeddings
                    )
                    
                    # Store in session state
                    st.session_state.pdfs[pdf_name] = {
                        "path": pdf_path,
                        "vectorstore": vectorstore
                    }
                except Exception as e:
                    st.error(f"Error loading vector store for {pdf_name}: {str(e)}")

# Load existing PDFs on startup
load_existing_pdfs()

# Sidebar for PDF management
with st.sidebar:
    st.title("📚 PDF Files")
    
    # File uploader in sidebar
    uploaded_file = st.file_uploader("Upload new PDF", type="pdf")
    
    if uploaded_file is not None:
        # Save uploaded file to pdfs directory
        pdf_path = os.path.join("pdfs", uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Process the PDF
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        # Process text for Vietnamese content
        for page in pages:
            page.page_content = process_vietnamese_text(page.page_content)
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(pages)
        
        try:
            # Create vector store using multilingual embeddings
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=f"chroma_db/{uploaded_file.name}"
            )
            
            # Store in session state
            st.session_state.pdfs[uploaded_file.name] = {
                "path": pdf_path,
                "vectorstore": vectorstore
            }
            st.session_state.current_pdf = uploaded_file.name
            st.success(f"PDF '{uploaded_file.name}' processed successfully!")
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
    
    # Display list of processed PDFs
    st.markdown("### Your PDFs")
    for pdf_name in st.session_state.pdfs.keys():
        if st.button(f"📄 {pdf_name}", key=pdf_name):
            st.session_state.current_pdf = pdf_name
            st.session_state.chat_history = []  # Clear chat history when switching PDFs
            st.rerun()

# Main content area
st.title("📚 PDF Question Answering System")

if st.session_state.current_pdf:
    st.write(f"Currently chatting about: **{st.session_state.current_pdf}**")
    
    # Initialize conversation if not already done
    if st.session_state.current_pdf not in st.session_state.pdfs:
        st.error("Selected PDF not found. Please upload it again.")
    else:
        if "conversation" not in st.session_state or st.session_state.conversation is None:
            try:
                # Initialize Ollama LLM
                llm = Ollama(model="llama3.2")
                
                # Create conversation chain
                memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True
                )
                
                # Create retriever with top 3 most relevant chunks
                retriever = st.session_state.pdfs[st.session_state.current_pdf]["vectorstore"].as_retriever(
                    search_kwargs={"k": 3}  # Get top 3 most relevant chunks
                )
                
                st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=retriever,
                    memory=memory,
                    combine_docs_chain_kwargs={"prompt": PROMPT}
                )
            except Exception as e:
                st.error(f"Error initializing conversation: {str(e)}")
                st.stop()
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user">
                    <div class="message">
                        <strong>You:</strong><br>
                        {message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot">
                    <div class="message">
                        <strong>Assistant:</strong><br>
                        {message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Chat input using form
        with st.form(key="chat_form", clear_on_submit=True):
            user_question = st.text_input("Ask a question about your PDF:")
            submit_button = st.form_submit_button("Send")
            
            if submit_button and user_question:
                # Process Vietnamese question
                processed_question = process_vietnamese_text(user_question)
                
                # Add user message to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_question,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                
                with st.spinner("Thinking..."):
                    try:
                        # Get relevant chunks before getting the response
                        retriever = st.session_state.pdfs[st.session_state.current_pdf]["vectorstore"].as_retriever(
                            search_kwargs={"k": 3}
                        )
                        relevant_docs = retriever.get_relevant_documents(processed_question)
                        
                        # Print relevant chunks to terminal
                        print("\n=== Các đoạn văn bản liên quan ===")
                        for i, doc in enumerate(relevant_docs, 1):
                            print(f"\nĐoạn {i}:")
                            print(doc.page_content)
                            print("-" * 80)
                        
                        # Use invoke instead of __call__
                        response = st.session_state.conversation.invoke({"question": processed_question})
                        
                        # Add bot response to chat history
                        st.session_state.chat_history.append({
                            "role": "bot",
                            "content": response["answer"],
                            "timestamp": datetime.now().strftime("%H:%M")
                        })
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")
                
                st.rerun()
else:
    st.info("Please upload a PDF file to start chatting!")
