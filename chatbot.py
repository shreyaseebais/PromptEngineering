import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

# openAIkey
OPENAI_API_KEY = "pasteKeyHere"

# Upload pdf files 
st.header("My first chatbot")

with st.sidebar : 
  st.title("Your documents")
  file = st.file_uploader("Upload a pdf file and start asking questions.", type ="pdf")

# Extract the text
if file is not None :
  pdf_reader = PdfReader(file)
  text=""
  for page in pdf_reader.pages:
    text+=page.extract_text()
    #st.write(text)
  
# break it into chunks
text_splitter = RecursiveCharacterTextSplitter(
  separators="\n",
  chunk_size=1000,
  chunk_overlap=150,
  length_function=len
)


# generating embedding 
embeddings = openAIEmbeddings(openai_api_key= OPEN_AI_KEY)

# creating vector store - FAISS 
vector_store = FAISS.from_texts(chunks, embddings)

# get user question 
user_question = st.text_input("Type your question here ")

# do similarity search 
if user_question :
  match = vector_store.similarity_search(user_question)
  # st.write(match)

  # define the LLM 
  llm = ChatOpenAI(
    openai_api_key = OPEN_API_KEY, 
    temperature = 0,
    max_tokens = 1000,
    model_name = "gpt-3.5-turbo"
  )

  # output results
  # chain -> take the question, get relevant document, pass it to LLM, generate the output
  chain  = load_qa_chain(llm, chain_type="stuff")
  response = chain.run(input_documents = match, question  =user_question)
  st.write(response)  
