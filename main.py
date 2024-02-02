import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GooglePalm
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import base64
import os
import tempfile

load_dotenv()
VectorStore = None
embeddings = HuggingFaceEmbeddings()
llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=1)



def displayPDF_in_sidebar(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.sidebar.markdown(pdf_display, unsafe_allow_html=True)


with st.sidebar:
    st.title("LLM Chat App")
    st.markdown('''This app is an LLM-powered chatbot.''')

    add_vertical_space(1)
    st.write('Developed by [Sai Kumar Adapa](https://www.linkedin.com/in/sai-kumar-adapa-5a16b2228/)')

    pdf_sidebar = st.sidebar.file_uploader("Upload PDF for sidebar", type='pdf')
    if pdf_sidebar:
        temp_pdf = tempfile.NamedTemporaryFile(delete=False)
        temp_pdf.write(pdf_sidebar.read())
        temp_pdf.close()
        displayPDF_in_sidebar(temp_pdf.name)


def main():
    st.header("Chat with PDF...")
    main_placeholder = st.empty()
    main_placeholder.markdown('Say Hi!!...👋👋 to &nbsp;&nbsp;&nbsp;&nbsp;[Sai Kumar Adapa](https://www.linkedin.com/in/sai-kumar-adapa-5a16b2228/)')
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    if pdf:
        temp_pdf = tempfile.NamedTemporaryFile(delete=False)
        temp_pdf.write(pdf.read())
        temp_pdf.close()

        storeName = temp_pdf.name[:-4]
        if os.path.exists(storeName):
            VectorStore = FAISS.load_local(folder_path=storeName, embeddings=embeddings)
            main_placeholder.text("Data Base Already Existed...✅✅✅")
        else:
            main_placeholder.text("Data Base Creation...Started...✅✅✅")
            pdf_reader = PdfReader(temp_pdf.name)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)

            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            VectorStore.save_local(storeName)

            main_placeholder.text("Data Base Creation...Done...✅✅✅")

        query = st.text_input("Ask a question about your PDF file")
        if query:
            main_placeholder.text("I am Thinking...😇😇😇")
            docs = VectorStore.similarity_search(query, k=2)

            chain = load_qa_chain(llm=llm, chain_type='stuff')
            response = chain.run({'input_documents': docs, 'question': query})
            main_placeholder.text("I got it...✅✅✅")
            st.write(response)


if __name__ == '__main__':
    main()
