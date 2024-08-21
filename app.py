import streamlit as stlt
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.llms import Ollama

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_community.embeddings import OllamaEmbeddings
from flask import Flask, request, jsonify
import os

TEXT_EMBEDDING_MODEL = os.getenv('TEXT_EMBEDDING_MODEL', 'nomic-embed-text')
embeddings = OllamaEmbeddings(model=TEXT_EMBEDDING_MODEL,show_progress=True)
folder_path="sauvegarde_pdfs"



# Fonction pour  EXTRACTION DU FICHIER PDF ET RETOURNE UN TEXTE
def get_pdf_text(pdf_docs):
    text = ""
    images = []
    # EXTRACTION DU TEXTE
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

#Fonction qui nous permet de decouper le texte en petit morceau egaux dont chaque morceau contient 1000 caracteres
def get_text_chunks(texte):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(texte)
    return chunks

#Fonction qui reçois le texte chunks et le transforme en vecteur grace au model de langage FastEmbedding de flask



def get_vectorstore(chunks):
    vector_store = Chroma.from_texts(texts=chunks, embedding=embeddings, persist_directory=folder_path)

    vector_store.persist()
    return vector_store

#Cette fonction initialise une chaîne de conversation en utilisant le modèle de langage "mistral" et crée également une mémoire pour stocker les échanges précédents


def get_conversation_chain(vector_store):
    llm_mistral = Ollama(model="mistral")

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_mistral,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

#Cette fonction permet d'interagir avec l'utilisateur. Les messages sont affichés dans un format particulier, avec un style différent pour les messages de l'utilisateur et ceux du BOT

def handle_userinput(user_question):
    response = stlt.session_state.conversation({'question': user_question})
    stlt.session_state.chat_history = response['chat_history']

    for i, message in enumerate(stlt.session_state.chat_history):
        if i % 2 == 0:
            stlt.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            stlt.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True) 

 
def main():
    load_dotenv()
    stlt.set_page_config(page_title="Mes debut dans l'IA",
                       page_icon=":pen:")
    stlt.write(css, unsafe_allow_html=True)

    if "conversation" not in stlt.session_state:
        stlt.session_state.conversation = None
    if "chat_history" not in stlt.session_state:
        stlt.session_state.chat_history = None

    stlt.header("Commencez avec PDF-IA")
    user_question = stlt.text_input("Posez votre question sur vos Documents:")
    

    if user_question:
        handle_userinput(user_question)

    with stlt.sidebar:
        stlt.subheader("Vos DOCUMENTS")
        fichier_pdf = stlt.file_uploader(
            "Inserer votre pdf et cliquer sur 'INSERER' pour l'inserer", accept_multiple_files=True)
        
        if stlt.button("INSERER"):
            with stlt.spinner("Chargement"):
                #TEXTE
                # get pdf text
                variable_texte = get_pdf_text(fichier_pdf)
                # get the text chunks
                text_chunks = get_text_chunks(variable_texte)
                # create vector store TEXTE AND IMAGE
                vectorstore = get_vectorstore(text_chunks,)
                print(text_chunks)

                # create conversation chain
                stlt.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == "__main__":
    main()
   
        
       
