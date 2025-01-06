import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# Streamlit App Configuration
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

# Sidebar for API Key Input
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

# Input Field for URL
generic_url = st.text_input("Enter a YouTube or Website URL", label_visibility="collapsed")

# Initialize the Groq API with the model
llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

# Prompt Template
prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize the Content"):
    if not groq_api_key.strip():
        st.error("Please provide the Groq API Key.")
    elif not generic_url.strip():
        st.error("Please enter a URL to summarize.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. Supported formats include YouTube videos or website links.")
    else:
        try:
            with st.spinner("Processing the URL..."):
                # Initialize the LLM with the API key
                llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

                # Load content based on the URL type
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": (
                                "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/116.0.0.0 Safari/537.36"
                            )
                        },
                    )
                docs = loader.load()

                # Summarize the loaded content
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                # Display the summary
                st.success(output_summary)
        except Exception as e:
            st.error(f"An error occurred: {e}")
