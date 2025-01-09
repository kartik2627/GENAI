import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEndpoint

# Streamlit app configuration
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜", layout="wide")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize content with advanced AI models")

# Sidebar for user inputs
with st.sidebar:
    st.markdown("## Configuration")
    hf_api_key = st.text_input("Hugging Face API Token", value="", type="password")
    repo_id = st.text_input("Model Repository ID", value="mistralai/Mistral-7B-Instruct-v0.3")
    max_length = st.number_input("Max Length", min_value=50, max_value=500, value=150, step=50)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

# Main input for URL
generic_url = st.text_input("Enter the URL (YouTube or Website)", placeholder="https://...")

# Prompt template for summarization
prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

@st.cache_data(show_spinner=False)
def load_content(url):
    """Load content from a URL (YouTube or website)."""
    if "youtube.com" in url:
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    else:
        loader = UnstructuredURLLoader(
            urls=[url],
            ssl_verify=False,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                )
            },
        )
    return loader.load()

@st.cache_data(show_spinner=False)
def summarize_content(api_key, repo, max_len, temp, docs):
    """Summarize content using Hugging Face models."""
    llm = HuggingFaceEndpoint(repo_id=repo, max_length=max_len, temperature=temp, token=api_key)
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
    return chain.run(docs)

# Summarize button logic
if st.button("Summarize"):
    if not hf_api_key.strip():
        st.error("❗ Hugging Face API token is required.")
    elif not generic_url.strip():
        st.error("❗ URL cannot be empty.")
    elif not validators.url(generic_url):
        st.error("❗ Invalid URL. Please provide a valid YouTube or website URL.")
    else:
        try:
            with st.spinner("Fetching and summarizing content..."):
                docs = load_content(generic_url)
                summary = summarize_content(hf_api_key, repo_id, max_length, temperature, docs)
                st.success("✅ Summarization completed successfully!")
                st.markdown(f"### Summary")
                st.write(summary)
        except Exception as e:
            st.error(f"🚨 An error occurred: {str(e)}")
            st.exception(e)

# Add helpful information
st.markdown("---")
st.markdown("### Instructions")
st.markdown(
    """
- **Hugging Face API Token**: Required to authenticate with the Hugging Face platform.
- **Model Repository ID**: Specify the model to use for summarization (e.g., `mistralai/Mistral-7B-Instruct-v0.3`).
- **Max Length**: Maximum number of tokens for the output.
- **Temperature**: Adjusts the randomness of the output (lower is more deterministic).
"""
)
