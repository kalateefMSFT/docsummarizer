import tempfile
import os
import openai
import PyPDF2

from io import StringIO

from langchain.chat_models import AzureChatOpenAI

from utils import doc_to_text, token_counter

# Load environment variables
load_dotenv()

# Configure OpenAI API
openai.api_type = "azure"
openai.api_version = "2023-05-15" # os.getenv('OPENAI_API_VERSION')
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")

os.environ["LANGCHAIN_HANDLER"] = "langchain"

def pdf_to_text(pdf_file):
    """
    Convert a PDF file to a string of text.

    :param pdf_file: The PDF file to convert.

    :return: A string of text.
    """
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = StringIO()
    for i in range(len(pdf_reader.pages)):
        p = pdf_reader.pages[i]
        text.write(p.extract_text())
    return text.getvalue().encode('utf-8')


def token_limit(doc, maximum=200000):
    """
    Check if a document has more tokens than a specified maximum.

    :param doc: The langchain Document object to check.

    :param maximum: The maximum number of tokens allowed.

    :return: True if the document has less than the maximum number of tokens, False otherwise.
    """
    text = doc_to_text(doc)
    count = token_counter(text)
    print(count)
    if count > maximum:
        return False
    return True


def token_minimum(doc, minimum=2000):
    """
    Check if a document has more tokens than a specified minimum.

    :param doc: The langchain Document object to check.

    :param minimum: The minimum number of tokens allowed.

    :return: True if the document has more than the minimum number of tokens, False otherwise.
    """
    text = doc_to_text(doc)
    count = token_counter(text)
    if count < minimum:
        return False
    return True


def create_temp_file(uploaded_file):
    """
    Create a temporary file from an uploaded file.

    :param uploaded_file: The uploaded file to create a temporary file from.

    :return: The path to the temporary file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp_file:
        if uploaded_file.type == 'application/pdf':
            temp_file.write(pdf_to_text(uploaded_file))
        else:
            temp_file.write(uploaded_file.getvalue())
    return temp_file.name


def create_chat_model():
    """
    Create a chat model ensuring that the token limit of the overall summary is not exceeded - GPT-4 has a higher token limit.

    :param api_key: The OpenAI API key to use for the chat model.

    :param use_gpt_4: Whether to use GPT-4 or not.

    :return: A chat model.
    """
    return AzureChatOpenAI(temperature=0, max_tokens=2000, deployment_name='gpt-3.5-turbo')


