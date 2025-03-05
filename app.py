# =============================
# 0. INSTALL DEPENDENCIES
# =============================
#!pip install openai pillow gradio faiss-cpu langchain langchain-openai langchain-community pylatexenc requests

# Install LaTeX tools for PDF compilation
#!apt-get update
#!apt-get install -y texlive-latex-extra texlive-xetex latexmk

# =============================
# 1. MOUNT GOOGLE DRIVE (OPTIONAL)
# =============================
#from google.colab import drive
#drive.mount('/content/drive')

# =============================
# 2. IMPORT LIBRARIES
# =============================
import os
import requests
import json
import base64
import gradio as gr
from PIL import Image
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


#from langchain.vectorstores import FAISS
#from langchain.embeddings.openai import OpenAIEmbeddings

from pylatexenc.latex2text import LatexNodes2Text
from langchain.docstore.document import Document

# =============================
# 3. EXTRACT LATEX CONTENT FROM DOCUMENTS
# =============================
def extract_latex_to_documents(folder_path):
    """
    Extract LaTeX content from .tex files in the specified folder and convert it to Documents.
    """
    documents = []
    for file in os.listdir(folder_path):
        if file.endswith('.tex'):
            with open(os.path.join(folder_path, file), 'r') as f:
                latex_content = f.read()
                plain_text = LatexNodes2Text().latex_to_text(latex_content)
                documents.append(Document(page_content=plain_text, metadata={"source": file}))
    return documents

# Specify the folder path containing your LaTeX files
folder_path = "latex_files"  # Replace with your folder path
docs = extract_latex_to_documents(folder_path)

# =============================
# 4. CREATE OR LOAD FAISS INDEX
# =============================
# Load the API key from Hugging Face Secrets
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore_path = "latex_faiss_index"

if not os.path.exists(vectorstore_path):
    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local(vectorstore_path)
else:
    vectorstore = FAISS.load_local(vectorstore_path, embedding_model, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever()

# =============================
# 5. SETUP RAG PIPELINE WITH FULL LATEX PROMPT
# =============================
llm = ChatOpenAI(model="gpt-4", temperature=0.0)

# Prompt: ask the model to return a FULL LaTeX document
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a mathematics and statistics professor. You are given some context and a question.

Please provide a COMPLETE LaTeX document as your answer. The LaTeX document should compile on its own. 
It should have a preamble that includes standard packages for mathematics (like amsmath, amssymb) and be wrapped in a standard article documentclass. 
The answer should be self-contained, starting with `\\documentclass{{article}}` and ending with `\\end{{document}}`. 
Ensure that you use 1cm boarder margins. Do not write who you are in your answer. Do not include any additional commentary outside the LaTeX document.

Context: {context}

Question: {question}

Answer with a complete LaTeX document:
"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)

# =============================
# 6. HELPER FUNCTIONS
# =============================

def pdf_to_base64(pdf_path):
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()
    return base64.b64encode(pdf_data).decode('utf-8')


def handle_rag_query(user_query):
    """
    Query the RAG system, get a full LaTeX doc, compile it to PDF, and return embedded PDF as HTML.
    """
    if not user_query.strip():
        return "Please enter a query before submitting.", ""

    try:
        response = qa_chain.invoke({"query": user_query})
        answer_latex = response["result"]  # This is the full LaTeX document

        # Write the LaTeX code to a file
        with open("output.tex", "w") as f:
            f.write(answer_latex)

        # Compile the LaTeX file into a PDF
        compilation = os.system("latexmk -pdf output.tex")
        if compilation != 0:
            return "Error compiling LaTeX document. Please check the LaTeX syntax.", ""

        # Convert PDF to base64
        pdf_base64 = pdf_to_base64("output.pdf")
        html_pdf = f"""
        <embed src="data:application/pdf;base64,{pdf_base64}"
               width="100%" height="600px" type="application/pdf" />
        """

        # Return only the PDF HTML (no source documents)
        return html_pdf, ""
    except Exception as e:
        return f"An error occurred: {str(e)}", ""

def handle_query_with_loader(query):
    """
    Show a loader message, then show PDF.
    """
    if not query.strip():
        yield "Please enter a query before submitting.", ""
        return

    yield "Processing... Please wait. ⏳", ""
    pdf_html, srcs = handle_rag_query(query)
    yield pdf_html, srcs

def encode_image(image_path):
    """
    Encode image to base64 string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def interpret_image_with_gpt4_vision(image_path, user_query="Extract all text from this image and provide them in LaTeX format. It should have a preamble that includes standard packages for mathematics (like amsmath, amssymb) and be wrapped in a standard article documentclass. The answer should be self-contained, starting with `\\documentclass{{article}}` and ending with `\\end{{document}}`. Ensure that you use 1cm boarder margins. Do not include any additional commentary outside the LaTeX document. Do not start and end with '```latex' and '```'."):
    """
    Use GPT-4o to interpret the image and extract text.
    """
    try:
        # Encode image as base64
        with open(image_path, "rb") as img_f:
            image_data = base64.b64encode(img_f.read()).decode('utf-8')

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
        }

        # Determine image format
        try:
            image_format = Image.open(image_path).format.lower()
            if image_format not in ['jpeg', 'jpg', 'png', 'webp', 'gif']:
                return "Unsupported image format. Please upload a PNG, JPEG, WEBP, or GIF image."
        except Exception as e:
            return f"Error determining image format: {str(e)}"

        payload = {
            "model": "gpt-4o-mini",  # or "gpt-4o" depending on your access
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_query},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_format};base64,{image_data}",
                                "detail": "low"  # You can set to "auto", "low" or "high" based on your needs
                            },
                        }
                    ]
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.0
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=60)

        if response.status_code != 200:
            return f"OpenAI API returned an error: {response.status_code} - {response.text}"

        response_data = response.json()

        if "choices" in response_data and len(response_data["choices"]) > 0:
            interpretation = response_data["choices"][0]["message"]["content"]
            # Basic validation of response
            if not interpretation or "no text detected" in interpretation.lower():
                return "No text detected in the image. Please try uploading a clearer image."
            return interpretation.strip()
        else:
            return "Could not interpret the image."

    except requests.exceptions.Timeout:
        return "The request to OpenAI timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return f"An error occurred while making the request: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def extract_text_from_image(image_path):
    """
    Wrapper function to interpret image and handle errors.
    """
    interpretation = interpret_image_with_gpt4_vision(image_path)
    return interpretation

# =============================
# 7. GRADIO INTERFACE WITH TABS
# =============================

theme = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="orange",
    font="IBM Plex Sans"
)

# Custom CSS to center the title
custom_css = """
#title-markdown {
    text-align: center;
}
"""

def compile_latex_to_pdf(latex_code):
    """
    Compile LaTeX code to PDF and return the base64 encoded PDF.
    """
    try:
        # Write the LaTeX code to a file
        with open("output.tex", "w") as f:
            f.write(latex_code)

        # Compile the LaTeX file into a PDF
        compilation = os.system("latexmk -pdf output.tex")
        if compilation != 0:
            return "Error compiling LaTeX document. Please check the LaTeX syntax.", ""

        # Convert PDF to base64
        pdf_base64 = pdf_to_base64("output.pdf")
        html_pdf = f"""
        <embed src="data:application/pdf;base64,{pdf_base64}"
               width="100%" height="600px" type="application/pdf" />
        """

        return html_pdf, ""
    except Exception as e:
        return f"An error occurred: {str(e)}", ""

with gr.Blocks(theme=theme, css=custom_css) as interface:
    gr.Markdown("## Statistics Study Assistant", elem_id="title-markdown")

    with gr.Tabs():
        # Tab 1: RAG Query
        with gr.Tab("Ask a Statistics Question"):
            gr.Markdown("### Ask an Introductory Statistics Question", elem_id="instruction-markdown")
            gr.Markdown(
                "Enter your question below. The assistant will produce a fully LaTeX-formatted PDF as the answer."
            )
            with gr.Row():
                with gr.Column():
                    text_query = gr.Textbox(lines=2, placeholder="e.g. What is the formula for standard deviation?", label="Question")
                    submit_button = gr.Button("Submit Question")
                    status_text = gr.Markdown("")
                    answer_output = gr.HTML(label="PDF Answer")
                    sources_output = gr.Markdown(label="Sources")

                    submit_button.click(
                        fn=handle_query_with_loader,
                        inputs=text_query,
                        outputs=[answer_output, sources_output]
                    )

        # Tab 2: Image Upload + Extraction
        with gr.Tab("Extract Mathematical/Statistical Text from Images"):
            gr.Markdown("### Upload an Image to Extract LaTeX-Formatted Text", elem_id="instruction-markdown")
            gr.Markdown(
                "Upload an image containing mathematical expressions. The system will extract text and provide it in an editable LaTeX format."
            )
            with gr.Row():
                with gr.Column():
                    image_input = gr.File(label="Upload an image (PNG, JPG, GIF, or WEBP)")
                    extract_button = gr.Button("Extract Text")
                    latex_code_input = gr.Textbox(label="LaTeX Code", interactive=True, lines=10)
                    recompile_button = gr.Button("Recompile LaTeX")
                    pdf_output = gr.HTML(label="PDF Output")

                    def extract_and_display(file):
                        if file is None:
                            return "Please upload an image before extracting text.", ""
                        # Check file extension
                        allowed_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.gif']
                        _, ext = os.path.splitext(file.name)
                        if ext.lower() not in allowed_extensions:
                            return "Unsupported file type. Please upload a PNG, JPEG, WEBP, or GIF image.", ""
                        image_path = file.name
                        latex_code = extract_text_from_image(image_path)
                        pdf_html, _ = compile_latex_to_pdf(latex_code)
                        return latex_code, pdf_html

                    def recompile_latex(latex_code):
                        pdf_html, _ = compile_latex_to_pdf(latex_code)
                        return pdf_html

                    extract_button.click(
                        fn=extract_and_display,
                        inputs=image_input,
                        outputs=[latex_code_input, pdf_output]
                    )

                    recompile_button.click(
                        fn=recompile_latex,
                        inputs=latex_code_input,
                        outputs=pdf_output
                    )

        # Tab 3: About Me
        with gr.Tab("About"):
            gr.Markdown("### About The Creator", elem_id="about-me-markdown")
            gr.Markdown(
                """
                **Creator:** Samuel Ewusi Dadzie  
                **Affiliation:** PhD Student in Statistics, University of Georgia  
                **Project:** This app is my submission for Generative AI Competition 2.0, sponsored by the Office of Instruction and the Center for Teaching and Learning at the University of Georgia.
                """
            )

            gr.Markdown("### Technical Details", elem_id="technical-details-markdown")
            gr.Markdown(
                """
                At the core of this app is a **Retrieval-Augmented Generation (RAG) pipeline**, built on OpenAI’s GPT-4. The RAG references a carefully curated knowledge base derived from the creator’s (Samuel Ewusi Dadzie) introductory statistics notes (in LaTeX format).

                The **image processing component** harnesses GPT-4o Mini’s vision capabilities, enabling it to accurately recognize mathematical symbols and formulas within images in PNG, JPG, GIF, or WEBP formats.
                """
            )

interface.launch(share=False)