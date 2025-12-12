# Langchain-rag-with-csv
# 📄 RAG Question Answering System with PDF Support

A Retrieval-Augmented Generation (RAG) pipeline designed to ingest PDF documents, index their content using vector embeddings, and perform context-aware question answering using a local Large Language Model (LLM).

---

## 📂 Dataset

The system is designed to process arbitrary PDF documents. For this specific implementation, the following data was used as the knowledge base:

* **Source:** Uploaded PDF Documents
* **Description:**
    * **COVID-19 Humanitarian Report:** Data on global cases, vaccination rates, and economic impacts in Q2 2022.
    * **USAID Contact List:** Personnel coordination tables for the Africa Region.
    * **Estuary Newsletter (2007):** Environmental reports, specifically detailing the Cosco Busan oil spill in San Francisco Bay.

---

## 🛠 Technologies Used

* **Language:** Python
* **Orchestration:** `LangChain` (Community, Core, HuggingFace)
* **LLM:** `Google Gemma-2b-it` (via Hugging Face Transformers)
* **Vector Store:** `FAISS` (Facebook AI Similarity Search)
* **Embeddings:** `HuggingFaceBgeEmbeddings` (sentence-transformers/all-MiniLM-L6-v2)
* **Document Loading:** `PyPDF`

---

## ✨ Key Features

✅ **Document Ingestion**
   * Utilizes `PyPDFDirectoryLoader` to scan directories and load multiple PDF files simultaneously.

✅ **Text Preprocessing**
   * **Splitting:** Implements `RecursiveCharacterTextSplitter` with a chunk size of 1000 and overlap of 200 to maintain context across boundaries.

✅ **Vector Embeddings**
   * Converts text chunks into dense vector representations using the `sentence-transformers/all-MiniLM-L6-v2` model.

✅ **Similarity Search**
   * Uses **FAISS** to create an efficient index for fast retrieval of relevant document chunks based on user queries.

✅ **Local LLM Integration**
   * Loads the **Gemma-2b-it** model locally using a `HuggingFacePipeline`.
   * **Configuration:** `temperature=0.2`, `max_new_tokens=512`.

✅ **RAG Chain**
   * Combines the retriever and the LLM into a `RunnableParallel` chain.
   * Injects retrieved context dynamically into the prompt to ground the LLM's answers in factual data.

---

## 📊 Results & Usage Example

The system successfully retrieved specific historical data from the "Estuary" newsletter PDF to answer natural language questions.

* **Query:** *"What caused the oil spill in the San Francisco Bay on November 7, 2007?"*
* **Retrieved Context:** The system identified the text describing the cargo ship *Cosco Busan* running into the Bay Bridge support post.
* **Generated Answer:** The model correctly identified that the ship gashed its fuel tank, sending 58,000 gallons of oil into the bay.

---

## ⚙️ Installation & Setup

1.  **Install Dependencies:**
    ```bash
    pip install langchain-community langchain-huggingface pypdf faiss-cpu accelerate
    ```

2.  **Hugging Face Authentication:**
    You must have a Hugging Face token with access to the Gemma model.
    ```python
    from huggingface_hub import login
    login(token="YOUR_HUGGINGFACE_TOKEN")
    ```

3.  **Run the Pipeline:**
    Load your PDFs into the `data/` directory and execute the Jupyter Notebook cells to build the index and query the model.
