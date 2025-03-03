# 🏠 Property Information RAG Assistant

A powerful property information assistant built with Python, OpenAI, and LanceDB that uses Retrieval Augmented Generation (RAG) to answer questions about property listings.

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green.svg)](https://openai.com/)
[![LanceDB](https://img.shields.io/badge/LanceDB-0.20.0-orange.svg)](https://lancedb.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42.2-red.svg)](https://streamlit.io/)
[![Pydantic](https://img.shields.io/badge/Pydantic-2.10.6-purple.svg)](https://pydantic.dev/)

## 🚀 Features

- **Vector Search**: Fast and accurate semantic search using LanceDB and OpenAI embeddings
- **Conversational UI**: Clean user interface built with Streamlit
- **Rich Property Information**: Query details about properties, including:
  - Pet policies (allowed pets, fees, weight limits)
  - Location information
  - Amenities and features
  - Pricing details
- **Intelligent Query Understanding**: Converts natural language questions into optimized search queries
- **Well-formatted Responses**: Clear, structured answers with relevant property details

## 📋 Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Credits](#credits)
- [License](#license)

## 🔧 Installation

1. Clone the repository:
   ```bash
   git clone http://www.github.com/pakagronglb/python-rag-agent-pydantic-ai
   cd python-rag-agent-pantic-ai
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your OpenAI API key in a `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## 💻 Usage

1. Run the setup script to initialize the knowledge base:
   ```bash
   python lancedb_setup.py
   ```

2. Launch the Streamlit web interface:
   ```bash
   python -m streamlit run app.py
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:8501
   ```

4. Enter your property-related questions in the input field and click "Search"

## 🏗️ Project Structure

- `app.py`: Streamlit web interface
- `agent_run.py`: Command-line version of the query agent
- `lancedb_setup.py`: Knowledge base setup and vector database operations
- `knowledge-file/`: Directory containing markdown property information files

## ⚙️ How It Works

1. **Data Ingestion**: Property information is stored as markdown files in the `knowledge-file` directory
2. **Vector Embedding**: The text is chunked and embedded using OpenAI's text-embedding-3-small model
3. **Query Processing**: User queries are converted to optimized search queries using an LLM
4. **Vector Search**: LanceDB retrieves the most semantically similar documents
5. **Response Generation**: An LLM generates a well-formatted response based on the retrieved context

## 🙏 Credits

This project was inspired by [Jie Jenn's YouTube tutorial](https://www.youtube.com/watch?v=P212vYt6Xo8) on building RAG applications with Python. His clear explanations and practical approach were instrumental in creating this application.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

🏠 Property Information Assistant
    [Header]

Ask a Question
    [Input Form with background]
    [Full-width Search button]

📝 Response
    [Response area with background]
