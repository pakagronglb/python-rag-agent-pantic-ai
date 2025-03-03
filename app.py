import streamlit as st
import lancedb
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from lancedb_setup import setup_lancedb, retrieve_similar_docs

# Initialize agents and database connection
@st.cache_resource
def init_agents_and_db():
    db_path = "./db"
    table_name = "knowledge"
    db = lancedb.connect(db_path)
    knowledge_table = db.open_table(table_name) if db.table_names() else setup_lancedb()

    knowledge_query_agent = Agent(
        name='Knowledge Query Agent',
        model=OpenAIModel('gpt-4o-mini'),
        deps_type=str,
        result_type=str,
        system_prompt='Convert the user query into a clear and concise search query for finding relevant property information.',
    )

    main_agent = Agent(
        name='Main Agent',
        model=OpenAIModel('gpt-4o-mini'),
        system_prompt='''You are a helpful real estate assistant. When providing information about properties:
1. Present information in a clear, organized manner using markdown formatting
2. List specific details about properties when available (location, amenities, policies, etc.)
3. Include important numbers (prices, fees, limits) when present
4. If information is not available in the context, clearly state that
5. Keep responses concise but informative''',
    )
    
    return knowledge_table, knowledge_query_agent, main_agent

def search_properties(query: str, knowledge_table, knowledge_query_agent, main_agent):
    if not query.strip():
        st.warning("Please enter a question")
        return

    with st.spinner("🔍 Searching..."):
        try:
            # Get search query from knowledge agent
            res = knowledge_query_agent.run_sync(query)
            knowledge_query = res.data

            # Retrieve relevant documents
            retrieved_docs = retrieve_similar_docs(knowledge_table, knowledge_query, limit=5)

            # Build context from retrieved documents
            knowledge_context = ""
            for doc in retrieved_docs:
                knowledge_context += doc.get('text', '') + "\n\n"

            if not knowledge_context.strip():
                st.error("❌ I couldn't find any relevant information about that in our database.")
                return

            # Generate response using main agent
            prompt = f"""Context:
{knowledge_context}

User Question:
{query}

Please provide a clear and organized answer based on the above context. If specific information is not available in the context, please state that clearly."""

            response = main_agent.run_sync(prompt)
            return response.data

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            return None

def main():
    st.set_page_config(
        page_title="Property Information Assistant",
        page_icon="🏠",
        layout="wide"
    )

    # Initialize agents and database
    knowledge_table, knowledge_query_agent, main_agent = init_agents_and_db()

    # Header
    st.title("🏠 Property Information Assistant")
    st.markdown("Ask me anything about the properties in our database!")

    # Create two columns
    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("Ask a Question")
        # Initialize session state for query if it doesn't exist
        if 'query' not in st.session_state:
            st.session_state.query = ""
        
        # Text input for query
        query = st.text_input("Type your question here", key="query")
        
        # Search button
        if st.button("Search", type="primary"):
            if query:
                response = search_properties(query, knowledge_table, knowledge_query_agent, main_agent)
                if response:
                    st.session_state.response = response
            else:
                st.warning("Please enter a question")

    with col2:
        st.subheader("📝 Response")
        if 'response' in st.session_state:
            st.markdown(st.session_state.response)
        else:
            st.info("Your answer will appear here")

if __name__ == "__main__":
    main() 