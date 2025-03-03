import lancedb
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from lancedb_setup import setup_lancedb, retrieve_similar_docs

def setup_knowledge_query_agent():
    """
    Set up the knowledge query agent.
    """
    agent = Agent(
        name='Knowledge Query Agent',
        model=OpenAIModel('gpt-4o-mini'),
        deps_type=str,
        result_type=str,
        system_prompt='Convert the user query into a clear and concise search query for finding relevant property information.',
    )
    return agent

def setup_main_agent():
    """
    Set up the main agent.
    """
    agent = Agent(
        name='Main Agent',
        model=OpenAIModel('gpt-4o-mini'),
        system_prompt='''You are a helpful real estate assistant. When providing information about properties:
1. Present information in a clear, organized manner
2. List specific details about properties when available (location, amenities, policies, etc.)
3. Include important numbers (prices, fees, limits) when present
4. If information is not available in the context, clearly state that
5. Keep responses concise but informative''',
    )
    return agent

def main():
    """
    Main execution flow for the application.
    """
    db_path = "./db"
    table_name = "knowledge"    
    db = lancedb.connect(db_path)
    
    knowledge_table = db.open_table(table_name) if db.table_names() else setup_lancedb()

    knowledge_query_agent = setup_knowledge_query_agent()
    agent = setup_main_agent()
    message_history = None

    print("\n🏠 Welcome to the Property Information Assistant!")
    print("Ask me anything about the properties in our database.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("\n💬 Your question: ")
        if query.lower() == 'exit':
            print("\nThank you for using the Property Information Assistant! Goodbye! 👋\n")
            break

        print("\n🔍 Searching...")
        
        res = knowledge_query_agent.run_sync(query)
        knowledge_query = res.data

        retrieved_docs = retrieve_similar_docs(knowledge_table, knowledge_query, limit=5)

        knowledge_context = ""
        for doc in retrieved_docs:
            knowledge_context += doc.get('text', '') + "\n\n"

        if not knowledge_context.strip():
            print("❌ I couldn't find any relevant information about that in our database.")
            continue

        prompt = f"""Context:
{knowledge_context}

User Question:
{query}

Please provide a clear and organized answer based on the above context. If specific information is not available in the context, please state that clearly."""

        response = agent.run_sync(prompt, message_history=message_history)
        print("\n📝 Answer:")
        print(response.data)

        message_history = response.all_messages()

if __name__ == '__main__':
    main()