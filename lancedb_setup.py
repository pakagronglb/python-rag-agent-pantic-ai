from pathlib import Path
import tiktoken
import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from lancedb.table import LanceTable
from lancedb.rerankers import LinearCombinationReranker

openai_func = get_registry().get('openai').create(name='text-embedding-3-small')

class Document(LanceModel):
    """
    Defines the schema for documents to be stored in the LanceDB table.
    """
    id: str
    text: str = openai_func.SourceField()
    vector: Vector(openai_func.ndims()) = openai_func.VectorField()

def chunk_text(text: str, max_tokens: int = 8192, encoding_name: str = 'cl100k_base'):
    """
    Chunk text into smaller parts to fit within a maximum token limit.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)

    for i in range(0, len(tokens), max_tokens):
        yield encoding.decode(tokens[i : i + max_tokens])

def create_lancedb_table(db_path: str, table_name: str, overwrite: bool = True):
    """
    Connect to LanceDB and create a table for storing knowledge documents.
    """
    db = lancedb.connect(db_path)
    mode = 'overwrite' if overwrite else 'create'
    table = db.create_table(table_name, schema=Document, mode=mode)
    return table

def drop_lancedb_table(db_path: str, table_name: str):
    """
    Drop a LanceDB table if it exists.
    """
    db = lancedb.connect(db_path)
    db.drop_table(table_name, ignore_missing=True)

def add_documents_to_table(table: LanceTable, knowledge_base_dir: str, max_tokens: int = 8192):
    """
    Add markdown documents from a local directory to the LanceDB table.
    """
    docs = []
    knowledge_base = Path(knowledge_base_dir)

    for md_file in knowledge_base.glob('*.md'):
        print(f'Processing {md_file.name}')
        with open(md_file, 'r', encoding='utf-8') as f:
            text = f.read()
            for i, chunk in enumerate(chunk_text(text, max_tokens=max_tokens)):
                doc_id = f'{md_file.stem}_{i}'
                docs.append({'id': doc_id, 'text': chunk})

    if docs:
        table.add(docs)
        print(f'Added {len(docs)} documents (chunks) to the table.')
    else:
        print('No documents found or added.')

def retrieve_similar_docs(table: LanceTable, query: str, limit: int = 100):
    """
    Retrieve documents from the LanceDB table using vector similarity search.
    """
    results = (
        table.search(query)
        .limit(limit)
        .to_list()
    )
    return results

def setup_lancedb():
    """
    Set up LanceDB table with initial configuration.
    """
    db_path = './db'
    table_name = 'knowledge'
    knowledge_base_dir = './knowledge-file'

    table = create_lancedb_table(db_path, table_name, overwrite=True)
    add_documents_to_table(table, knowledge_base_dir)

if __name__ == '__main__':
    setup_lancedb()