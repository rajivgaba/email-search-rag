# Email Search RAG System Documentation

## Overview

This notebook implements a Retrieval-Augmented Generation (RAG) system for searching and querying email threads. The system processes email data, creates embeddings, stores them in a vector database, and provides intelligent responses to user queries using Large Language Models.

## Architecture

The RAG pipeline consists of 6 main components:

1. **Data Loading** - Load email data from Kaggle dataset
2. **Text Chunking** - Split large email texts into manageable chunks
3. **Vector Database** - Create embeddings and store in ChromaDB
4. **Retrieval Layer** - Query and retrieve relevant email chunks
5. **Prompt Engineering** - Create structured prompts for LLM
6. **Response Generation** - Generate contextual responses using LLM

## Dataset

**Source**: [Email Thread Summary Dataset](https://www.kaggle.com/datasets/marawanxmamdouh/email-thread-summary-dataset)

**Files**:
- `email_thread_details.csv` - Individual email details (21,684 records)
- `email_thread_summaries.csv` - Thread summaries (4,167 records)

**Schema**:
- `thread_id`: Unique identifier for email threads
- `subject`: Email subject line
- `timestamp`: Email timestamp
- `from`: Sender information
- `to`: Recipient information
- `body`: Email content
- `summary`: Thread summary (summaries file only)

## Dependencies

### Core Libraries
```python
pandas                  # Data manipulation
langchain              # Text processing utilities
openai                 # LLM API client
chromadb               # Vector database
sentence_transformers  # Embedding models
```

### Supporting Libraries
```python
dotenv                 # Environment variables
tqdm                   # Progress bars
multiprocessing        # Parallel processing
torch                  # PyTorch for GPU operations
gc                     # Garbage collection
```

## Setup Instructions

### 1. Environment Setup
```bash
pip install kaggle langchain openai dotenv pandas chromadb sentence_transformers protobuf ipywidgets
```

### 2. Environment Variables
Create a `.env` file with:
```env
PPLX_API_KEY=your_perplexity_api_key
OPENAI_API_KEY=your_openai_api_key
PPLX_BASE_MODEL=your_model_name
PPLX_BASE_URL=your_api_base_url
```

### 3. Data Download
```python
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
api.dataset_download_files('marawanxmamdouh/email-thread-summary-dataset', path='./data/', unzip=True)
```

## Implementation Details

### Text Chunking Strategy
- **Chunk Size**: 1000 characters
- **Overlap**: 30 characters
- **Method**: RecursiveCharacterTextSplitter from LangChain
- **Purpose**: Maintains context while ensuring manageable chunk sizes

### Embedding Model
- **Model**: `all-MiniLM-L6-v2` (SentenceTransformers)
- **Device**: MPS (Apple Silicon) or CUDA (NVIDIA)
- **Dimensions**: 384-dimensional embeddings
- **Performance**: Fast inference with good semantic understanding

### Vector Database Configuration
- **Database**: ChromaDB with persistent storage
- **Collection**: `email_collection`
- **Storage Path**: `./chroma_emails`
- **Batch Size**: 5000 documents per batch
- **Total Documents**: ~50,003 chunks

### Retrieval & Reranking
1. **Initial Retrieval**: Top 10 results using semantic similarity
2. **Reranking**: Cross-encoder model `cross-encoder/ms-marco-MiniLM-L-6-v2`
3. **Final Selection**: Top 3 reranked results for context

## Key Functions

### `get_query_results(input_query)`
Retrieves and reranks relevant email chunks for a given query.

**Parameters**:
- `input_query` (str): User's search query

**Returns**:
- DataFrame with top 3 relevant documents and metadata

**Process**:
1. Query ChromaDB for top 10 similar chunks
2. Apply cross-encoder reranking
3. Return top 3 results with metadata

### `generate_response(user_query)`
Generates comprehensive responses using retrieved context and LLM.

**Parameters**:
- `user_query` (str): User's question

**Process**:
1. Retrieve relevant email chunks
2. Construct system and user prompts
3. Call LLM API for response generation
4. Format and display results with citations

### `batch_add_with_progress(collection, docs, ids, metas, batch_size=5000)`
Efficiently adds documents to ChromaDB with progress tracking.

**Parameters**:
- `collection`: ChromaDB collection object
- `docs`: List of document texts
- `ids`: List of document IDs
- `metas`: List of metadata dictionaries
- `batch_size`: Number of documents per batch

## Data Processing Pipeline

### 1. Data Preparation
```python
# Merge emails by thread_id
grouped = email_details.groupby("thread_id")["body"].apply(lambda x: "\n\n".join(x)).reset_index()

# Create chunks with metadata
for i, row in grouped.iterrows():
    text_chunks = text_splitter.split_text(row['body'])
    # Store with thread_id and chunk_index metadata
```

### 2. Metadata Structure
```python
# Email details chunks
{
    "thread_id": "1234",
    "source": "email_details",
    "chunk_index": 0
}

# Email summaries
{
    "thread_id": "1234", 
    "source": "email_summaries"
}
```

### 3. Document ID Format
- **Chunks**: `{thread_id}::chunk::{chunk_index}`
- **Summaries**: `{thread_id}::summary`

## Prompt Engineering

### System Prompt
```
You are a helpful assistance that helps organisation find and validate past decisions, 
stratagies, and data by scanning the huge corpus of email threads.
```

### User Prompt Template
- Query context and retrieved documents
- Guidelines for accurate responses
- Citation requirements
- Response formatting instructions

## Performance Optimization

### Memory Management
```python
# GPU cache clearing
if torch.cuda.is_available():
    torch.cuda.empty_cache()
elif torch.backends.mps.is_available():
    torch.mps.empty_cache()

gc.collect()
```

### Batch Processing
- Documents processed in batches of 5000
- Progress tracking with tqdm
- Efficient memory usage

## Usage Examples

### Basic Query
```python
generate_response("who approved the photoshoot for the newsletter")
```

### Complex Query
```python
generate_response("search for any emails related to harassment of people")
```

### Financial Query
```python
generate_response("how much fine was required to be paid in T-6 leak")
```

## Output Format

Each query returns:
1. **User Query**: Original question
2. **Top 3 Results**: Retrieved documents with metadata
3. **LLM Response**: Generated answer with citations
4. **Citations**: Source references for verification

## Limitations

1. **Context Window**: Limited by LLM token limits
2. **Embedding Quality**: Dependent on sentence transformer model
3. **Data Coverage**: Limited to provided email dataset
4. **Language Support**: Optimized for English text
5. **Real-time Updates**: Static dataset, no live email integration

## Troubleshooting

### Common Issues

1. **GPU Memory Error**
   - Run memory clearing cell
   - Reduce batch size
   - Use CPU instead of GPU

2. **ChromaDB Connection Issues**
   - Check persistent directory permissions
   - Restart ChromaDB client
   - Clear existing collections

3. **API Rate Limits**
   - Implement retry logic
   - Add delays between requests
   - Monitor API usage

### Performance Tips

1. **Faster Queries**: Use smaller embedding models
2. **Better Accuracy**: Increase chunk overlap
3. **Memory Efficiency**: Process in smaller batches
4. **GPU Utilization**: Use appropriate device settings

## Future Enhancements

1. **Real-time Integration**: Connect to live email systems
2. **Multi-modal Support**: Handle attachments and images
3. **Advanced Filtering**: Date ranges, sender filters
4. **Conversation Threading**: Better thread reconstruction
5. **Semantic Search**: Enhanced query understanding
6. **Export Features**: Save results and conversations

## Security Considerations

1. **Data Privacy**: Ensure email data is properly anonymized
2. **API Keys**: Store securely in environment variables
3. **Access Control**: Implement user authentication
4. **Data Retention**: Follow organizational policies
5. **Audit Logging**: Track query history and access

## Monitoring & Maintenance

1. **Performance Metrics**: Query response times
2. **Accuracy Tracking**: User feedback on results
3. **Resource Usage**: GPU/CPU utilization
4. **Error Logging**: API failures and exceptions
5. **Data Updates**: Periodic dataset refreshes