# Parallel Query (MultiQuery Retrieval-Augmented Generation)

This project demonstrates a **Parallel Query (MultiQuery Retrieval-Augmented Generation)** pipeline using LangChain and OpenAI. The pipeline generates multiple search queries for a given question, retrieves relevant documents, and uses the retrieved context to generate a final answer.

## Features

- **MultiQuery Generation**: Generates diverse search queries based on the user's question.
- **Parallel Document Retrieval**: Executes multiple queries in parallel to retrieve relevant documents.
- **Contextual Answer Generation**: Combines retrieved documents to generate a comprehensive answer.
- **Vector Store Integration**: Uses embeddings to store and retrieve documents efficiently.

## How It Works

1. **Input Question**: The user provides a question.
2. **Query Generation**: The system generates multiple search queries to cover different aspects of the question.
3. **Document Retrieval**: Relevant documents are retrieved for each query using a vector store.
4. **Answer Generation**: The retrieved documents are used as context to generate a final answer using a language model.

## Code Structure

### 1. **Document Setup**

Sample documents are provided to simulate a knowledge base:

```javascript
const documents = [
  new Document({ pageContent: "The capital of France is Paris." }),
  new Document({ pageContent: "Paris is a beautiful city in France." }),
  new Document({ pageContent: "France is known for its wine and cheese." }),
  new Document({ pageContent: "The Eiffel Tower is in Paris." }),
];
```

### 2. **Vector Store**

The documents are split into chunks and stored in a vector store for efficient retrieval:

```javascript
const vectorStore = await MemoryVectorStore.fromDocuments(splits, embeddings);
const retriever = vectorStore.asRetriever();
```

### 3. **MultiQuery Generation**

A prompt is used to generate multiple search queries:

```javascript
const multiQueryPrompt = PromptTemplate.fromTemplate(
  `Generate maximum 10 search queries based on the user's question to retrieve relevant documents.
  The queries should be diverse and cover different aspects of the question.

  Question: {question}
  Queries:`
);
```

### 4. **Document Retrieval**

The generated queries are executed in parallel to retrieve relevant documents:

```javascript
async function retrieveDocuments(queries) {
  const results = await Promise.all(
    queries.map((query) => retriever.invoke(query))
  );
  return results.flat();
}
```

### 5. **Answer Generation**

The retrieved documents are used to generate a final answer:

```javascript
const answerPrompt = PromptTemplate.fromTemplate(
  `You are a helpful assistant. Answer the question based on the provided context.
  Context: {context}
  Question: {question}
  Answer: `
);
```

### 6. **Pipeline Execution**

The pipeline is implemented as a RunnableSequence:

```javascript
const multiQueryRAGChain = RunnableSequence.from([
  {
    question: (input) => input.question,
    context: async (input) => {
      const queries = await generateMultiQueryChain.invoke({
        question: input.question,
      });
      const retrievedDocs = await retrieveDocuments(queries);
      return retrievedDocs;
    },
  },
  {
    Answer: answerPrompt,
    Question: (input) => input.question,
    Context: (input) => input.context.map((doc) => doc.pageContent).join("\n"),
  },
  async (input) => {
    const response = await model.invoke(input.Answer);
    return response;
  },
]);
```

### 7. **Usage**

The main function runs the pipeline and logs the final answer:

```javascript
async function main() {
  const question = "What is Paris known for?";
  const multiQueryResult = await multiQueryRAGChain.invoke({ question });
  console.log("Final Answer:", multiQueryResult);
}
main().catch(console.error);
```

## Prerequisites

- Node.js installed on your system.
- OpenAI API key stored in a .env file:

```text
OPENAI_API_KEY=your_openai_api_key
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-repo/parallel-query.git
cd parallel-query
```

2. Install dependencies

```bash
npm install
```

3. Run the script

```bash
node ParallelQuery.js
```

## Example Output

For the question "What is Paris known for?", the output might look like:

```
--- MultiQuery Retrieval ---
Final Answer: Paris is known for the Eiffel Tower, its beauty as a city in France, being the capital of France, and France's wine and cheese.
```
