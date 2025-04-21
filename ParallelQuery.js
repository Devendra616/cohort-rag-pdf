/* 
    ! MultiQuery/ Parallel Query / Fan Out
    An LLM is used to generate multiple search queries. These search queries can then be executed in parallel, and the retrieved results passed in altogether. This is really useful when a single question may rely on multiple sub questions.
*/

import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import dotenv from "dotenv";
dotenv.config();

// Initialize OpenAI model and embeddings
const model = new ChatOpenAI({ temperature: 0.2, model: "gpt-4o-mini" });
// 🔑 create embeddings
const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-small",
  apiKey: process.env.OPENAI_API_KEY,
  batchSize: 512, //max 2048
  dimensions: 1024,
});

// Sample documents
const documents = [
  new Document({ pageContent: "The capital of France is Paris." }),
  new Document({ pageContent: "Paris is a beautiful city in France." }),
  new Document({ pageContent: "France is known for its wine and cheese." }),
  new Document({ pageContent: "The Eiffel Tower is in Paris." }),
];

// Create a vector store from the documents
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 50,
});
const splits = await splitter.splitDocuments(documents);
const vectorStore = await MemoryVectorStore.fromDocuments(splits, embeddings);
// console.log("🚀 ~ vectorStore:", vectorStore);
const retriever = vectorStore.asRetriever();

// --- MultiQuery Implementation ---

// 1. Define a prompt for generating multiple queries
const multiQueryPrompt = PromptTemplate.fromTemplate(
  `Generate maximum 10 search queries based on the user's question to retrieve relevant documents.
  The queries should be diverse and cover different aspects of the question.

  Question: {question}
  Queries:`
);

// 2. Create a Runnable chain for generating multiple queries
const generateMultiQueryChain = RunnableSequence.from([
  multiQueryPrompt,
  model.pipe(new StringOutputParser()),
  (output) => output.split(",").map((query) => query.trim()),
]);

// 3. Define a function to retrieve documents for each generated query
async function retrieveDocuments(queries) {
  console.log(`Queries:${queries}`);
  const results = await Promise.all(
    queries.map((query) => retriever.invoke(query))
  );
  // Flatten the results and remove duplicates (optional, but recommended)
  const uniqueDocuments = Array.from(
    new Set(results.flat().map((doc) => JSON.stringify(doc)))
  ).map((json) => JSON.parse(json));
  return uniqueDocuments;
}

// 4. Define a prompt for the final answer
const answerPrompt =
  PromptTemplate.fromTemplate(`You are a helpful assistant. Answer the question based on the provided context.
Context: {context}
Question: {question}
Answer: `);

// 5. Create the final chain
const multiQueryRAGChain = RunnableSequence.from([
  {
    question: (input) => input.question,
    context: async (input) => {
      const queries = await generateMultiQueryChain.invoke({
        question: input.question,
      });
      const retrievedDocs = await retrieveDocuments(queries);
      //   console.log("retrievedDocs:", retrievedDocs);
      return retrievedDocs;
    },
  },
  {
    Answer: answerPrompt,
    Question: (input) => input.question,
    Context: (input) => input.context.map((doc) => doc.pageContent).join("\n"),
  },
  async (input) => {
    const response = await model.invoke(input.Answer); //process answerPrompt
    return response;
  },
  new StringOutputParser(),
]);

async function main() {
  const question = "What is Paris known for?";

  console.log("--- MultiQuery Retrieval ---");
  const multiQueryResult = await multiQueryRAGChain.invoke({ question });
  console.log("final Answer:", multiQueryResult); // Ensure the final answer is logged
}

main().catch(console.error);
