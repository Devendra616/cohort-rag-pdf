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
import { RetrievalQAChain } from "langchain/chains";
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
console.log("🚀 ~ vectorStore:", vectorStore);
const retriever = vectorStore.asRetriever();
