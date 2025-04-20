import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { PineconeStore } from "@langchain/pinecone";
import { Pinecone } from "@pinecone-database/pinecone";
import { PromptTemplate } from "@langchain/core/prompts";
import {
  RunnableSequence,
  RunnablePassthrough,
} from "@langchain/core/runnables";
import dotenv from "dotenv";
import { formatDocumentsAsString } from "langchain/util/document";
import { StringOutputParser } from "@langchain/core/output_parsers";
dotenv.config();

// ⬆️ load a document
const loader = new PDFLoader(
  "./data/docs/SukanyaSamriddhiAccountSchemeRule.pdf",
  {
    splitPages: false,
  }
);

const docs = await loader.load();
// console.log(docs);

// ✂️ split the document
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 150,
});
const chunks = await splitter.splitDocuments(docs);
console.log(`Split the document into ${chunks.length} sub-documents.`);

// 🔑 create embeddings
const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-small",
  apiKey: process.env.OPENAI_API_KEY,
  batchSize: 512, //max 2048
  dimensions: 1024,
});

const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pinecone.index(process.env.PINECONE_INDEX);
// console.log("🚀 ~ index:", index);

// instantiate VectorStore
const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
  pineconeIndex: index,
  // Maximum number of batch requests to allow at once. Each batch is 1000 vectors.
  maxConcurrency: 5,
});
const pineconeStore = await vectorStore.addDocuments(chunks);
console.log("🚀 ~ pineconeStore:", pineconeStore);

/* const similaritySearchResults = await vectorStore.similaritySearch(
  "emotional reactions",
  2
);
console.log("similaritySearchResults", similaritySearchResults);
for (const doc of similaritySearchResults) {
  console.log(`* ${doc.pageContent} [${JSON.stringify(doc.metadata, null)}]`);
} */

// optional filter
const filter = {}; //match to metadata
const retriever = vectorStore.asRetriever({
  filter,
  k: 10, // no.of results
});

// Retrireval chain
const customPromptTemplate = `Answer the quetions based only on the context provided.
If you don't know the answer, just say politely that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
{context}
Question: {question}
`;
const model = new ChatOpenAI({ temperature: 0.2 });
const prompt = PromptTemplate.fromTemplate(customPromptTemplate);
const chain = RunnableSequence.from([
  {
    context: retriever.pipe(formatDocumentsAsString),
    question: new RunnablePassthrough(),
  },
  prompt,
  model,
  new StringOutputParser(),
]);
const reply = await chain.invoke("Can I premature close the account ?");
console.log(reply);
