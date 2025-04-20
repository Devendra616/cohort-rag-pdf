import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

const loadDocs = async () => {
  const loader = new PDFLoader("./data/docs/mehrabian1974.pdf", {
    splitPages: false,
  });
  const pdf = await loader.load();
  return pdf;
};

const docs = loadDocs();
console.log(docs);
/* 
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 150,
});
const chunks = await splitter.splitDocuments(docs);
const { OpenAIEmbeddings } = require("langchain/embeddings/openai");
const { PineconeStore } = require("langchain/vectorstores/pinecone");

const embeddings = new OpenAIEmbeddings();
const pinecone = new PineconeClient();
await pinecone.init({ apiKey: process.env.PINECONE_API_KEY });

const pineconeStore = await PineconeStore.fromDocuments(chunks, embeddings, {
  indexName: process.env.PINECONE_INDEX_NAME,
});
const { RetrievalQAChain } = require("langchain/chains");

const chain = VectorDBQAChain.fromLLM(
  new OpenAI({ temperature: 0 }),
  pineconeStore.asRetriever({ k: 3 }),
  { returnSourceDocuments: true }
);

const response = await chain.call({ query: "What is the refund policy?" });
console.log(response.text);
 */
