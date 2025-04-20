import { Pinecone } from "@pinecone-database/pinecone";
import dotenv from "dotenv";
dotenv.config();

console.log(process.env.PINECONE_API_KEY);
const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});
console.log("🚀 ~ pc:", pc);
