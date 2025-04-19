import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

const loader = new PDFLoader("./data/docs/mehrabian1974.pdf", {
  splitPages: false,
});

const docs = await loader.load();
console.log(docs);
