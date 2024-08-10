import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate import infer_auto_device_map, init_empty_weights
import warnings

# 忽略所有的UserWarning
warnings.filterwarnings('ignore', category=UserWarning)

class PDFKnowledgeBase:
    def __init__(self, pdf_path, transformer_model='paraphrase-MiniLM-L6-v2', device='cpu'):
        self.pdf_path = pdf_path
        self.device = device
        self.paragraphs = self.extract_text_from_pdf()
        self.model = SentenceTransformer(transformer_model, device=device)
        self.paragraph_embeddings = self.create_embeddings()
        self.index = self.create_faiss_index()

    def extract_text_from_pdf(self):
        doc = fitz.open(self.pdf_path)
        text = ""
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    paragraph = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span["size"] == 10:
                                paragraph += span["text"]
                        paragraph += "\n"
                    if paragraph.strip():
                        text += paragraph.strip() + "\n\n"

        paragraphs = text.strip().split('\n\n')
        return paragraphs

    def create_embeddings(self, batch_size=16):
        paragraph_embeddings = []
        for i in range(0, len(self.paragraphs), batch_size):
            batch = self.paragraphs[i:i+batch_size]
            embeddings = self.model.encode(batch)
            paragraph_embeddings.extend(embeddings)
        if len(paragraph_embeddings) > 0:
            return np.array(paragraph_embeddings)
        else:
            raise ValueError("No embeddings were created. Check if the paragraphs are correctly extracted.")

    def create_faiss_index(self):
        if self.paragraph_embeddings.shape[0] > 0:
            index = faiss.IndexFlatL2(self.paragraph_embeddings.shape[1])
            index.add(self.paragraph_embeddings)
            return index
        else:
            raise ValueError("No valid embeddings found. Cannot create FAISS index.")

    def search_paragraphs(self, query, k=3):
        query_embedding = self.model.encode([query])
        D, I = self.index.search(query_embedding, k)
        return [self.paragraphs[i] for i in I[0]]


class QwenModel:
    def __init__(self, model_name="Qwen/Qwen2-0.5B", device="cuda"):
        self.device = device
        with init_empty_weights():
            self.qwen_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_response(self, context, query):
        content = f"Context: {context}\n\nAnswer the following question based on the above context:\n{query}\n\nAnswer only the following question without restating the context or question:\n"
        messages = [{"role": "user", "content": content}]
        text = self.qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        model_inputs = self.qwen_tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.qwen_model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.7,           # 温度
            repetition_penalty=2.2,    # 重复惩罚
            max_length=256,            # 最大生成长度
            top_k=50,                  # Top-k 采样
            top_p=0.9,                 # Top-p 采样
            num_beams=5                # 束搜索
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.qwen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

def re_rank_retrieved_paragraphs(query, retrieved_paragraphs, re_ranker_model, top_n=2):
    # 组合查询和段落对进行打分
    query_paragraph_pairs = [[query, paragraph] for paragraph in retrieved_paragraphs]
    scores = re_ranker_model.predict(query_paragraph_pairs)
    # 根据得分对段落进行排序
    sorted_indices = np.argsort(scores)[::-1]
    top_paragraphs = [retrieved_paragraphs[i] for i in sorted_indices[:top_n]]
    return top_paragraphs
if __name__ == '__main__':
    pdf_path = "/content/2007.12099v3.pdf"
    query = "What tricks are used when training pp-yole?"

    # 创建知识库并进行RAG检索
    knowledge_base = PDFKnowledgeBase(pdf_path)
    retrieved_paragraphs = knowledge_base.search_paragraphs(query)

    # 使用Cross-Encoder模型进行重新排序
    re_ranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    top_paragraphs = re_rank_retrieved_paragraphs(query, retrieved_paragraphs, re_ranker_model)

    retrieved_text = " ".join(top_paragraphs)

    # 使用千问2-0.5B模型生成回答
    qwen = QwenModel()
    response = qwen.generate_response(retrieved_text, query)

    print(response)
