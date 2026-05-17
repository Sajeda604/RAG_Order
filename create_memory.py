"""
create_memory.py
─────────────────
يقرأ stores_data.txt ويبني FAISS vectorstore.
شغّله مرة واحدة: python create_memory.py
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

load_dotenv()

DB_PATH = "vectorstore/db_faiss"
DATA_FILE = "stores_data.txt"


def build():
    os.makedirs(DB_PATH, exist_ok=True)

    # تحقق إذا موجودة مسبقاً
    if os.path.isfile(os.path.join(DB_PATH, "index.faiss")):
        print("✅ Vectorstore موجودة مسبقاً — لا حاجة لإعادة البناء.")
        print("   🔄 لإعادة البناء: احذف مجلد vectorstore/ ثم شغّل من جديد.")
        return

    if not os.path.isfile(DATA_FILE):
        raise FileNotFoundError(f"❌ الملف '{DATA_FILE}' غير موجود.")

    print(f"📄 تحميل {DATA_FILE} ...")
    loader = TextLoader(DATA_FILE, encoding="utf-8")
    docs = loader.load()
    print(f"✅ تم تحميل {len(docs)} مستند")

    # تقطيع النص
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n===", "\n---", "\n", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ {len(chunks)} مقطع")

    # بناء Embeddings
    print("🧠 بناء Embeddings ...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        # ← هذا النموذج أفضل للعربية من all-MiniLM-L6-v2
    )

    # بناء FAISS
    print("⚙️  بناء FAISS ...")
    db = FAISS.from_documents(chunks, embeddings)

    # حفظ
    db.save_local(DB_PATH)
    print(f"🎉 تم الحفظ في: {DB_PATH}")


if __name__ == "__main__":
    build()
