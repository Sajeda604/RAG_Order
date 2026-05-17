"""
chatbot_chain.py
─────────────────
RAG chain مبنية على FAISS + Groq LLaMA.
البوت يفهم الطلب من بيانات المتاجر ويستخرج تفاصيل الأوردر.
"""

import os
import json
import re
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

load_dotenv()

DB_PATH = "vectorstore/db_faiss"

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
أنت مساعد طلبات ذكي لتطبيق توصيل متعدد المتاجر.
مهمتك: مساعدة العميل في تقديم طلبه بناءً على المتاجر والمنتجات المتاحة.

قواعد مهمة:
1. أجب فقط بناءً على المعلومات الموجودة في السياق (context).
2. إذا طلب العميل منتجاً غير موجود، أخبره بلطف وأقترح البديل.
3. إذا اكتمل الطلب، أعد ملخصاً واضحاً يحتوي:
   - اسم المتجر
   - المنتجات والكميات
   - السعر الإجمالي
   - طريقة الدفع
4. اكتشف لغة العميل (عربي/إنجليزي) وأجب بنفس اللغة.
5. كن ودوداً ومختصراً.
6. إذا لم يحدد طريقة دفع → افترض كاش.

عند اكتمال الطلب أضف هذا السطر في نهاية ردك بالضبط:
ORDER_READY::{{"store":"اسم المتجر","items":[{{"name":"المنتج","qty":1,"price":0.0}}],"total":0.0,"payment":"cash"}}

السياق:
{context}
"""

_rag_chain = None


def _load_chain():
    global _rag_chain
    if _rag_chain:
        return _rag_chain

    if not os.path.isfile(os.path.join(DB_PATH, "index.faiss")):
        raise FileNotFoundError(
            "❌ Vectorstore غير موجودة. شغّل أولاً: python create_memory.py"
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=800,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    combine_chain = create_stuff_documents_chain(llm, prompt)
    _rag_chain = create_retrieval_chain(
        db.as_retriever(search_kwargs={"k": 6}),
        combine_chain,
    )
    return _rag_chain


def chat(user_message: str) -> dict:
    """
    يرسل رسالة للـ RAG chain ويعيد:
    {
        "reply": "نص الرد",
        "order_ready": True/False,
        "order_data": {...} أو None
    }
    """
    chain = _load_chain()
    result = chain.invoke({"input": user_message})
    reply: str = result.get("answer", "")

    # استخرج ORDER_READY إذا اكتمل الطلب
    order_data = None
    order_ready = False

    match = re.search(r"ORDER_READY::(\{.*\})", reply, re.DOTALL)
    if match:
        order_ready = True
        try:
            order_data = json.loads(match.group(1))
        except json.JSONDecodeError:
            order_data = None
        # نظّف الرد من السطر التقني
        reply = reply[:match.start()].strip()

    return {
        "reply": reply,
        "order_ready": order_ready,
        "order_data": order_data,
    }
