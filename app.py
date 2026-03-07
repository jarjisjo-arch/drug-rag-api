from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import requests
import os

app = Flask(__name__)

# -------------------- إعداد النموذج --------------------
# اختر النموذج المناسب:
# - all-MiniLM-L6-v2: نموذج صغير (80 MB) - أسرع وأقل استهلاكاً للذاكرة (مفعل حالياً)
# - paraphrase-multilingual-MiniLM-L12-v2: نموذج كبير (470 MB) - أدق لكنه يستهلك ذاكرة أكبر
MODEL_NAME = "all-MiniLM-L6-v2"  # <-- غيّر هذا السطر فقط للعودة للنموذج الكبير
# -------------------------------------------------------

print(f"🔄 جاري تحميل قاعدة البيانات باستخدام النموذج: {MODEL_NAME}...")
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
db = FAISS.load_local("drug_database", embeddings, allow_dangerous_deserialization=True)
print("✅ قاعدة البيانات جاهزة!")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-08cb0518f2de46359748c8413fb46ee7")

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '')
    
    docs = db.similarity_search(question, k=5)
    context = "\n---\n".join([doc.page_content for doc in docs])
    
    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": f"أنت صيدلي خبير. أجب من المعلومات فقط:\n{context}"},
                {"role": "user", "content": question}
            ],
            "temperature": 0.3
        }
    )
    
    return jsonify({
        'answer': response.json()['choices'][0]['message']['content']
    })

@app.route('/', methods=['GET'])
def home():
    return f"✅ API شغال! النموذج المستخدم: {MODEL_NAME}. أرسل POST request إلى /ask"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
