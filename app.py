from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import requests
import os

app = Flask(__name__)

# -------------------- النموذج الأصلي (الكبير) --------------------
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
# ----------------------------------------------------------------

print(f"🔄 جاري تحميل قاعدة البيانات باستخدام النموذج: {MODEL_NAME}...")
try:
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    # المسار المطلق لـ PythonAnywhere (غير jarjisjo لاسم مستخدمك)
    db = FAISS.load_local("/home/jarjisjo/drug_database", embeddings, allow_dangerous_deserialization=True)
    print("✅ قاعدة البيانات جاهزة!")
except Exception as e:
    print(f"❌ فشل تحميل قاعدة البيانات: {e}")
    db = None

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-08cb0518f2de46359748c8413fb46ee7")

@app.route('/ask', methods=['POST'])
def ask():
    if db is None:
        return jsonify({'error': 'قاعدة البيانات غير متوفرة'}), 500
    
    data = request.json
    question = data.get('question', '')
    
    docs = db.similarity_search(question, k=3)
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
    status = "✅" if db else "❌"
    return f"{status} API شغال! النموذج المستخدم: {MODEL_NAME}. قاعدة البيانات: {'جاهزة' if db else 'غير متوفرة'}"

if __name__ == '__main__':
    # PythonAnywhere لا يحتاج هذا الجزء، لكن نتركه للتوافق
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
