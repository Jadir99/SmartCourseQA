import os
import pickle
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# Charger les variables d'environnement
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Variables globales pour le système RAG
qa_chain = None
documents = None
vectorstore = None

def initialize_rag_system():
    """Initialise le système RAG au démarrage de l'application"""
    global qa_chain, documents, vectorstore
    
    print("🚀 Initialisation du système RAG...")
    
    # 1. Charger les documents
    if os.path.exists("raw_documents.pkl"):
        with open("raw_documents.pkl", "rb") as f:
            raw_documents = pickle.load(f)
        print(f"✅ Documents chargés : {len(raw_documents)} PDFs")
    else:
        folder_path = "data/"
        raw_documents = []
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(folder_path, filename)
                reader = PdfReader(pdf_path)
                page_texts = []
                for page in reader.pages:
                    content = (page.extract_text() or "").replace("\n", " ").strip()
                    if content:
                        page_texts.append(content)
                full_text = " ".join(page_texts)
                if full_text:
                    raw_documents.append({"text": full_text, "metadata": {"source": filename}})
        
        with open("raw_documents.pkl", "wb") as f:
            pickle.dump(raw_documents, f)
        print(f"✅ {len(raw_documents)} PDFs chargés et sauvegardés")
    
    # 2. Créer les chunks
    if os.path.exists("documents_chunks.pkl"):
        with open("documents_chunks.pkl", "rb") as f:
            documents = pickle.load(f)
        print(f"✅ Chunks chargés : {len(documents)} chunks")
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=350,
            chunk_overlap=120
        )
        
        documents = []
        for item in raw_documents:
            splitted_docs = text_splitter.create_documents([item["text"]], metadatas=[item["metadata"]])
            for idx, doc in enumerate(splitted_docs):
                doc.metadata["chunk_index"] = idx
            documents.extend(splitted_docs)
        
        with open("documents_chunks.pkl", "wb") as f:
            pickle.dump(documents, f)
        print(f"✅ {len(documents)} chunks créés et sauvegardés")
    
    # 3. Créer les embeddings et FAISS
    embedding_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        encode_kwargs={"normalize_embeddings": True}
    )
    
    if os.path.exists("faiss_index"):
        vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
        print("✅ Index FAISS chargé")
    else:
        vectorstore = FAISS.from_documents(documents, embedding_model)
        vectorstore.save_local("faiss_index")
        print("✅ Index FAISS créé et sauvegardé")
    
    # 4. Configurer le retriever hybride
    dense_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 4}
    )
    
    sparse_retriever = BM25Retriever.from_documents(documents)
    sparse_retriever.k = 8
    
    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.65, 0.35]
    )
    
    # 5. Configurer le LLM
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.4,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/Jadir99/SmartCourseQA",
            "X-Title": "RAG Flask App",
        }
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=hybrid_retriever,
        return_source_documents=True,
        chain_type="stuff"
    )
    
    print("✅ Système RAG initialisé avec succès!")
    return qa_chain, documents, vectorstore


@app.route('/')
def landing():
    """Landing page du projet"""
    return render_template('landing.html')


@app.route('/chatbot')
def chatbot():
    """Page du chatbot"""
    return render_template('index.html')


@app.route('/quiz')
def quiz_page():
    """Page de génération de quiz"""
    return render_template('quiz.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Endpoint pour le chatbot"""
    try:
        data = request.json
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'Question vide'}), 400
        
        # Obtenir la réponse du RAG
        result = qa_chain(question)
        
        # Formater les sources
        sources = []
        for doc in result.get("source_documents", []):
            sources.append({
                'source': doc.metadata.get('source', 'Unknown'),
                'chunk': doc.metadata.get('chunk_index', 0),
                'content': doc.page_content[:200] + "..."
            })
        
        return jsonify({
            'answer': result["result"],
            'sources': sources
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate-quiz', methods=['POST'])
def generate_quiz():
    """Générer un quiz basé sur les cours"""
    try:
        data = request.json
        topic = data.get('topic', 'intelligence artificielle')
        num_questions = data.get('num_questions', 5)
        
        # Créer le prompt pour générer le quiz
        quiz_prompt = f"""
Génère exactement {num_questions} questions de quiz à choix multiples (QCM) sur le sujet: "{topic}".

INSTRUCTIONS IMPORTANTES:
- Génère EXACTEMENT {num_questions} questions complètes
- Chaque question doit avoir EXACTEMENT 4 options (A, B, C, D)
- Utilise les informations des cours pour créer des questions pertinentes
- Les questions doivent être claires et précises

FORMAT STRICT À SUIVRE POUR CHAQUE QUESTION:

Question 1: [Texte de la question ici]
A) [Première option]
B) [Deuxième option]
C) [Troisième option]
D) [Quatrième option]
Réponse correcte: A
Explication: [Explication de pourquoi cette réponse est correcte]

---

Question 2: [Texte de la question ici]
A) [Première option]
B) [Deuxième option]
C) [Troisième option]
D) [Quatrième option]
Réponse correcte: B
Explication: [Explication de pourquoi cette réponse est correcte]

---

Continue ainsi jusqu'à la Question {num_questions}.
N'OUBLIE PAS le séparateur "---" entre chaque question.
"""
        
        result = qa_chain(quiz_prompt)
        quiz_text = result["result"]
        
        print(f"📝 Quiz généré pour le sujet: {topic}")
        print(f"📊 Texte brut du quiz (premiers 500 caractères):\n{quiz_text[:500]}...")
        
        # Parser le quiz
        questions = parse_quiz(quiz_text, num_questions)
        
        print(f"✅ {len(questions)} questions parsées avec succès")
        
        # Sauvegarder le quiz dans la session
        session['current_quiz'] = questions
        session['quiz_topic'] = topic
        
        return jsonify({
            'questions': questions,
            'topic': topic,
            'count': len(questions)
        })
    
    except Exception as e:
        print(f"❌ Erreur lors de la génération du quiz: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/evaluate-quiz', methods=['POST'])
def evaluate_quiz():
    """Évaluer les réponses de l'utilisateur"""
    try:
        data = request.json
        user_answers = data.get('answers', {})  # {question_index: answer}
        
        # Récupérer le quiz de la session
        quiz = session.get('current_quiz', [])
        
        if not quiz:
            return jsonify({'error': 'Aucun quiz actif'}), 400
        
        # Calculer le score
        results = []
        correct_count = 0
        
        for i, question in enumerate(quiz):
            user_answer = user_answers.get(str(i))
            correct_answer = question.get('correct_answer')
            is_correct = user_answer == correct_answer
            
            if is_correct:
                correct_count += 1
            
            results.append({
                'question_index': i,
                'question': question.get('question'),
                'user_answer': user_answer,
                'correct_answer': correct_answer,
                'is_correct': is_correct,
                'explanation': question.get('explanation')
            })
        
        score = (correct_count / len(quiz)) * 100 if quiz else 0
        
        return jsonify({
            'score': score,
            'correct_count': correct_count,
            'total_questions': len(quiz),
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def parse_quiz(quiz_text, expected_questions):
    """Parse le texte du quiz généré en structure JSON"""
    questions = []
    
    print(f"\n🔍 Début du parsing du quiz...")
    print(f"📄 Texte complet du quiz:\n{quiz_text}\n")
    
    # Diviser par les séparateurs ---
    sections = quiz_text.split('---')
    print(f"📋 Nombre de sections trouvées: {len(sections)}")
    
    for idx, section in enumerate(sections):
        if not section.strip():
            continue
        
        print(f"\n📌 Traitement de la section {idx + 1}:")
        print(section[:200])
        
        lines = [line.strip() for line in section.strip().split('\n') if line.strip()]
        
        question_data = {
            'question': '',
            'options': {},
            'correct_answer': '',
            'explanation': ''
        }
        
        current_question_found = False
        
        for line in lines:
            # Question (plusieurs formats possibles)
            if any(line.lower().startswith(prefix) for prefix in ['question', 'q.']):
                # Extraire le texte après "Question X:" ou "Q.X:"
                if ':' in line:
                    question_data['question'] = line.split(':', 1)[1].strip()
                else:
                    question_data['question'] = line
                current_question_found = True
                print(f"  ✓ Question trouvée: {question_data['question'][:50]}...")
            
            # Si pas de préfixe "Question" mais c'est la première ligne et pas d'option
            elif not current_question_found and not line.startswith(('A)', 'B)', 'C)', 'D)', 'Réponse', 'Explication')):
                question_data['question'] = line
                current_question_found = True
                print(f"  ✓ Question trouvée (sans préfixe): {question_data['question'][:50]}...")
            
            # Options A, B, C, D
            elif any(line.startswith(prefix) for prefix in ['A)', 'B)', 'C)', 'D)']):
                option_letter = line[0]
                option_text = line[2:].strip()
                question_data['options'][option_letter] = option_text
                print(f"  ✓ Option {option_letter} trouvée")
            
            # Réponse correcte
            elif any(keyword in line.lower() for keyword in ['réponse correcte', 'correct answer', 'réponse:', 'answer:']):
                # Extraire la lettre de la réponse
                parts = line.split(':')
                if len(parts) > 1:
                    answer = parts[1].strip().upper()
                    # Extraire juste la lettre (A, B, C, ou D)
                    for char in answer:
                        if char in ['A', 'B', 'C', 'D']:
                            question_data['correct_answer'] = char
                            print(f"  ✓ Réponse correcte: {char}")
                            break
            
            # Explication
            elif any(keyword in line.lower() for keyword in ['explication', 'explanation']):
                if ':' in line:
                    question_data['explanation'] = line.split(':', 1)[1].strip()
                else:
                    question_data['explanation'] = line
                print(f"  ✓ Explication trouvée")
        
        # Validation: ajouter seulement si la question est complète
        if (question_data['question'] and 
            len(question_data['options']) == 4 and 
            question_data['correct_answer'] in ['A', 'B', 'C', 'D']):
            questions.append(question_data)
            print(f"  ✅ Question {len(questions)} ajoutée avec succès")
        else:
            print(f"  ❌ Question incomplète ignorée:")
            print(f"     - Question: {'✓' if question_data['question'] else '✗'}")
            print(f"     - Options: {len(question_data['options'])}/4")
            print(f"     - Réponse correcte: {'✓' if question_data['correct_answer'] else '✗'}")
    
    print(f"\n📊 Résultat final: {len(questions)}/{expected_questions} questions parsées")
    
    # Si pas assez de questions, afficher un avertissement
    if len(questions) < expected_questions:
        print(f"⚠️ ATTENTION: Seulement {len(questions)} questions valides sur {expected_questions} attendues!")
        print(f"💡 Le LLM n'a peut-être pas généré toutes les questions au bon format.")
    
    return questions


if __name__ == '__main__':
    # Initialiser le système RAG au démarrage
    initialize_rag_system()
    
    # Lancer l'application Flask
    print("\n🌐 Application Flask démarrée sur http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
