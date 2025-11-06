"""
Évaluation des Paramètres de Découpage (Chunking) pour le Système RAG
========================================================================

Ce script teste différentes configurations de découpage de documents
pour trouver les paramètres optimaux pour votre système RAG.

Paramètres à évaluer:
- chunk_size: Taille des chunks (nombre de caractères)
- chunk_overlap: Chevauchement entre chunks
- separators: Séparateurs utilisés pour découper
"""

import os
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import json
from datetime import datetime
import numpy as np


class ChunkingEvaluator:
    """Classe pour évaluer différentes stratégies de chunking"""
    
    def __init__(self, pdf_folder="data"):
        self.pdf_folder = pdf_folder
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.documents = []
        self.results = []
        
    def load_documents(self):
        """Charge tous les documents PDF"""
        print("📚 Chargement des documents PDF...")
        all_docs = []
        
        for filename in os.listdir(self.pdf_folder):
            if filename.endswith('.pdf'):
                filepath = os.path.join(self.pdf_folder, filename)
                print(f"  - Chargement: {filename}")
                loader = PyPDFLoader(filepath)
                docs = loader.load()
                all_docs.extend(docs)
        
        self.documents = all_docs
        print(f"✅ {len(all_docs)} pages chargées depuis {len(os.listdir(self.pdf_folder))} fichiers\n")
        return all_docs
    
    def create_chunks(self, chunk_size, chunk_overlap, separators=None):
        """Crée des chunks avec les paramètres donnés"""
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(self.documents)
        return chunks
    
    def analyze_chunks(self, chunks, config_name):
        """Analyse les caractéristiques des chunks"""
        chunk_lengths = [len(chunk.page_content) for chunk in chunks]
        
        stats = {
            "config_name": config_name,
            "total_chunks": len(chunks),
            "avg_chunk_length": np.mean(chunk_lengths),
            "median_chunk_length": np.median(chunk_lengths),
            "min_chunk_length": np.min(chunk_lengths),
            "max_chunk_length": np.max(chunk_lengths),
            "std_chunk_length": np.std(chunk_lengths),
            "chunks_under_100": sum(1 for l in chunk_lengths if l < 100),
            "chunks_100_500": sum(1 for l in chunk_lengths if 100 <= l < 500),
            "chunks_500_1000": sum(1 for l in chunk_lengths if 500 <= l < 1000),
            "chunks_over_1000": sum(1 for l in chunk_lengths if l >= 1000),
        }
        
        return stats
    
    def test_retrieval_quality(self, chunks, test_queries):
        """Teste la qualité de récupération avec des requêtes"""
        print("  🔍 Test de récupération...")
        
        # Créer le vectorstore
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        retrieval_scores = []
        
        for query in test_queries:
            # Récupérer les top-k documents
            docs = vectorstore.similarity_search_with_score(query, k=3)
            
            # Score basé sur la similarité (plus bas = meilleur avec FAISS)
            avg_score = np.mean([score for _, score in docs])
            retrieval_scores.append(avg_score)
        
        return {
            "avg_retrieval_score": np.mean(retrieval_scores),
            "min_retrieval_score": np.min(retrieval_scores),
            "max_retrieval_score": np.max(retrieval_scores),
        }
    
    def evaluate_configuration(self, chunk_size, chunk_overlap, separators, config_name, test_queries):
        """Évalue une configuration complète"""
        print(f"\n{'='*70}")
        print(f"🧪 Test de configuration: {config_name}")
        print(f"   Chunk Size: {chunk_size}")
        print(f"   Chunk Overlap: {chunk_overlap}")
        print(f"   Separators: {separators[:3]}...")
        print(f"{'='*70}")
        
        # Créer les chunks
        chunks = self.create_chunks(chunk_size, chunk_overlap, separators)
        
        # Analyser les chunks
        stats = self.analyze_chunks(chunks, config_name)
        
        # Tester la récupération
        retrieval_stats = self.test_retrieval_quality(chunks, test_queries)
        
        # Combiner les stats
        result = {
            **stats,
            **retrieval_stats,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "timestamp": datetime.now().isoformat()
        }
        
        self.results.append(result)
        
        # Afficher les résultats
        print(f"\n📊 Résultats:")
        print(f"   Total chunks: {result['total_chunks']}")
        print(f"   Longueur moyenne: {result['avg_chunk_length']:.0f} caractères")
        print(f"   Longueur médiane: {result['median_chunk_length']:.0f} caractères")
        print(f"   Écart-type: {result['std_chunk_length']:.0f}")
        print(f"\n📈 Distribution des tailles:")
        print(f"   < 100 chars: {result['chunks_under_100']} ({result['chunks_under_100']/result['total_chunks']*100:.1f}%)")
        print(f"   100-500 chars: {result['chunks_100_500']} ({result['chunks_100_500']/result['total_chunks']*100:.1f}%)")
        print(f"   500-1000 chars: {result['chunks_500_1000']} ({result['chunks_500_1000']/result['total_chunks']*100:.1f}%)")
        print(f"   > 1000 chars: {result['chunks_over_1000']} ({result['chunks_over_1000']/result['total_chunks']*100:.1f}%)")
        print(f"\n🎯 Qualité de récupération:")
        print(f"   Score moyen: {result['avg_retrieval_score']:.4f}")
        print(f"   Score min: {result['min_retrieval_score']:.4f}")
        print(f"   Score max: {result['max_retrieval_score']:.4f}")
        
        return result
    
    def run_all_tests(self):
        """Exécute tous les tests avec différentes configurations"""
        print("\n" + "="*70)
        print("🚀 DÉMARRAGE DE L'ÉVALUATION DES PARAMÈTRES DE CHUNKING")
        print("="*70)
        
        # Charger les documents
        self.load_documents()
        
        # Requêtes de test (adaptez selon votre domaine)
        test_queries = [
            "Qu'est-ce que l'apprentissage supervisé?",
            "Expliquez le fonctionnement des réseaux de neurones",
            "Quelle est la différence entre classification et régression?",
            "Comment fonctionne le gradient descent?",
            "Qu'est-ce que l'overfitting et comment l'éviter?",
        ]
        
        # Configurations à tester
        configurations = [
            # Configuration 1: Chunks petits
            {
                "chunk_size": 300,
                "chunk_overlap": 50,
                "separators": ["\n\n", "\n", ". ", " ", ""],
                "name": "Petits chunks (300/50)"
            },
            # Configuration 2: Chunks moyens (actuel)
            {
                "chunk_size": 500,
                "chunk_overlap": 100,
                "separators": ["\n\n", "\n", ". ", " ", ""],
                "name": "Chunks moyens (500/100) - ACTUEL"
            },
            # Configuration 3: Chunks moyens avec plus d'overlap
            {
                "chunk_size": 500,
                "chunk_overlap": 150,
                "separators": ["\n\n", "\n", ". ", " ", ""],
                "name": "Chunks moyens + overlap (500/150)"
            },
            # Configuration 4: Chunks grands
            {
                "chunk_size": 800,
                "chunk_overlap": 150,
                "separators": ["\n\n", "\n", ". ", " ", ""],
                "name": "Grands chunks (800/150)"
            },
            # Configuration 5: Chunks très grands
            {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "separators": ["\n\n", "\n", ". ", " ", ""],
                "name": "Très grands chunks (1000/200)"
            },
            # Configuration 6: Chunks moyens avec séparateurs optimisés
            {
                "chunk_size": 500,
                "chunk_overlap": 100,
                "separators": ["\n\n\n", "\n\n", "\n", ". ", "? ", "! ", " ", ""],
                "name": "Chunks moyens + séparateurs optimisés (500/100)"
            },
            # Configuration 7: Équilibre optimal
            {
                "chunk_size": 600,
                "chunk_overlap": 120,
                "separators": ["\n\n\n", "\n\n", "\n", ". ", " ", ""],
                "name": "Équilibre optimal (600/120)"
            },
        ]
        
        # Tester chaque configuration
        for config in configurations:
            self.evaluate_configuration(
                chunk_size=config["chunk_size"],
                chunk_overlap=config["chunk_overlap"],
                separators=config["separators"],
                config_name=config["name"],
                test_queries=test_queries
            )
        
        # Analyser les résultats
        self.analyze_results()
        
        # Sauvegarder les résultats
        self.save_results()
    
    def analyze_results(self):
        """Analyse comparative de tous les résultats"""
        print("\n" + "="*70)
        print("📊 ANALYSE COMPARATIVE DES RÉSULTATS")
        print("="*70)
        
        # Trouver la meilleure configuration selon différents critères
        best_retrieval = min(self.results, key=lambda x: x['avg_retrieval_score'])
        most_chunks = max(self.results, key=lambda x: x['total_chunks'])
        least_chunks = min(self.results, key=lambda x: x['total_chunks'])
        most_consistent = min(self.results, key=lambda x: x['std_chunk_length'])
        
        print(f"\n🏆 Meilleure qualité de récupération:")
        print(f"   {best_retrieval['config_name']}")
        print(f"   Score: {best_retrieval['avg_retrieval_score']:.4f}")
        
        print(f"\n📦 Plus de chunks (granularité fine):")
        print(f"   {most_chunks['config_name']}")
        print(f"   Total: {most_chunks['total_chunks']} chunks")
        
        print(f"\n📦 Moins de chunks (chunks plus longs):")
        print(f"   {least_chunks['config_name']}")
        print(f"   Total: {least_chunks['total_chunks']} chunks")
        
        print(f"\n📏 Plus cohérent (écart-type le plus faible):")
        print(f"   {most_consistent['config_name']}")
        print(f"   Écart-type: {most_consistent['std_chunk_length']:.0f}")
        
        # Recommandation
        print(f"\n" + "="*70)
        print("💡 RECOMMANDATIONS")
        print("="*70)
        
        # Calculer un score composite (pondéré)
        for result in self.results:
            # Score composite: privilégier la qualité de récupération et la cohérence
            result['composite_score'] = (
                result['avg_retrieval_score'] * 0.5 +  # 50% qualité récupération
                (result['std_chunk_length'] / 1000) * 0.3 +  # 30% cohérence (normalisé)
                (1 / (result['total_chunks'] / 100)) * 0.2  # 20% nombre raisonnable de chunks
            )
        
        best_overall = min(self.results, key=lambda x: x['composite_score'])
        
        print(f"\n🎯 Configuration recommandée (score composite):")
        print(f"   {best_overall['config_name']}")
        print(f"   Chunk Size: {best_overall['chunk_size']}")
        print(f"   Chunk Overlap: {best_overall['chunk_overlap']}")
        print(f"   Total chunks: {best_overall['total_chunks']}")
        print(f"   Score récupération: {best_overall['avg_retrieval_score']:.4f}")
        print(f"   Longueur moyenne: {best_overall['avg_chunk_length']:.0f} caractères")
        
        print(f"\n💡 Pour utiliser cette configuration dans app.py, modifiez:")
        print(f"   text_splitter = RecursiveCharacterTextSplitter(")
        print(f"       chunk_size={best_overall['chunk_size']},")
        print(f"       chunk_overlap={best_overall['chunk_overlap']},")
        print(f"       separators=[\"\\n\\n\", \"\\n\", \". \", \" \", \"\"]")
        print(f"   )")
    
    def save_results(self):
        """Sauvegarde les résultats dans un fichier JSON"""
        output_file = "chunking_evaluation_results.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "evaluation_date": datetime.now().isoformat(),
                "total_documents": len(self.documents),
                "configurations_tested": len(self.results),
                "results": self.results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Résultats sauvegardés dans: {output_file}")
        
        # Créer aussi un rapport markdown
        self.create_markdown_report()
    
    def create_markdown_report(self):
        """Crée un rapport markdown détaillé"""
        report_file = "chunking_evaluation_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 📊 Rapport d'Évaluation des Paramètres de Chunking\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Documents analysés:** {len(self.documents)} pages\n\n")
            f.write(f"**Configurations testées:** {len(self.results)}\n\n")
            
            f.write("## 🎯 Résumé Exécutif\n\n")
            
            # Table comparative
            f.write("## 📈 Tableau Comparatif\n\n")
            f.write("| Configuration | Chunks | Taille Moy. | Score Récup. | Écart-type |\n")
            f.write("|--------------|--------|-------------|--------------|------------|\n")
            
            for result in self.results:
                f.write(f"| {result['config_name']} | ")
                f.write(f"{result['total_chunks']} | ")
                f.write(f"{result['avg_chunk_length']:.0f} | ")
                f.write(f"{result['avg_retrieval_score']:.4f} | ")
                f.write(f"{result['std_chunk_length']:.0f} |\n")
            
            f.write("\n## 📊 Détails par Configuration\n\n")
            
            for result in self.results:
                f.write(f"### {result['config_name']}\n\n")
                f.write(f"**Paramètres:**\n")
                f.write(f"- Chunk Size: {result['chunk_size']}\n")
                f.write(f"- Chunk Overlap: {result['chunk_overlap']}\n\n")
                f.write(f"**Statistiques:**\n")
                f.write(f"- Total chunks: {result['total_chunks']}\n")
                f.write(f"- Longueur moyenne: {result['avg_chunk_length']:.0f} caractères\n")
                f.write(f"- Longueur médiane: {result['median_chunk_length']:.0f} caractères\n")
                f.write(f"- Écart-type: {result['std_chunk_length']:.0f}\n")
                f.write(f"- Score récupération: {result['avg_retrieval_score']:.4f}\n\n")
                f.write(f"**Distribution:**\n")
                f.write(f"- < 100 chars: {result['chunks_under_100']}\n")
                f.write(f"- 100-500 chars: {result['chunks_100_500']}\n")
                f.write(f"- 500-1000 chars: {result['chunks_500_1000']}\n")
                f.write(f"- > 1000 chars: {result['chunks_over_1000']}\n\n")
        
        print(f"📄 Rapport markdown créé: {report_file}")


def main():
    """Fonction principale"""
    evaluator = ChunkingEvaluator(pdf_folder="data")
    evaluator.run_all_tests()
    
    print("\n" + "="*70)
    print("✅ ÉVALUATION TERMINÉE!")
    print("="*70)
    print("\nConsultez les fichiers générés:")
    print("  - chunking_evaluation_results.json (données brutes)")
    print("  - chunking_evaluation_report.md (rapport détaillé)")
    print("\n")


if __name__ == "__main__":
    main()
