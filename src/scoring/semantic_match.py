import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import asyncio
import re
import chromadb
from chromadb.utils import embedding_functions

class SemanticMatcher:
    def __init__(self):
        self.embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        )
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=None  # Uses environment variable GOOGLE_API_KEY
        )
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.get_collection(
                name="resume_jd_embeddings",
                embedding_function=self.embedding_function
            )
        except:
            self.collection = self.chroma_client.create_collection(
                name="resume_jd_embeddings",
                embedding_function=self.embedding_function,
                metadata={"description": "Resume and JD embeddings for semantic matching"}
            )
    
    async def calculate_embedding_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """Calculate semantic similarity using embeddings with ChromaDB"""
        
        try:
            # Store embeddings in ChromaDB
            doc_id1 = f"text1_{hash(text1)}"
            doc_id2 = f"text2_{hash(text2)}"
            
            # Add documents to collection
            self.collection.upsert(
                documents=[text1, text2],
                ids=[doc_id1, doc_id2],
                metadatas=[{"type": "text1"}, {"type": "text2"}]
            )
            
            # Query similar documents
            results = self.collection.query(
                query_texts=[text1],
                n_results=2,
                include=['distances']
            )
            
            # Calculate similarity (distance to similarity conversion)
            if len(results['distances'][0]) > 1:
                distance = results['distances'][0][1]  # Distance to text2
                similarity = max(0, 1 - distance)  # Convert distance to similarity
                return similarity * 100
            else:
                # Fallback to direct embedding calculation
                embeddings = await self.embeddings_model.aembed_documents([text1, text2])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                return similarity * 100
            
        except Exception as e:
            print(f"Error calculating embedding similarity: {e}")
            return self.calculate_tfidf_similarity(text1, text2)
    
    def calculate_tfidf_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """Calculate similarity using TF-IDF as fallback"""
        
        try:
            # Fit and transform texts
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
            
            # Convert to percentage
            return similarity * 100
            
        except Exception as e:
            print(f"Error calculating TF-IDF similarity: {e}")
            return 50.0  # Default middle score
    
    async def calculate_section_similarity(
        self,
        resume_sections: Dict[str, str],
        jd_sections: Dict[str, str]
    ) -> Dict[str, float]:
        """Calculate similarity for specific sections"""
        
        section_similarities = {}
        
        for section_name in ['skills', 'experience', 'education', 'projects']:
            resume_section = resume_sections.get(section_name, '')
            jd_section = jd_sections.get(section_name, '')
            
            if resume_section and jd_section:
                similarity = await self.calculate_embedding_similarity(
                    resume_section,
                    jd_section
                )
                section_similarities[section_name] = similarity
            else:
                section_similarities[section_name] = 0.0
        
        return section_similarities
    
    async def calculate_contextual_relevance(
        self,
        resume_text: str,
        jd_context: str
    ) -> float:
        """Calculate contextual relevance using semantic understanding"""
        
        # Extract key phrases from JD
        jd_key_phrases = self._extract_key_phrases(jd_context)
        
        # Calculate relevance for each key phrase
        relevance_scores = []
        
        for phrase in jd_key_phrases:
            # Find best matching segment in resume
            best_match_score = 0
            resume_segments = self._split_into_segments(resume_text)
            
            for segment in resume_segments:
                score = await self.calculate_embedding_similarity(phrase, segment)
                best_match_score = max(best_match_score, score)
            
            relevance_scores.append(best_match_score)
        
        # Return average relevance
        if relevance_scores:
            return sum(relevance_scores) / len(relevance_scores)
        else:
            return 50.0
    
    def _extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """Extract key phrases from text"""
        
        # Simple extraction based on sentence importance
        sentences = re.split(r'[.!?]', text)
        
        # Filter and clean sentences
        key_phrases = []
        keywords = ['required', 'must have', 'essential', 'responsible', 'experience', 
                   'knowledge', 'skills', 'proficient', 'expert', 'strong']
        
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if any(keyword in sentence_lower for keyword in keywords):
                if 20 < len(sentence) < 200:  # Reasonable length
                    key_phrases.append(sentence.strip())
        
        return key_phrases[:max_phrases]
    
    def _split_into_segments(self, text: str, segment_size: int = 200) -> List[str]:
        """Split text into meaningful segments"""
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        segments = []
        for para in paragraphs:
            if len(para) > segment_size:
                # Split long paragraphs
                words = para.split()
                for i in range(0, len(words), segment_size // 5):
                    segment = ' '.join(words[i:i + segment_size // 5])
                    if segment:
                        segments.append(segment)
            elif para.strip():
                segments.append(para)
        
        return segments
    
    async def calculate_skill_embedding_match(
        self,
        required_skills: List[str],
        resume_text: str
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate semantic match for skills"""
        
        skill_matches = {}
        
        for skill in required_skills:
            # Create skill context
            skill_context = f"Experience with {skill} including practical implementation and projects"
            
            # Find best matching segment in resume
            best_score = 0
            resume_segments = self._split_into_segments(resume_text)
            
            for segment in resume_segments:
                score = await self.calculate_embedding_similarity(skill_context, segment)
                best_score = max(best_score, score)
            
            skill_matches[skill] = best_score
        
        # Calculate average score
        if skill_matches:
            avg_score = sum(skill_matches.values()) / len(skill_matches)
        else:
            avg_score = 0
        
        return avg_score, skill_matches
    
    async def calculate_domain_similarity(
        self,
        resume_text: str,
        domain_description: str
    ) -> float:
        """Calculate domain-specific similarity"""
        
        # Create domain-specific context
        domain_context = f"""
        Domain expertise in {domain_description}
        Including relevant experience, projects, and technical knowledge
        """
        
        # Calculate similarity
        similarity = await self.calculate_embedding_similarity(
            resume_text[:2000],  # Use first 2000 chars
            domain_context
        )
        
        return similarity
    
    async def calculate_role_fit(
        self,
        resume_text: str,
        role_description: str
    ) -> Dict[str, Any]:
        """Calculate overall role fit using semantic analysis"""
        
        # Different aspects of role fit
        aspects = {
            'technical_fit': f"Technical skills and expertise for {role_description}",
            'experience_fit': f"Relevant work experience for {role_description}",
            'project_fit': f"Projects and practical work related to {role_description}",
            'cultural_fit': f"Soft skills and team collaboration for {role_description}"
        }
        
        fit_scores = {}
        
        for aspect_name, aspect_context in aspects.items():
            score = await self.calculate_embedding_similarity(
                resume_text[:1000],
                aspect_context
            )
            fit_scores[aspect_name] = score
        
        # Calculate overall fit
        overall_fit = sum(fit_scores.values()) / len(fit_scores)
        
        return {
            'overall_fit': overall_fit,
            'aspect_scores': fit_scores,
            'strengths': [k for k, v in fit_scores.items() if v > 70],
            'weaknesses': [k for k, v in fit_scores.items() if v < 50]
        }
    
    async def generate_similarity_matrix(
        self,
        resume_texts: List[str],
        jd_text: str
    ) -> np.ndarray:
        """Generate similarity matrix for batch processing"""
        
        # Get embeddings for all texts
        all_texts = resume_texts + [jd_text]
        embeddings = await self.embeddings_model.aembed_documents(all_texts)
        
        # Calculate similarity matrix
        jd_embedding = embeddings[-1]
        resume_embeddings = embeddings[:-1]
        
        similarities = []
        for resume_emb in resume_embeddings:
            sim = cosine_similarity([resume_emb], [jd_embedding])[0][0]
            similarities.append(sim * 100)
        
        return np.array(similarities)
    
    async def calculate_comprehensive_semantic_score(
        self,
        resume_text: str,
        jd_text: str,
        jd_requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate comprehensive semantic matching score"""
        
        results = {}
        
        # Overall document similarity
        overall_similarity = await self.calculate_embedding_similarity(
            resume_text[:3000],
            jd_text[:3000]
        )
        results['overall_similarity'] = overall_similarity
        
        # Skill semantic matching
        if jd_requirements.get('required_skills'):
            skill_score, skill_matches = await self.calculate_skill_embedding_match(
                jd_requirements['required_skills'],
                resume_text
            )
            results['skill_semantic_score'] = skill_score
            results['skill_semantic_matches'] = skill_matches
        
        # Contextual relevance
        contextual_score = await self.calculate_contextual_relevance(
            resume_text,
            jd_text
        )
        results['contextual_relevance'] = contextual_score
        
        # Domain similarity
        if jd_requirements.get('domain'):
            domain_score = await self.calculate_domain_similarity(
                resume_text,
                jd_requirements['domain']
            )
            results['domain_similarity'] = domain_score
        
        # Role fit analysis
        role_fit = await self.calculate_role_fit(
            resume_text,
            jd_requirements.get('role_title', 'the position')
        )
        results['role_fit'] = role_fit
        
        # Calculate weighted semantic score
        weights = {
            'overall_similarity': 0.30,
            'skill_semantic_score': 0.25,
            'contextual_relevance': 0.20,
            'domain_similarity': 0.15,
            'role_fit': 0.10
        }
        
        total_score = 0
        for key, weight in weights.items():
            if key == 'role_fit' and 'role_fit' in results:
                score = results['role_fit']['overall_fit']
            elif key in results:
                score = results[key]
            else:
                score = 50  # Default middle score
            
            total_score += score * weight
        
        results['total_semantic_score'] = total_score
        results['weights_used'] = weights
        
        return results