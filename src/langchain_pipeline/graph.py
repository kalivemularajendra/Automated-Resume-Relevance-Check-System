import os
from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EvaluationState(TypedDict):
    resume_text: str
    jd_text: str
    jd_parsed: Dict[str, Any]
    hard_match_score: float
    semantic_match_score: float
    relevance_score: float
    verdict: str
    missing_skills: List[str]
    matched_skills: List[str]
    recommendations: List[str]
    experience_match: str

class ResumeEvaluationGraph:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.3
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        )
        self.build_graph()
    
    def _ensure_jd_dict(self, jd_data):
        """Ensure JD data is a dictionary, parse if it's a string"""
        if isinstance(jd_data, str):
            try:
                import json
                return json.loads(jd_data)
            except (json.JSONDecodeError, TypeError):
                # Return a fallback structure if parsing fails
                return {
                    "required_skills": [],
                    "preferred_skills": [],
                    "experience_range": {"min": 0, "max": 10},
                    "education_requirements": [],
                    "responsibilities": [],
                    "keywords": []
                }
        elif isinstance(jd_data, dict):
            return jd_data
        else:
            # Return fallback for any other type
            return {
                "required_skills": [],
                "preferred_skills": [],
                "experience_range": {"min": 0, "max": 10},
                "education_requirements": [],
                "responsibilities": [],
                "keywords": []
            }
    
    def build_graph(self):
        """Build the evaluation graph using LangGraph"""
        workflow = StateGraph(EvaluationState)
        
        # Add nodes
        workflow.add_node("extract_features", self.extract_features)
        workflow.add_node("hard_match", self.calculate_hard_match)
        workflow.add_node("semantic_match", self.calculate_semantic_match)
        workflow.add_node("calculate_final_score", self.calculate_final_score)
        workflow.add_node("generate_feedback", self.generate_feedback)
        
        # Add edges
        workflow.add_edge("extract_features", "hard_match")
        workflow.add_edge("hard_match", "semantic_match")
        workflow.add_edge("semantic_match", "calculate_final_score")
        workflow.add_edge("calculate_final_score", "generate_feedback")
        workflow.add_edge("generate_feedback", END)
        
        # Set entry point
        workflow.set_entry_point("extract_features")
        
        self.app = workflow.compile()
    
    async def extract_features(self, state: EvaluationState) -> EvaluationState:
        """Extract features from resume and JD"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert resume analyzer. Extract key information from the resume."),
            ("human", """Extract the following from this resume:
            1. Technical skills
            2. Years of experience
            3. Education level
            4. Key projects
            5. Certifications
            
            Resume: {resume_text}
            
            Provide a structured extraction.""")
        ])
        
        response = await self.llm.ainvoke(
            prompt.format_messages(resume_text=state["resume_text"][:3000])
        )
        
        # Store extracted features for next steps
        state["extracted_features"] = response.content
        
        return state
    
    async def calculate_hard_match(self, state: EvaluationState) -> EvaluationState:
        """Calculate hard match score based on keyword matching"""
        
        resume_lower = state["resume_text"].lower()
        jd_requirements = self._ensure_jd_dict(state["jd_parsed"])
        
        matched_skills = []
        missing_skills = []
        
        # Check required skills
        for skill in jd_requirements.get("required_skills", []):
            if skill.lower() in resume_lower:
                matched_skills.append(skill)
            else:
                missing_skills.append(skill)
        
        # Calculate hard match score
        total_required = len(jd_requirements.get("required_skills", []))
        if total_required > 0:
            hard_score = (len(matched_skills) / total_required) * 100
        else:
            hard_score = 50  # Default score if no requirements specified
        
        # Check experience
        exp_pattern = r'(\d+)\+?\s*years?'
        import re
        exp_matches = re.findall(exp_pattern, resume_lower)
        
        if exp_matches:
            candidate_exp = int(exp_matches[0])
            min_exp = jd_requirements.get("experience_range", {}).get("min", 0)
            max_exp = jd_requirements.get("experience_range", {}).get("max", 10)
            
            if min_exp <= candidate_exp <= max_exp:
                state["experience_match"] = "Perfect"
                hard_score += 10
            elif candidate_exp < min_exp:
                state["experience_match"] = "Under-qualified"
                hard_score -= 10
            else:
                state["experience_match"] = "Over-qualified"
                hard_score += 5
        else:
            state["experience_match"] = "Not specified"
        
        state["hard_match_score"] = min(100, max(0, hard_score))
        state["matched_skills"] = matched_skills
        state["missing_skills"] = missing_skills
        
        return state
    
    async def calculate_semantic_match(self, state: EvaluationState) -> EvaluationState:
        """Calculate semantic similarity using embeddings"""
        
        # Get embeddings
        resume_embedding = await self.embeddings.aembed_query(state["resume_text"][:3000])
        jd_embedding = await self.embeddings.aembed_query(state["jd_text"][:3000])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(
            [resume_embedding],
            [jd_embedding]
        )[0][0]
        
        # Convert to percentage score
        semantic_score = similarity * 100
        
        state["semantic_match_score"] = semantic_score
        
        return state
    
    async def calculate_final_score(self, state: EvaluationState) -> EvaluationState:
        """Calculate weighted final score and verdict"""
        
        # Weighted combination (60% hard match, 40% semantic)
        final_score = (
            state["hard_match_score"] * 0.6 +
            state["semantic_match_score"] * 0.4
        )
        
        # Determine verdict
        if final_score >= 75:
            verdict = "HIGH"
        elif final_score >= 50:
            verdict = "MEDIUM"
        else:
            verdict = "LOW"
        
        state["relevance_score"] = final_score
        state["verdict"] = verdict
        
        return state
    
    async def generate_feedback(self, state: EvaluationState) -> EvaluationState:
        """Generate personalized feedback and recommendations"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a career counselor providing constructive feedback."),
            ("human", """Based on this evaluation, provide 3-5 specific recommendations:
            
            Relevance Score: {score}
            Verdict: {verdict}
            Missing Skills: {missing_skills}
            Experience Match: {exp_match}
            
            Provide actionable recommendations for the candidate to improve their profile.
            Focus on:
            1. Skills to acquire
            2. Certifications to pursue
            3. Projects to showcase
            4. Resume improvements
            
            Keep each recommendation concise (1-2 sentences).""")
        ])
        
        response = await self.llm.ainvoke(
            prompt.format_messages(
                score=state["relevance_score"],
                verdict=state["verdict"],
                missing_skills=", ".join(state["missing_skills"][:5]),
                exp_match=state.get("experience_match", "N/A")
            )
        )
        
        # Parse recommendations
        recommendations = response.content.strip().split("\n")
        recommendations = [r.strip() for r in recommendations if r.strip()][:5]
        
        state["recommendations"] = recommendations
        
        return state
    
    async def evaluate(
        self,
        resume_text: str,
        jd_text: str,
        jd_parsed: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run the complete evaluation pipeline"""
        
        initial_state = EvaluationState(
            resume_text=resume_text,
            jd_text=jd_text,
            jd_parsed=jd_parsed,
            hard_match_score=0,
            semantic_match_score=0,
            relevance_score=0,
            verdict="",
            missing_skills=[],
            matched_skills=[],
            recommendations=[],
            experience_match=""
        )
        
        result = await self.app.ainvoke(initial_state)
        
        return {
            "relevance_score": result["relevance_score"],
            "verdict": result["verdict"],
            "hard_match_score": result["hard_match_score"],
            "semantic_match_score": result["semantic_match_score"],
            "missing_skills": result["missing_skills"],
            "matched_skills": result["matched_skills"],
            "recommendations": result["recommendations"],
            "experience_match": result["experience_match"]
        }