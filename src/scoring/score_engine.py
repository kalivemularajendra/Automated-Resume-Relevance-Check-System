from typing import Dict, Any, List, Tuple
import asyncio
from datetime import datetime
from .hard_match import HardMatcher
from .semantic_match import SemanticMatcher

class ScoreEngine:
    def __init__(self):
        self.hard_matcher = HardMatcher()
        self.semantic_matcher = SemanticMatcher()
        self.scoring_weights = {
            'hard_match': 0.6,
            'semantic_match': 0.4
        }
    
    async def evaluate_resume(
        self,
        resume_text: str,
        jd_text: str,
        jd_requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Main evaluation function combining all scoring methods"""
        
        evaluation_start = datetime.now()
        
        # Perform hard matching
        hard_match_results = self.hard_matcher.calculate_comprehensive_score(
            resume_text,
            jd_requirements
        )
        
        # Perform semantic matching
        semantic_match_results = await self.semantic_matcher.calculate_comprehensive_semantic_score(
            resume_text,
            jd_text,
            jd_requirements
        )
        
        # Calculate final scores
        final_score = self.calculate_final_score(
            hard_match_results['total_hard_match_score'],
            semantic_match_results['total_semantic_score']
        )
        
        # Generate verdict
        verdict = self.generate_verdict(final_score)
        
        # Generate detailed feedback
        feedback = self.generate_detailed_feedback(
            hard_match_results,
            semantic_match_results,
            final_score,
            verdict
        )
        
        # Calculate evaluation time
        evaluation_time = (datetime.now() - evaluation_start).total_seconds()
        
        return {
            'relevance_score': final_score,
            'verdict': verdict,
            'hard_match_score': hard_match_results['total_hard_match_score'],
            'semantic_match_score': semantic_match_results['total_semantic_score'],
            'detailed_scores': {
                'hard_match': hard_match_results,
                'semantic_match': semantic_match_results
            },
            'matched_skills': hard_match_results.get('matched_skills', []),
            'missing_skills': hard_match_results.get('missing_skills', []),
            'experience_match': hard_match_results.get('experience_verdict', 'Not specified'),
            'recommendations': feedback['recommendations'],
            'strengths': feedback['strengths'],
            'areas_for_improvement': feedback['areas_for_improvement'],
            'evaluation_time': evaluation_time,
            'confidence_score': self.calculate_confidence_score(
                hard_match_results,
                semantic_match_results
            )
        }
    
    def calculate_final_score(
        self,
        hard_score: float,
        semantic_score: float
    ) -> float:
        """Calculate weighted final score"""
        
        final_score = (
            hard_score * self.scoring_weights['hard_match'] +
            semantic_score * self.scoring_weights['semantic_match']
        )
        
        return round(final_score, 2)
    
    def generate_verdict(self, score: float) -> str:
        """Generate verdict based on score"""
        
        if score >= 80:
            return "HIGH"
        elif score >= 60:
            return "MEDIUM"
        elif score >= 40:
            return "LOW"
        else:
            return "VERY LOW"
    
    def calculate_confidence_score(
        self,
        hard_results: Dict[str, Any],
        semantic_results: Dict[str, Any]
    ) -> float:
        """Calculate confidence in the evaluation"""
        
        # Check consistency between hard and semantic scores
        hard_score = hard_results['total_hard_match_score']
        semantic_score = semantic_results['total_semantic_score']
        
        score_difference = abs(hard_score - semantic_score)
        
        # Higher confidence when scores are consistent
        if score_difference < 10:
            confidence = 95
        elif score_difference < 20:
            confidence = 85
        elif score_difference < 30:
            confidence = 75
        else:
            confidence = 65
        
        # Adjust based on data quality
        if hard_results.get('matched_skills'):
            confidence += 5
        
        return min(100, confidence)
    
    def generate_detailed_feedback(
        self,
        hard_results: Dict[str, Any],
        semantic_results: Dict[str, Any],
        final_score: float,
        verdict: str
    ) -> Dict[str, List[str]]:
        """Generate detailed feedback for the candidate"""
        
        recommendations = []
        strengths = []
        areas_for_improvement = []
        
        # Analyze strengths
        if hard_results['skill_score'] > 70:
            strengths.append("Strong technical skill match")
        
        if hard_results.get('experience_score', 0) > 80:
            strengths.append("Experience level aligns well with requirements")
        
        if hard_results.get('project_score', 0) > 60:
            strengths.append("Relevant project experience demonstrated")
        
        if semantic_results.get('contextual_relevance', 0) > 70:
            strengths.append("Resume content highly relevant to job context")
        
        if semantic_results.get('role_fit', {}).get('overall_fit', 0) > 75:
            strengths.append("Good overall fit for the role")
        
        # Analyze areas for improvement
        if hard_results.get('missing_skills'):
            areas_for_improvement.append(
                f"Missing key skills: {', '.join(hard_results['missing_skills'][:3])}"
            )
            recommendations.append(
                f"Consider gaining experience in: {', '.join(hard_results['missing_skills'][:3])}"
            )
        
        if hard_results.get('experience_score', 100) < 50:
            areas_for_improvement.append("Experience level needs improvement")
            recommendations.append(
                "Gain more relevant experience through projects or internships"
            )
        
        if hard_results.get('certification_score', 0) < 30:
            areas_for_improvement.append("Lack of relevant certifications")
            recommendations.append(
                "Consider obtaining industry-relevant certifications"
            )
        
        if semantic_results.get('overall_similarity', 0) < 50:
            areas_for_improvement.append("Resume content needs better alignment with job requirements")
            recommendations.append(
                "Tailor your resume to better match the job description keywords and requirements"
            )
        
        # Role-specific recommendations
        if semantic_results.get('role_fit'):
            weaknesses = semantic_results['role_fit'].get('weaknesses', [])
            for weakness in weaknesses:
                if 'technical' in weakness:
                    recommendations.append("Strengthen technical skills through online courses or bootcamps")
                elif 'experience' in weakness:
                    recommendations.append("Highlight relevant experience more prominently")
                elif 'project' in weakness:
                    recommendations.append("Add more relevant projects to demonstrate practical skills")
        
        # Score-based recommendations
        if final_score < 40:
            recommendations.extend([
                "Major reskilling required for this role",
                "Consider entry-level positions or internships first",
                "Focus on building fundamental skills in the domain"
            ])
        elif final_score < 60:
            recommendations.extend([
                "Address the skill gaps identified above",
                "Gain more hands-on experience through personal projects",
                "Network with professionals in the field"
            ])
        elif final_score < 80:
            recommendations.extend([
                "Fine-tune your resume to better highlight relevant experience",
                "Consider obtaining advanced certifications",
                "Showcase more quantifiable achievements"
            ])
        else:
            recommendations.extend([
                "Your profile is strong - ensure you prepare well for interviews",
                "Research the company culture and values",
                "Prepare specific examples demonstrating your expertise"
            ])
        
        return {
            'recommendations': recommendations[:5],  # Top 5 recommendations
            'strengths': strengths,
            'areas_for_improvement': areas_for_improvement
        }
    
    async def batch_evaluate(
        self,
        resumes: List[Tuple[str, str]],  # List of (resume_text, candidate_id)
        jd_text: str,
        jd_requirements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Evaluate multiple resumes in batch"""
        
        tasks = []
        for resume_text, candidate_id in resumes:
            task = self.evaluate_resume(resume_text, jd_text, jd_requirements)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Add candidate IDs to results
        for i, result in enumerate(results):
            result['candidate_id'] = resumes[i][1]
        
        # Sort by score
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return results
    
    def generate_comparison_report(
        self,
        evaluations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate comparative analysis of multiple candidates"""
        
        if not evaluations:
            return {}
        
        # Calculate statistics
        scores = [e['relevance_score'] for e in evaluations]
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        
        # Categorize candidates
        high_matches = [e for e in evaluations if e['verdict'] == 'HIGH']
        medium_matches = [e for e in evaluations if e['verdict'] == 'MEDIUM']
        low_matches = [e for e in evaluations if e['verdict'] in ['LOW', 'VERY LOW']]
        
        # Find common missing skills
        all_missing_skills = []
        for e in evaluations:
            all_missing_skills.extend(e.get('missing_skills', []))
        
        skill_frequency = {}
        for skill in all_missing_skills:
            skill_frequency[skill] = skill_frequency.get(skill, 0) + 1
        
        common_missing_skills = sorted(
            skill_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            'total_evaluated': len(evaluations),
            'statistics': {
                'average_score': avg_score,
                'highest_score': max_score,
                'lowest_score': min_score,
                'score_range': max_score - min_score
            },
            'categorization': {
                'high_matches': len(high_matches),
                'medium_matches': len(medium_matches),
                'low_matches': len(low_matches)
            },
            'top_candidates': evaluations[:5],
            'common_skill_gaps': [skill for skill, _ in common_missing_skills],
            'recommendations': {
                'immediate_interview': [e['candidate_id'] for e in high_matches],
                'further_screening': [e['candidate_id'] for e in medium_matches],
                'skills_development': [e['candidate_id'] for e in low_matches]
            }
        }