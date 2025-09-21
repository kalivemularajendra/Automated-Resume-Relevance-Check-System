from typing import Dict, Any, List
import hashlib
import re

def format_score(score: float) -> str:
    """Format score with color coding"""
    if score >= 75:
        return f"🟢 {score:.1f}%"
    elif score >= 50:
        return f"🟡 {score:.1f}%"
    else:
        return f"🔴 {score:.1f}%"

def generate_verdict(score: float) -> str:
    """Generate verdict based on score"""
    if score >= 75:
        return "HIGH"
    elif score >= 50:
        return "MEDIUM"
    else:
        return "LOW"

def calculate_hash(text: str) -> str:
    """Calculate hash of text for duplicate detection"""
    return hashlib.md5(text.encode()).hexdigest()

def extract_years_experience(text: str) -> int:
    """Extract years of experience from text"""
    patterns = [
        r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
        r'experience\s*:\s*(\d+)\+?\s*years?',
        r'(\d+)\+?\s*years?\s*in'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            return int(matches[0])
    
    return 0

def normalize_skills(skills: List[str]) -> List[str]:
    """Normalize skill names for better matching"""
    skill_mappings = {
        'js': 'javascript',
        'py': 'python',
        'ml': 'machine learning',
        'dl': 'deep learning',
        'np': 'numpy',
        'pd': 'pandas',
        'tf': 'tensorflow',
        'k8s': 'kubernetes'
    }
    
    normalized = []
    for skill in skills:
        skill_lower = skill.lower().strip()
        normalized.append(skill_mappings.get(skill_lower, skill_lower))
    
    return list(set(normalized))

def calculate_skill_match_percentage(
    required_skills: List[str],
    candidate_skills: List[str]
) -> float:
    """Calculate percentage of required skills matched"""
    if not required_skills:
        return 100.0
    
    required_set = set(normalize_skills(required_skills))
    candidate_set = set(normalize_skills(candidate_skills))
    
    matched = required_set.intersection(candidate_set)
    return (len(matched) / len(required_set)) * 100

def generate_skill_gap_report(
    required_skills: List[str],
    candidate_skills: List[str]
) -> Dict[str, List[str]]:
    """Generate detailed skill gap analysis"""
    required_set = set(normalize_skills(required_skills))
    candidate_set = set(normalize_skills(candidate_skills))
    
    return {
        'matched': list(required_set.intersection(candidate_set)),
        'missing': list(required_set - candidate_set),
        'additional': list(candidate_set - required_set)
    }

def format_recommendations(recommendations: List[str]) -> str:
    """Format recommendations for display"""
    formatted = []
    for i, rec in enumerate(recommendations, 1):
        formatted.append(f"{i}. {rec}")
    return "\n".join(formatted)

def calculate_weighted_score(
    hard_match: float,
    semantic_match: float,
    experience_match: float = None
) -> float:
    """Calculate weighted final score"""
    if experience_match is not None:
        return (hard_match * 0.5) + (semantic_match * 0.3) + (experience_match * 0.2)
    else:
        return (hard_match * 0.6) + (semantic_match * 0.4)

def categorize_score(score: float) -> Dict[str, Any]:
    """Categorize score with detailed feedback"""
    if score >= 90:
        return {
            'category': 'Excellent Match',
            'color': '#28a745',
            'icon': '🌟',
            'message': 'Outstanding fit for the role!'
        }
    elif score >= 75:
        return {
            'category': 'Strong Match',
            'color': '#5cb85c',
            'icon': '✅',
            'message': 'Very good fit for the role.'
        }
    elif score >= 60:
        return {
            'category': 'Good Match',
            'color': '#f0ad4e',
            'icon': '👍',
            'message': 'Decent fit with some gaps.'
        }
    elif score >= 40:
        return {
            'category': 'Partial Match',
            'color': '#d9534f',
            'icon': '⚠️',
            'message': 'Significant gaps to address.'
        }
    else:
        return {
            'category': 'Poor Match',
            'color': '#dc3545',
            'icon': '❌',
            'message': 'Major skill gaps present.'
        }