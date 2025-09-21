import re
from typing import Dict, List, Tuple, Any
from difflib import SequenceMatcher
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
from fuzzywuzzy import fuzz, process

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class HardMatcher:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.skill_synonyms = self._load_skill_synonyms()
        self.bm25 = None  # Will be initialized when needed
        
    def _load_skill_synonyms(self) -> Dict[str, List[str]]:
        """Load skill synonyms for better matching"""
        return {
            'python': ['py', 'python3', 'python2'],
            'javascript': ['js', 'es6', 'es5', 'ecmascript'],
            'machine learning': ['ml', 'machine-learning', 'machinelearning'],
            'deep learning': ['dl', 'deep-learning', 'deeplearning'],
            'artificial intelligence': ['ai', 'a.i.', 'artificial-intelligence'],
            'natural language processing': ['nlp', 'natural-language-processing'],
            'computer vision': ['cv', 'computer-vision', 'image processing'],
            'data science': ['data-science', 'datascience', 'data scientist'],
            'sql': ['structured query language', 'mysql', 'postgresql', 'sqlite'],
            'nosql': ['no-sql', 'non-relational', 'mongodb', 'cassandra'],
            'react': ['reactjs', 'react.js', 'react native'],
            'angular': ['angularjs', 'angular.js'],
            'vue': ['vuejs', 'vue.js'],
            'node': ['nodejs', 'node.js'],
            'docker': ['containerization', 'docker container'],
            'kubernetes': ['k8s', 'container orchestration'],
            'aws': ['amazon web services', 'amazon cloud'],
            'azure': ['microsoft azure', 'ms azure'],
            'gcp': ['google cloud platform', 'google cloud'],
            'git': ['github', 'gitlab', 'bitbucket', 'version control'],
            'ci/cd': ['continuous integration', 'continuous deployment', 'jenkins', 'travis'],
            'agile': ['scrum', 'kanban', 'sprint'],
            'java': ['java8', 'java11', 'jvm'],
            'c++': ['cpp', 'c plus plus'],
            'tensorflow': ['tf', 'tf2'],
            'pytorch': ['torch'],
            'pandas': ['pd'],
            'numpy': ['np'],
            'scikit-learn': ['sklearn', 'scikit learn'],
        }
    
    def calculate_skill_match(
        self,
        required_skills: List[str],
        resume_text: str
    ) -> Tuple[float, List[str], List[str]]:
        """Calculate skill matching score"""
        resume_lower = resume_text.lower()
        matched_skills = []
        missing_skills = []
        
        for skill in required_skills:
            skill_lower = skill.lower()
            
            # Check direct match
            if self._skill_exists_in_text(skill_lower, resume_lower):
                matched_skills.append(skill)
                continue
            
            # Check synonyms
            if skill_lower in self.skill_synonyms:
                found = False
                for synonym in self.skill_synonyms[skill_lower]:
                    if self._skill_exists_in_text(synonym, resume_lower):
                        matched_skills.append(skill)
                        found = True
                        break
                
                if not found:
                    missing_skills.append(skill)
            else:
                # Fuzzy matching for skills not in synonym list
                if self._fuzzy_skill_match(skill_lower, resume_lower):
                    matched_skills.append(skill)
                else:
                    missing_skills.append(skill)
        
        # Calculate score
        if required_skills:
            score = (len(matched_skills) / len(required_skills)) * 100
        else:
            score = 100  # If no required skills, give full score
        
        return score, matched_skills, missing_skills
    
    def _skill_exists_in_text(self, skill: str, text: str) -> bool:
        """Check if skill exists in text using multiple methods"""
        # 1. Exact word boundary match
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            return True
        
        # 2. Fuzzy matching with high threshold
        if self._fuzzy_skill_match_enhanced(skill, text, threshold=85):
            return True
            
        # 3. BM25 scoring for context-aware matching
        if self._bm25_skill_match(skill, text, threshold=1.0):
            return True
            
        return False
    
    def _fuzzy_skill_match_enhanced(self, skill: str, text: str, threshold: float = 85) -> bool:
        """Enhanced fuzzy matching using fuzzywuzzy"""
        # Extract potential matches using fuzzywuzzy
        best_match, score = process.extractOne(
            skill.lower(),
            text.lower().split(),
            scorer=fuzz.ratio
        )
        
        if score >= threshold:
            return True
            
        # Check for partial matches in phrases
        words = text.lower().split()
        for i in range(len(words)):
            for j in range(i+1, min(i+4, len(words)+1)):  # Check up to 3-word phrases
                phrase = ' '.join(words[i:j])
                score = fuzz.ratio(skill.lower(), phrase)
                if score >= threshold:
                    return True
                    
        return False
    
    def _bm25_skill_match(self, skill: str, text: str, threshold: float = 1.0) -> bool:
        """BM25-based skill matching for contextual relevance"""
        try:
            # Tokenize the text into sentences/paragraphs
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            
            if not sentences:
                return False
            
            # Tokenize each sentence
            tokenized_sentences = [sentence.lower().split() for sentence in sentences]
            
            # Initialize BM25
            bm25 = BM25Okapi(tokenized_sentences)
            
            # Query with the skill
            query = skill.lower().split()
            scores = bm25.get_scores(query)
            
            # Check if any sentence has a score above threshold
            max_score = max(scores) if scores else 0
            return max_score >= threshold
            
        except Exception as e:
            print(f"BM25 matching error: {e}")
            return False
    
    def _fuzzy_skill_match(self, skill: str, text: str, threshold: float = 0.85) -> bool:
        """Legacy fuzzy matching for skills (keeping for backward compatibility)"""
        # Tokenize text
        tokens = word_tokenize(text)
        
        # Check each token and bigram
        for i, token in enumerate(tokens):
            # Single token match
            if SequenceMatcher(None, skill, token).ratio() >= threshold:
                return True
            
            # Bigram match
            if i < len(tokens) - 1:
                bigram = f"{token} {tokens[i+1]}"
                if SequenceMatcher(None, skill, bigram).ratio() >= threshold:
                    return True
        
        return False
    
    def calculate_experience_match(
        self,
        resume_text: str,
        min_exp: int,
        max_exp: int
    ) -> Tuple[float, str]:
        """Calculate experience matching score"""
        
        # Extract years of experience from resume
        exp_patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
            r'experience\s*[:–-]\s*(\d+)\+?\s*(?:years?|yrs?)',
            r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:professional|industry|relevant)',
            r'(?:total|overall)\s*(?:experience|exp)\s*[:–-]\s*(\d+)\+?\s*(?:years?|yrs?)',
        ]
        
        years = None
        for pattern in exp_patterns:
            matches = re.findall(pattern, resume_text.lower())
            if matches:
                # Take the maximum experience mentioned
                years = max([int(m) for m in matches if m.isdigit()])
                break
        
        if years is None:
            return 50, "Experience not specified"
        
        # Calculate score based on experience match
        if min_exp <= years <= max_exp:
            score = 100
            verdict = f"Perfect match ({years} years)"
        elif years < min_exp:
            # Penalize for under-experience
            gap = min_exp - years
            score = max(0, 100 - (gap * 20))  # -20 points per year gap
            verdict = f"Under-qualified ({years} years, need {min_exp}+)"
        else:
            # Slight bonus for over-experience
            score = 90
            verdict = f"Over-qualified ({years} years)"
        
        return score, verdict
    
    def calculate_education_match(
        self,
        resume_text: str,
        required_education: List[str]
    ) -> Tuple[float, List[str]]:
        """Calculate education matching score"""
        
        resume_lower = resume_text.lower()
        matched_education = []
        
        education_keywords = {
            "bachelor": ["bachelor", "b.tech", "b.e.", "bs", "bsc", "bca", "undergraduate"],
            "master": ["master", "m.tech", "m.e.", "ms", "msc", "mca", "mba", "graduate"],
            "phd": ["phd", "ph.d", "doctorate", "doctoral"],
            "diploma": ["diploma", "certification", "certificate"],
        }
        
        education_levels = {
            "phd": 4,
            "master": 3,
            "bachelor": 2,
            "diploma": 1
        }
        
        # Check for education matches
        for req_edu in required_education:
            req_edu_lower = req_edu.lower()
            
            # Check direct match
            if req_edu_lower in resume_lower:
                matched_education.append(req_edu)
                continue
            
            # Check using education keywords
            for edu_type, keywords in education_keywords.items():
                if any(keyword in req_edu_lower for keyword in keywords):
                    if any(keyword in resume_lower for keyword in keywords):
                        matched_education.append(req_edu)
                        break
        
        # Calculate score
        if required_education:
            score = (len(matched_education) / len(required_education)) * 100
        else:
            score = 100
        
        return score, matched_education
    
    def calculate_certification_match(
        self,
        resume_text: str,
        preferred_certifications: List[str]
    ) -> Tuple[float, List[str]]:
        """Check for relevant certifications"""
        
        resume_lower = resume_text.lower()
        matched_certs = []
        
        # Common certification patterns
        cert_patterns = {
            'aws': ['aws certified', 'amazon web services', 'solutions architect', 'cloud practitioner'],
            'azure': ['azure certified', 'microsoft certified', 'az-900', 'az-104', 'az-204'],
            'gcp': ['google cloud certified', 'gcp certified', 'professional cloud'],
            'pmp': ['pmp', 'project management professional'],
            'scrum': ['certified scrum master', 'csm', 'psm', 'scrum master'],
            'data': ['certified data', 'data scientist', 'data analyst', 'data engineer'],
            'security': ['security+', 'cissp', 'ceh', 'certified ethical hacker'],
            'devops': ['devops certified', 'devops engineer', 'sre certified'],
        }
        
        for cert in preferred_certifications:
            cert_lower = cert.lower()
            
            # Direct match
            if cert_lower in resume_lower:
                matched_certs.append(cert)
                continue
            
            # Pattern-based match
            for cert_type, patterns in cert_patterns.items():
                if cert_type in cert_lower:
                    if any(pattern in resume_lower for pattern in patterns):
                        matched_certs.append(cert)
                        break
        
        # Calculate score (certifications are usually bonus)
        if preferred_certifications:
            score = (len(matched_certs) / len(preferred_certifications)) * 100
        else:
            score = 0  # No bonus if no certifications required
        
        return score, matched_certs
    
    def calculate_keyword_density(
        self,
        resume_text: str,
        important_keywords: List[str]
    ) -> Tuple[float, Dict[str, int]]:
        """Calculate keyword density and frequency"""
        
        resume_lower = resume_text.lower()
        keyword_freq = {}
        
        for keyword in important_keywords:
            keyword_lower = keyword.lower()
            # Count occurrences
            count = len(re.findall(r'\b' + re.escape(keyword_lower) + r'\b', resume_lower))
            if count > 0:
                keyword_freq[keyword] = count
        
        # Calculate density score
        if important_keywords:
            matched_keywords = len(keyword_freq)
            score = (matched_keywords / len(important_keywords)) * 100
        else:
            score = 100
        
        return score, keyword_freq
    
    def calculate_project_relevance(
        self,
        resume_text: str,
        domain_keywords: List[str]
    ) -> Tuple[float, int]:
        """Check for relevant projects"""
        
        # Extract project section
        project_section = self._extract_section(resume_text, 'project')
        
        if not project_section:
            return 0, 0
        
        # Count relevant projects
        relevant_projects = 0
        project_blocks = self._split_into_blocks(project_section)
        
        for block in project_blocks:
            block_lower = block.lower()
            if any(keyword.lower() in block_lower for keyword in domain_keywords):
                relevant_projects += 1
        
        # Calculate score
        if relevant_projects == 0:
            score = 0
        elif relevant_projects == 1:
            score = 50
        elif relevant_projects == 2:
            score = 75
        else:
            score = 100
        
        return score, relevant_projects
    
    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract a specific section from resume"""
        
        section_headers = {
            'project': ['project', 'portfolio', 'work sample', 'notable work'],
            'experience': ['experience', 'employment', 'work history', 'professional experience'],
            'education': ['education', 'academic', 'qualification', 'degree'],
            'skill': ['skill', 'technical skill', 'competenc', 'expertise'],
        }
        
        text_lower = text.lower()
        headers = section_headers.get(section_name.lower(), [section_name.lower()])
        
        # Find section start
        start_pos = -1
        for header in headers:
            pattern = r'\b' + header + r's?\b'
            match = re.search(pattern, text_lower)
            if match:
                start_pos = match.start()
                break
        
        if start_pos == -1:
            return ""
        
        # Find next section start
        next_sections = ['experience', 'education', 'skill', 'project', 'certification', 
                        'achievement', 'reference', 'hobbies', 'interests']
        
        end_pos = len(text)
        for next_section in next_sections:
            if next_section not in headers:
                pattern = r'\b' + next_section + r's?\b'
                match = re.search(pattern, text_lower[start_pos + len(header):])
                if match:
                    end_pos = min(end_pos, start_pos + len(header) + match.start())
        
        return text[start_pos:end_pos]
    
    def _split_into_blocks(self, text: str) -> List[str]:
        """Split text into logical blocks (e.g., individual projects)"""
        
        # Split by bullet points, numbers, or multiple newlines
        blocks = re.split(r'\n\s*[•·▪▫◦‣⁃]\s*|\n\s*\d+\.\s*|\n\n+', text)
        
        # Filter out empty blocks
        blocks = [block.strip() for block in blocks if block.strip()]
        
        return blocks
    
    def calculate_comprehensive_score(
        self,
        resume_text: str,
        jd_requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate comprehensive hard match score"""
        
        results = {}
        
        # Skills matching
        skill_score, matched_skills, missing_skills = self.calculate_skill_match(
            jd_requirements.get('required_skills', []),
            resume_text
        )
        results['skill_score'] = skill_score
        results['matched_skills'] = matched_skills
        results['missing_skills'] = missing_skills
        
        # Experience matching
        exp_score, exp_verdict = self.calculate_experience_match(
            resume_text,
            jd_requirements.get('experience_range', {}).get('min', 0),
            jd_requirements.get('experience_range', {}).get('max', 10)
        )
        results['experience_score'] = exp_score
        results['experience_verdict'] = exp_verdict
        
        # Education matching
        edu_score, matched_education = self.calculate_education_match(
            resume_text,
            jd_requirements.get('education_requirements', [])
        )
        results['education_score'] = edu_score
        results['matched_education'] = matched_education
        
        # Certification matching
        cert_score, matched_certs = self.calculate_certification_match(
            resume_text,
            jd_requirements.get('preferred_certifications', [])
        )
        results['certification_score'] = cert_score
        results['matched_certifications'] = matched_certs
        
        # Keyword density
        keyword_score, keyword_freq = self.calculate_keyword_density(
            resume_text,
            jd_requirements.get('keywords', [])
        )
        results['keyword_score'] = keyword_score
        results['keyword_frequency'] = keyword_freq
        
        # Project relevance
        project_score, project_count = self.calculate_project_relevance(
            resume_text,
            jd_requirements.get('domain_keywords', jd_requirements.get('required_skills', []))
        )
        results['project_score'] = project_score
        results['relevant_projects'] = project_count
        
        # Calculate weighted total score
        weights = {
            'skill_score': 0.35,
            'experience_score': 0.25,
            'education_score': 0.15,
            'certification_score': 0.10,
            'keyword_score': 0.10,
            'project_score': 0.05
        }
        
        total_score = sum(
            results[key] * weight 
            for key, weight in weights.items()
        )
        
        results['total_hard_match_score'] = total_score
        results['weights_used'] = weights
        
        return results