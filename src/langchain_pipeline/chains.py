from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.callbacks import LangChainTracer
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json

# Pydantic models for structured outputs
class SkillExtraction(BaseModel):
    technical_skills: List[str] = Field(description="Technical skills mentioned")
    soft_skills: List[str] = Field(description="Soft skills mentioned")
    tools: List[str] = Field(description="Tools and technologies")
    certifications: List[str] = Field(description="Certifications mentioned")

class ExperienceExtraction(BaseModel):
    total_years: float = Field(description="Total years of experience")
    relevant_years: float = Field(description="Years of relevant experience")
    companies: List[str] = Field(description="List of companies worked at")
    roles: List[str] = Field(description="Job roles held")
    
class ProjectExtraction(BaseModel):
    project_titles: List[str] = Field(description="Project titles")
    technologies_used: List[str] = Field(description="Technologies used in projects")
    project_descriptions: List[str] = Field(description="Brief project descriptions")

class GapAnalysis(BaseModel):
    missing_skills: List[str] = Field(description="Skills required but not found")
    partial_matches: List[str] = Field(description="Skills partially matched")
    additional_skills: List[str] = Field(description="Additional skills candidate has")
    recommendations: List[str] = Field(description="Specific recommendations")

class ResumeChains:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.2,
            max_output_tokens=2048
        )
        
        # Initialize output parsers
        self.skill_parser = PydanticOutputParser(pydantic_object=SkillExtraction)
        self.experience_parser = PydanticOutputParser(pydantic_object=ExperienceExtraction)
        self.project_parser = PydanticOutputParser(pydantic_object=ProjectExtraction)
        self.gap_parser = PydanticOutputParser(pydantic_object=GapAnalysis)
        
        # Initialize chains
        self.skill_extraction_chain = self._create_skill_extraction_chain()
        self.experience_extraction_chain = self._create_experience_extraction_chain()
        self.project_extraction_chain = self._create_project_extraction_chain()
        self.gap_analysis_chain = self._create_gap_analysis_chain()
        self.feedback_generation_chain = self._create_feedback_generation_chain()
        
    def _create_skill_extraction_chain(self) -> LLMChain:
        """Create chain for extracting skills from resume"""
        
        system_template = """You are an expert resume analyst specializing in skill extraction.
        Extract all technical skills, soft skills, tools, and certifications from the resume.
        Be comprehensive and accurate."""
        
        human_template = """Extract skills from this resume:

Resume Text:
{resume_text}

{format_instructions}

Output:"""
        
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template)
            ],
            input_variables=["resume_text"],
            partial_variables={"format_instructions": self.skill_parser.get_format_instructions()}
        )
        
        return LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_parser=OutputFixingParser.from_llm(parser=self.skill_parser, llm=self.llm)
        )
    
    def _create_experience_extraction_chain(self) -> LLMChain:
        """Create chain for extracting experience information"""
        
        system_template = """You are an expert resume analyst specializing in work experience analysis.
        Extract experience details including years of experience, companies, and roles.
        Calculate both total and relevant experience based on the job requirements."""
        
        human_template = """Extract experience information from this resume:

Resume Text:
{resume_text}

Job Requirements (for relevance assessment):
{job_requirements}

{format_instructions}

Output:"""
        
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template)
            ],
            input_variables=["resume_text", "job_requirements"],
            partial_variables={"format_instructions": self.experience_parser.get_format_instructions()}
        )
        
        return LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_parser=OutputFixingParser.from_llm(parser=self.experience_parser, llm=self.llm)
        )
    
    def _create_project_extraction_chain(self) -> LLMChain:
        """Create chain for extracting project information"""
        
        system_template = """You are an expert resume analyst specializing in project evaluation.
        Extract all project information including titles, technologies used, and descriptions.
        Focus on projects relevant to software development and technical roles."""
        
        human_template = """Extract project information from this resume:

Resume Text:
{resume_text}

{format_instructions}

Output:"""
        
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template)
            ],
            input_variables=["resume_text"],
            partial_variables={"format_instructions": self.project_parser.get_format_instructions()}
        )
        
        return LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_parser=OutputFixingParser.from_llm(parser=self.project_parser, llm=self.llm)
        )
    
    def _create_gap_analysis_chain(self) -> LLMChain:
        """Create chain for gap analysis between resume and JD"""
        
        system_template = """You are an expert career counselor performing gap analysis.
        Compare the candidate's profile with job requirements and identify:
        1. Missing skills that are required
        2. Partially matched skills that need improvement
        3. Additional skills the candidate has beyond requirements
        4. Specific actionable recommendations for improvement"""
        
        human_template = """Perform gap analysis:

Candidate Skills:
{candidate_skills}

Job Requirements:
{job_requirements}

{format_instructions}

Output:"""
        
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template)
            ],
            input_variables=["candidate_skills", "job_requirements"],
            partial_variables={"format_instructions": self.gap_parser.get_format_instructions()}
        )
        
        return LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_parser=OutputFixingParser.from_llm(parser=self.gap_parser, llm=self.llm)
        )
    
    def _create_feedback_generation_chain(self) -> LLMChain:
        """Create chain for generating personalized feedback"""
        
        system_template = """You are a career counselor providing constructive feedback to job seekers.
        Based on the evaluation results, provide:
        1. Positive reinforcement for strengths
        2. Specific areas for improvement
        3. Actionable steps to increase their chances
        4. Resources or certifications to consider
        5. Resume optimization tips
        
        Be encouraging, specific, and practical."""
        
        human_template = """Generate personalized feedback for this candidate:

Evaluation Summary:
- Overall Score: {score}%
- Verdict: {verdict}
- Matched Skills: {matched_skills}
- Missing Skills: {missing_skills}
- Experience Match: {experience_match}

Gap Analysis:
{gap_analysis}

Generate comprehensive feedback with:
1. Congratulations on strengths (if any)
2. Critical gaps to address
3. 5 specific action items
4. Recommended timeline for improvement
5. Resources to explore

Feedback:"""
        
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template)
            ],
            input_variables=[
                "score", "verdict", "matched_skills", 
                "missing_skills", "experience_match", "gap_analysis"
            ]
        )
        
        return LLMChain(llm=self.llm, prompt=prompt)
    
    async def extract_all_information(
        self,
        resume_text: str,
        job_requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract all information from resume using chains"""
        
        results = {}
        
        # Extract skills
        try:
            skills = await self.skill_extraction_chain.arun(resume_text=resume_text)
            results['skills'] = skills.dict() if hasattr(skills, 'dict') else skills
        except Exception as e:
            print(f"Error extracting skills: {e}")
            results['skills'] = {}
        
        # Extract experience
        try:
            experience = await self.experience_extraction_chain.arun(
                resume_text=resume_text,
                job_requirements=json.dumps(job_requirements)
            )
            results['experience'] = experience.dict() if hasattr(experience, 'dict') else experience
        except Exception as e:
            print(f"Error extracting experience: {e}")
            results['experience'] = {}
        
        # Extract projects
        try:
            projects = await self.project_extraction_chain.arun(resume_text=resume_text)
            results['projects'] = projects.dict() if hasattr(projects, 'dict') else projects
        except Exception as e:
            print(f"Error extracting projects: {e}")
            results['projects'] = {}
        
        return results
    
    async def perform_gap_analysis(
        self,
        candidate_info: Dict[str, Any],
        job_requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform gap analysis between candidate and job requirements"""
        
        try:
            gap_analysis = await self.gap_analysis_chain.arun(
                candidate_skills=json.dumps(candidate_info),
                job_requirements=json.dumps(job_requirements)
            )
            return gap_analysis.dict() if hasattr(gap_analysis, 'dict') else gap_analysis
        except Exception as e:
            print(f"Error performing gap analysis: {e}")
            return {
                'missing_skills': [],
                'partial_matches': [],
                'additional_skills': [],
                'recommendations': ['Unable to perform detailed gap analysis']
            }
    
    async def generate_feedback(
        self,
        evaluation_results: Dict[str, Any]
    ) -> str:
        """Generate personalized feedback for the candidate"""
        
        try:
            feedback = await self.feedback_generation_chain.arun(
                score=evaluation_results.get('relevance_score', 0),
                verdict=evaluation_results.get('verdict', 'Unknown'),
                matched_skills=', '.join(evaluation_results.get('matched_skills', [])[:5]),
                missing_skills=', '.join(evaluation_results.get('missing_skills', [])[:5]),
                experience_match=evaluation_results.get('experience_match', 'Not specified'),
                gap_analysis=json.dumps(evaluation_results.get('gap_analysis', {}))
            )
            return feedback
        except Exception as e:
            print(f"Error generating feedback: {e}")
            return "Thank you for your application. Please review the detailed scores above for specific feedback."

class JDChains:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.1
        )
        
        self.requirement_extraction_chain = self._create_requirement_extraction_chain()
        self.keyword_extraction_chain = self._create_keyword_extraction_chain()
        
    def _create_requirement_extraction_chain(self) -> LLMChain:
        """Create chain for extracting requirements from JD"""
        
        template = """Extract detailed requirements from this job description:

Job Description:
{jd_text}

Extract and structure:
1. Required technical skills (must-have)
2. Preferred skills (nice-to-have)
3. Years of experience required (min and max)
4. Educational requirements
5. Key responsibilities
6. Domain/Industry context
7. Soft skills required
8. Certifications preferred

Format as JSON with clear categories."""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        return LLMChain(llm=self.llm, prompt=prompt)
    
    def _create_keyword_extraction_chain(self) -> LLMChain:
        """Create chain for extracting keywords from JD"""
        
        template = """Extract important keywords and phrases from this job description for resume matching:

Job Description:
{jd_text}

Extract:
1. Technical keywords (technologies, tools, frameworks)
2. Action verbs (develop, implement, design, etc.)
3. Domain-specific terms
4. Qualification keywords
5. Soft skill keywords

Return as a comprehensive list of keywords for matching."""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        return LLMChain(llm=self.llm, prompt=prompt)
    
    async def extract_requirements(self, jd_text: str) -> Dict[str, Any]:
        """Extract structured requirements from JD"""
        
        try:
            result = await self.requirement_extraction_chain.arun(jd_text=jd_text)
            # Parse JSON from response
            return json.loads(result) if isinstance(result, str) else result
        except Exception as e:
            print(f"Error extracting requirements: {e}")
            return {}
    
    async def extract_keywords(self, jd_text: str) -> List[str]:
        """Extract keywords from JD"""
        
        try:
            result = await self.keyword_extraction_chain.arun(jd_text=jd_text)
            # Parse keywords from response
            if isinstance(result, str):
                keywords = [k.strip() for k in result.split(',')]
                return keywords
            return result
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return []