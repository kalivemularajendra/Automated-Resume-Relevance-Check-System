import re
from typing import Dict, List, Any
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import PyPDF2
import docx2txt
from io import BytesIO

class JDRequirements(BaseModel):
    role_title: str = Field(description="Job title/position name")
    company: str = Field(description="Company name", default="Not specified")
    location: str = Field(description="Job location", default="Not specified")
    required_skills: List[str] = Field(description="List of required technical skills")
    preferred_skills: List[str] = Field(description="List of preferred/nice-to-have skills")
    experience_range: Dict[str, int] = Field(description="Min and max years of experience")
    education_requirements: List[str] = Field(description="Required educational qualifications")
    responsibilities: List[str] = Field(description="Key responsibilities")
    keywords: List[str] = Field(description="Important keywords from JD")

class MultipleJDs(BaseModel):
    job_descriptions: List[JDRequirements] = Field(description="List of job descriptions found in the document")
    total_count: int = Field(description="Total number of job descriptions found")

class JDParser:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
        )
        self.parser = PydanticOutputParser(pydantic_object=JDRequirements)
        self.multi_parser = PydanticOutputParser(pydantic_object=MultipleJDs)
    
    def extract_text(self, file) -> str:
        """Extract text from uploaded JD file"""
        text = ""
        
        if file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(BytesIO(file.read()))
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = docx2txt.process(BytesIO(file.read()))
        else:
            text = str(file.read(), "utf-8")
        
        return text
    
    def detect_multiple_jds(self, text: str) -> bool:
        """Detect if the document contains multiple job descriptions"""
        # Look for common separators and indicators of multiple JDs
        indicators = [
            r'job\s*(?:opening|position|description|title)\s*[#\d:]',
            r'position\s*[#\d:]',
            r'role\s*[#\d:]',
            r'vacancy\s*[#\d:]',
            r'we\s*are\s*hiring.*?for',
            r'open\s*positions?',
            r'available\s*roles?'
        ]
        
        count = 0
        for pattern in indicators:
            matches = re.findall(pattern, text, re.IGNORECASE)
            count += len(matches)
        
        # Also check for common section separators
        separators = [
            r'-{3,}',  # Multiple dashes
            r'={3,}',  # Multiple equals signs
            r'\*{3,}', # Multiple asterisks
            r'#{2,}',  # Multiple hash symbols
            r'position\s*\d+',
            r'job\s*\d+',
            r'\d+\.\s+[A-Z]',  # Numbered lists like "1. Title"
            r'\d+\)\s+[A-Z]',  # Numbered lists like "1) Title"
        ]
        
        separator_count = 0
        for pattern in separators:
            matches = re.findall(pattern, text, re.IGNORECASE)
            separator_count += len(matches)
        
        return count > 1 or separator_count > 1
    
    def split_multiple_jds(self, text: str) -> List[str]:
        """Split document containing multiple JDs into individual JD texts"""
        # Common patterns that separate different job descriptions
        split_patterns = [
            r'(?=job\s*(?:opening|position|description|title)\s*[#\d:])',
            r'(?=position\s*[#\d:])',
            r'(?=role\s*[#\d:])',
            r'(?=vacancy\s*[#\d:])',
            r'-{5,}',  # 5 or more dashes
            r'={5,}',  # 5 or more equals signs
            r'\*{5,}', # 5 or more asterisks
            r'#{3,}',  # 3 or more hash symbols
            r'(?=position\s*\d+)',
            r'(?=job\s*\d+)',
            r'(?=\d+\.\s+[A-Z])',  # Numbered lists like "1. Title"
            r'(?=\d+\)\s+[A-Z])',  # Numbered lists like "1) Title"
        ]
        
        # Try different split patterns
        jd_sections = [text]  # Start with the whole text
        
        for pattern in split_patterns:
            temp_sections = []
            for section in jd_sections:
                split_result = re.split(pattern, section, flags=re.IGNORECASE)
                # Filter out empty sections and very short ones
                filtered_splits = [s.strip() for s in split_result if len(s.strip()) > 200]
                temp_sections.extend(filtered_splits)
            
            # If we got more sections and they seem reasonable, use them
            if len(temp_sections) > len(jd_sections) and all(len(s) > 300 for s in temp_sections):
                jd_sections = temp_sections
                break
        
        # If no good splits found, try paragraph-based splitting
        if len(jd_sections) == 1:
            paragraphs = text.split('\n\n')
            current_jd = ""
            jd_sections = []
            
            for para in paragraphs:
                if any(keyword in para.lower() for keyword in ['position', 'job', 'role', 'vacancy', 'hiring']) and len(current_jd) > 500:
                    if current_jd.strip():
                        jd_sections.append(current_jd.strip())
                    current_jd = para
                else:
                    current_jd += "\n\n" + para
            
            if current_jd.strip():
                jd_sections.append(current_jd.strip())
        
        # Filter out sections that are too short to be meaningful JDs
        jd_sections = [jd for jd in jd_sections if len(jd.strip()) > 300]
        
        return jd_sections if len(jd_sections) > 1 else [text]
    
    async def parse_multiple_jds(self, jd_text: str) -> List[Dict[str, Any]]:
        """Parse document with multiple JDs"""
        if not self.detect_multiple_jds(jd_text):
            # Single JD - use existing method
            single_jd = await self.parse_with_llm(jd_text)
            return [single_jd]
        
        # Multiple JDs detected - split and parse each
        jd_sections = self.split_multiple_jds(jd_text)
        parsed_jds = []
        
        for i, section in enumerate(jd_sections):
            try:
                parsed_jd = await self.parse_with_llm(section)
                # Add section number for identification
                parsed_jd['section_number'] = i + 1
                parsed_jds.append(parsed_jd)
            except Exception as e:
                print(f"Error parsing JD section {i+1}: {e}")
                # Use fallback parsing
                fallback_jd = self.fallback_parse(section)
                fallback_jd['section_number'] = i + 1
                fallback_jd['role_title'] = f"Position {i + 1}"
                parsed_jds.append(fallback_jd)
        
        return parsed_jds
    
    async def parse_with_llm(self, jd_text: str) -> Dict[str, Any]:
        """Parse JD using LLM to extract structured information"""
        
        prompt = PromptTemplate(
            template="""Extract the following information from this job description:

Job Description:
{jd_text}

{format_instructions}

Provide accurate extraction focusing on:
1. Job title/position name
2. Company name (if mentioned)
3. Job location (if mentioned)
4. Required technical skills (must-have)
5. Preferred skills (nice-to-have)
6. Experience requirements (min and max years)
7. Educational qualifications
8. Key responsibilities
9. Important keywords for matching

If company or location is not explicitly mentioned, use "Not specified".

Output:""",
            input_variables=["jd_text"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        chain = prompt | self.llm | self.parser
        
        try:
            result = await chain.ainvoke({"jd_text": jd_text})
            return result.dict()
        except Exception as e:
            print(f"Error parsing JD: {e}")
            return self.fallback_parse(jd_text)
    
    def fallback_parse(self, jd_text: str) -> Dict[str, Any]:
        """Fallback parsing method using regex and keywords"""
        
        text_lower = jd_text.lower()
        
        # Extract job title (look for common patterns)
        title_patterns = [
            r'position[:\s]+([^\n\r\.]+)',
            r'role[:\s]+([^\n\r\.]+)',
            r'job title[:\s]+([^\n\r\.]+)',
            r'we are hiring[:\s]+([^\n\r\.]+)',
            r'looking for[:\s]+([^\n\r\.]+)'
        ]
        
        role_title = "Not specified"
        for pattern in title_patterns:
            match = re.search(pattern, jd_text, re.IGNORECASE)
            if match:
                role_title = match.group(1).strip()
                break
        
        # Extract company name
        company_patterns = [
            r'company[:\s]+([^\n\r\.]+)',
            r'organization[:\s]+([^\n\r\.]+)',
            r'at ([A-Z][a-zA-Z\s&]+)(?:\s+is|\s+are|\s+has|\s+was)',
        ]
        
        company = "Not specified"
        for pattern in company_patterns:
            match = re.search(pattern, jd_text, re.IGNORECASE)
            if match:
                company = match.group(1).strip()
                break
        
        # Extract location
        location_patterns = [
            r'location[:\s]+([^\n\r\.]+)',
            r'based in[:\s]+([^\n\r\.]+)',
            r'office[:\s]+([^\n\r\.]+)'
        ]
        
        location = "Not specified"
        for pattern in location_patterns:
            match = re.search(pattern, jd_text, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                break
        
        # Extract skills
        skill_patterns = [
            'python', 'java', 'javascript', 'react', 'angular', 'vue',
            'node', 'express', 'django', 'flask', 'sql', 'nosql',
            'mongodb', 'postgresql', 'mysql', 'redis', 'docker',
            'kubernetes', 'aws', 'azure', 'gcp', 'git', 'ci/cd',
            'machine learning', 'deep learning', 'nlp', 'computer vision'
        ]
        
        found_skills = [skill for skill in skill_patterns if skill in text_lower]
        
        # Extract experience
        exp_pattern = r'(\d+)[\+\-]?\s*(?:to|-)?\s*(\d+)?\s*years?'
        exp_matches = re.findall(exp_pattern, text_lower)
        
        min_exp = 0
        max_exp = 10
        if exp_matches:
            min_exp = int(exp_matches[0][0])
            max_exp = int(exp_matches[0][1]) if exp_matches[0][1] else min_exp + 3
        
        return {
            'role_title': role_title,
            'company': company,
            'location': location,
            'required_skills': found_skills[:5],
            'preferred_skills': found_skills[5:10] if len(found_skills) > 5 else [],
            'experience_range': {'min': min_exp, 'max': max_exp},
            'education_requirements': ['Bachelor\'s degree'],
            'responsibilities': [],
            'keywords': found_skills
        }