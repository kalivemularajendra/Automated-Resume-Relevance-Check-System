import PyPDF2
import docx2txt
import re
from typing import Dict, Any, List
from io import BytesIO
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree

class ResumeParser:
    def __init__(self):
        self._download_nltk_data()
        try:
            self.stop_words = set(stopwords.words('english'))
            self.nltk_available = True
        except Exception as e:
            print(f"⚠️ Warning: NLTK stopwords not available, using fallback: {e}")
            # Fallback stopwords list
            self.stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
                              'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
                              'to', 'was', 'will', 'with', 'have', 'had', 'this', 'they', 'them',
                              'their', 'we', 'you', 'your', 'but', 'not', 'or', 'do', 'does',
                              'did', 'can', 'could', 'would', 'should', 'may', 'might', 'must'}
            self.nltk_available = False
        print("✅ ResumeParser initialized with enhanced text processing")
    
    def _download_nltk_data(self):
        """Download required NLTK data with improved error handling"""
        required_data = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']
        
        for data in required_data:
            try:
                # Try to download each required dataset directly
                # This approach is more reliable than searching through various paths
                print(f"📥 Ensuring NLTK data is available: {data}")
                nltk.download(data, quiet=True)
            except Exception as e:
                print(f"⚠️ Warning: Could not download NLTK data '{data}': {e}")
                # Continue with other data downloads even if one fails
    
    def extract_text(self, file) -> str:
        """Extract text from uploaded file"""
        text = ""
        
        try:
            if file.type == "application/pdf":
                try:
                    pdf_reader = PyPDF2.PdfReader(BytesIO(file.read()))
                    if len(pdf_reader.pages) == 0:
                        return "Error: PDF file appears to be empty or corrupted (no pages found)."
                    
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text.strip():  # Only add non-empty pages
                            text += page_text + "\n"
                    
                    if not text.strip():
                        return "Error: Could not extract readable text from PDF. The file may be image-based or corrupted."
                        
                except PyPDF2.errors.EmptyFileError:
                    return "Error: PDF file is empty or corrupted."
                except PyPDF2.errors.PdfReadError as e:
                    return f"Error: Unable to read PDF file. {str(e)}"
                except Exception as e:
                    return f"Error: Unexpected error reading PDF. {str(e)}"
            
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                try:
                    text = docx2txt.process(BytesIO(file.read()))
                    if not text.strip():
                        return "Error: Could not extract text from Word document. The file may be empty or corrupted."
                except Exception as e:
                    return f"Error: Unable to read Word document. {str(e)}"
            
            else:
                return f"Error: Unsupported file type '{file.type}'. Please upload a PDF or Word document."
                
        except Exception as e:
            return f"Error: Failed to process file. {str(e)}"
        
        # Clean the extracted text
        cleaned_text = self.clean_text(text)
        if not cleaned_text.strip():
            return "Error: No readable text found in the file. The document may be image-based or corrupted."
            
        return cleaned_text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\-\.\,\@\+\#]', '', text)
        return text.strip()
    
    def extract_email(self, text: str) -> str:
        """Extract email from text"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        return emails[0] if emails else ""
    
    def extract_phone(self, text: str) -> str:
        """Extract phone number from text"""
        phone_pattern = r'[\+KATEX_INLINE_OPEN]?[1-9][0-9 .\-KATEX_INLINE_OPENKATEX_INLINE_CLOSE]{8,}[0-9]'
        phones = re.findall(phone_pattern, text)
        return phones[0] if phones else ""
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from text using NLTK and pattern matching with fallback"""
        # Enhanced skill patterns with synonyms
        skill_patterns = {
            'python': ['python', 'py', 'python3', 'python2'],
            'java': ['java', 'jvm', 'java8', 'java11'],
            'javascript': ['javascript', 'js', 'es6', 'ecmascript'],
            'react': ['react', 'reactjs', 'react.js'],
            'node': ['node', 'nodejs', 'node.js'],
            'sql': ['sql', 'mysql', 'postgresql', 'sqlite', 'tsql'],
            'mongodb': ['mongodb', 'mongo', 'nosql'],
            'machine learning': ['machine learning', 'ml', 'machine-learning'],
            'deep learning': ['deep learning', 'dl', 'neural networks'],
            'data science': ['data science', 'data scientist', 'data analysis'],
            'aws': ['aws', 'amazon web services', 'amazon cloud'],
            'docker': ['docker', 'containerization', 'containers'],
            'kubernetes': ['kubernetes', 'k8s', 'container orchestration'],
            'git': ['git', 'github', 'gitlab', 'version control'],
            'agile': ['agile', 'scrum', 'kanban'],
            'tensorflow': ['tensorflow', 'tf', 'keras'],
            'pytorch': ['pytorch', 'torch'],
            'angular': ['angular', 'angularjs', 'angular.js'],
            'vue': ['vue', 'vuejs', 'vue.js'],
            'c++': ['c++', 'cpp', 'c plus plus'],
            'c#': ['c#', 'csharp', 'c sharp'],
            'azure': ['azure', 'microsoft azure'],
            'gcp': ['gcp', 'google cloud', 'google cloud platform']
        }
        
        # Tokenize text using NLTK or fallback method
        try:
            if self.nltk_available:
                tokens = word_tokenize(text.lower())
            else:
                # Fallback tokenization using regex
                tokens = re.findall(r'\b\w+\b', text.lower())
        except Exception as e:
            print(f"⚠️ Warning: NLTK tokenization failed, using fallback: {e}")
            tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Remove stopwords
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
        
        found_skills = []
        text_lower = text.lower()
        
        # Check for skill patterns
        for skill, variants in skill_patterns.items():
            for variant in variants:
                if variant in text_lower:
                    found_skills.append(skill)
                    break
        
        return found_skills
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using NLTK with fallback"""
        entities = {
            'persons': [],
            'organizations': [],
            'locations': [],
            'dates': []
        }
        
        try:
            if self.nltk_available:
                # Tokenize and tag parts of speech
                tokens = word_tokenize(text)
                pos_tags = pos_tag(tokens)
                
                # Named entity chunking
                chunks = ne_chunk(pos_tags)
                
                for chunk in chunks:
                    if isinstance(chunk, Tree):
                        entity_name = ' '.join([token for token, pos in chunk.leaves()])
                        if chunk.label() == 'PERSON':
                            entities['persons'].append(entity_name)
                        elif chunk.label() in ['ORGANIZATION', 'GPE']:
                            entities['organizations'].append(entity_name)
                        elif chunk.label() in ['GPE', 'LOCATION']:
                            entities['locations'].append(entity_name)
            else:
                print("⚠️ NLTK entity recognition not available, using fallback patterns")
                # Fallback: use simple pattern matching for common entities
                
                # Simple person name patterns (capitalized words)
                person_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
                persons = re.findall(person_pattern, text)
                entities['persons'] = persons[:5]  # Limit to first 5 matches
                
                # Common organization patterns
                org_patterns = ['Inc\.', 'LLC', 'Corp\.', 'Company', 'University', 'College']
                for pattern in org_patterns:
                    matches = re.findall(rf'\b\w+\s+{pattern}\b', text, re.IGNORECASE)
                    entities['organizations'].extend(matches[:3])
            
            # Extract dates using regex (works regardless of NLTK)
            date_patterns = [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY or MM-DD-YYYY
                r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',    # YYYY/MM/DD or YYYY-MM-DD
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}\b',  # Month DD, YYYY
                r'\b\d{4}\b'  # Just year
            ]
            
            for pattern in date_patterns:
                dates = re.findall(pattern, text, re.IGNORECASE)
                entities['dates'].extend(dates)
                
        except Exception as e:
            print(f"⚠️ Entity extraction error: {e}")
        
        return entities
    
    def extract_education(self, text: str) -> List[str]:
        """Extract education information using NLTK and patterns"""
        education_keywords = [
            'bachelor', 'master', 'phd', 'doctorate', 'diploma', 'degree',
            'university', 'college', 'institute', 'school', 'education',
            'b.s', 'b.a', 'm.s', 'm.a', 'mba', 'ph.d'
        ]
        
        education_info = []
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in education_keywords):
                education_info.append(sentence.strip())
        
        return education_info
    
    def extract_experience(self, text: str) -> List[str]:
        """Extract work experience using NLTK and patterns"""
        experience_keywords = [
            'experience', 'worked', 'employed', 'position', 'role',
            'company', 'organization', 'years', 'months'
        ]
        
        experience_info = []
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Look for sentences with experience indicators and time periods
            if (any(keyword in sentence_lower for keyword in experience_keywords) and
                re.search(r'\b\d+\s*(year|month|yr|mo)', sentence_lower)):
                experience_info.append(sentence.strip())
        
        return experience_info

    def parse_sections(self, text: str) -> Dict[str, str]:
        """Parse resume into sections using NLTK-enhanced detection"""
        sections = {
            'education': '',
            'experience': '',
            'skills': '',
            'projects': '',
            'certifications': ''
        }
        
        # Simple section extraction (can be improved with ML)
        section_headers = {
            'education': ['education', 'academic', 'qualification'],
            'experience': ['experience', 'work history', 'employment'],
            'skills': ['skills', 'technical skills', 'competencies'],
            'projects': ['projects', 'portfolio'],
            'certifications': ['certifications', 'certificates', 'licenses']
        }
        
        lines = text.split('\n')
        current_section = None
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if line is a section header
            for section, headers in section_headers.items():
                if any(header in line_lower for header in headers):
                    current_section = section
                    break
            
            # Add content to current section
            if current_section:
                sections[current_section] += line + '\n'
        
        return sections
    
    def parse(self, file) -> Dict[str, Any]:
        """Main parsing method using NLTK-enhanced processing"""
        # Extract text from file
        text = self.extract_text(file)
        
        if text.startswith("Error:"):
            return {"error": text}
        
        try:
            # Basic information extraction
            parsed_data = {
                'text': text,
                'email': self.extract_email(text),
                'phone': self.extract_phone(text),
                'skills': self.extract_skills(text),
                'sections': self.parse_sections(text)
            }
            
            # NLTK-enhanced extraction
            entities = self.extract_entities(text)
            parsed_data['entities'] = entities
            parsed_data['education_sentences'] = self.extract_education(text)
            parsed_data['experience_sentences'] = self.extract_experience(text)
            
            # Extract potential name from entities
            if entities['persons']:
                parsed_data['candidate_name'] = entities['persons'][0]
            else:
                parsed_data['candidate_name'] = "Not found"
            
            # Extract companies from entities
            parsed_data['companies'] = entities['organizations']
            
            # Extract locations
            parsed_data['locations'] = entities['locations']
            
            # Extract dates
            parsed_data['dates'] = entities['dates']
            
            # Text statistics using NLTK
            tokens = word_tokenize(text)
            sentences = sent_tokenize(text)
            
            parsed_data['statistics'] = {
                'word_count': len([token for token in tokens if token.isalpha()]),
                'sentence_count': len(sentences),
                'total_tokens': len(tokens)
            }
            
            return parsed_data
            
        except Exception as e:
            return {"error": f"Parsing failed: {str(e)}"}
