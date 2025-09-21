# Resume Relevance Check System

A modern, AI-powered web application for evaluating the relevance of resumes against job descriptions. Built with **Streamlit**, **LangChain**, **Google Gemini**, and advanced NLP techniques, this tool streamlines the candidate screening process for recruiters and HR professionals.

---

## 🚀 Features

- **Job Description Parsing**: Extracts and analyzes single or multiple job descriptions from PDF/DOCX files.
- **Resume Parsing**: Extracts text, skills, and entities from resumes (PDF/DOCX) using spaCy.
- **AI-Powered Evaluation**: Uses LLMs (Google Gemini), semantic embeddings, and keyword/BM25/fuzzy matching for robust scoring.
- **Weighted Scoring**: Combines hard and semantic matches for a final relevance score.
- **Batch Processing**: Evaluate multiple resumes at once.
- **Analytics Dashboard**: Visualizes score distributions, trends, and insights.
- **Database Persistence**: Stores job descriptions, resumes, and evaluation results in SQLite.
- **Observability**: Integrated with LangSmith for LLM chain tracing and debugging.
- **Dark Mode UI**: Clean, responsive, and accessible interface.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit (with custom CSS for dark mode)
- **Backend/Orchestration**: Python, LangChain, LangGraph
- **LLM**: Google Gemini (via LangChain)
- **NLP**: spaCy
- **Keyword Matching**: TF-IDF, BM25, FuzzyWuzzy
- **Semantic Search**: Google AI Embeddings, ChromaDB
- **Database**: SQLite (via SQLAlchemy)
- **Observability**: LangSmith
- **Other**: python-dotenv, docx2txt, PyPDF2

---

## 📦 Installation

1. **Clone the repository**
<<<<<<< HEAD

=======
>>>>>>> 78fa01b97742b30198b158e950eb4dd4057f88be
   ```sh
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```

2. **Create a virtual environment**
<<<<<<< HEAD

=======
>>>>>>> 78fa01b97742b30198b158e950eb4dd4057f88be
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
<<<<<<< HEAD

=======
>>>>>>> 78fa01b97742b30198b158e950eb4dd4057f88be
   ```sh
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Copy `.env.example` to `.env` and fill in your API keys and settings.
   - **Never commit your `.env` file!**

---

## ⚙️ Configuration

Create a `.env` file in the project root with the following structure:

```env
# Google AI API Key
GOOGLE_API_KEY="your-google-api-key"

# LangSmith Configuration
LANGCHAIN_TRACING_V2="True"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your-langsmith-api-key"

# Database Configuration
DATABASE_URL=sqlite:///resume_evaluations.db

# Application Settings
APP_ENV=production
DEBUG=False
MAX_UPLOAD_SIZE=10485760
BATCH_PROCESSING_LIMIT=100
```

---

<<<<<<< HEAD
## 🚀 Deployment on Streamlit Cloud

### Prerequisites

- GitHub repository with your code
- Streamlit Cloud account

### Steps

1. **Push your code to GitHub**

   ```sh
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**

   - Go to [Streamlit Cloud](https://share.streamlit.io/)
   - Click "New app" and connect your GitHub repository
   - Set the main file path to `app.py`
   - Click "Deploy"

3. **Configure Secrets**

   - After deployment, go to your app page
   - Click the "⋮" menu and select "Edit Secrets"
   - Add your environment variables (without quotes):
     ```
     GOOGLE_API_KEY=your-google-api-key
     LANGCHAIN_TRACING_V2=True
     LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
     LANGCHAIN_API_KEY=your-langsmith-api-key
     DATABASE_URL=sqlite:///resume_evaluations.db
     APP_ENV=production
     DEBUG=False
     MAX_UPLOAD_SIZE=10485760
     BATCH_PROCESSING_LIMIT=100
     ```
   - Save the secrets

4. **Redeploy**
   - Streamlit Cloud will automatically redeploy with the new secrets

### Troubleshooting

- If you get spaCy model errors, the `packages.txt` and updated `requirements.txt` should handle this
- Make sure all secrets are properly set in the Streamlit Cloud secrets editor
- Check the deployment logs for any missing dependencies

---

## 🖥️ Local Development

1. **Run the Streamlit app**

=======
## 🖥️ Usage

1. **Run the Streamlit app**
>>>>>>> 78fa01b97742b30198b158e950eb4dd4057f88be
   ```sh
   streamlit run app.py
   ```

2. **Open your browser** and go to [http://localhost:8501](http://localhost:8501)

3. **Workflow:**
   - **Upload Job Description(s)**: Go to the JD Management page and upload your JD file(s).
   - **Upload Resume(s)**: Go to the Resume Evaluation page and upload candidate resumes.
   - **View Results**: See scores, verdicts, and analytics in real time.
   - **Batch Processing**: Upload multiple resumes for bulk evaluation.
   - **Analytics**: Explore score distributions, trends, and company insights.

---

## 📊 Screenshots

> _Add screenshots of the main pages here for better documentation._

---

## 🧩 Project Structure

```
.
├── app.py
├── requirements.txt
<<<<<<< HEAD
├── packages.txt
├── .env.example
├── .gitignore
├── README.md
=======
├── .env.example
├── .gitignore
>>>>>>> 78fa01b97742b30198b158e950eb4dd4057f88be
├── src/
│   ├── parsers/
│   │   ├── resume_parser.py
│   │   └── jd_parser.py
│   ├── scoring/
│   │   ├── score_engine.py
│   │   ├── hard_match.py
│   │   └── semantic_match.py
│   ├── langchain_pipeline/
│   │   ├── graph.py
│   │   └── chains.py
│   ├── database/
│   │   └── db_handler.py
│   └── utils/
│       └── helpers.py
└── .streamlit/
    └── config.toml
```

---

## 📝 Notes

- **Security**: Do NOT commit your `.env` file or any file containing secrets.
- **Extensibility**: The system is modular—add new scoring methods, LLMs, or analytics easily.
- **Customization**: You can adjust scoring weights, add new entity extractors, or connect to a different database.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

<<<<<<< HEAD
## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

=======
>>>>>>> 78fa01b97742b30198b158e950eb4dd4057f88be
## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [Google Gemini](https://ai.google.dev/)
- [spaCy](https://spacy.io/)
- [ChromaDB](https://www.trychroma.com/)
- [FuzzyWuzzy](https://github.com/seatgeek/fuzzywuzzy)
- [BM25](https://github.com/dorianbrown/rank_bm25)
- [LangSmith](https://smith.langchain.com/)

---

**Enjoy using the Resume Relevance Check System!**
