# Intelligent Learning Analytics and Agentic Study Coach

Intelligent Learning Analytics is a Streamlit-based application for student performance analysis and personalized study planning.  
It combines classical machine learning for risk classification with an agentic AI coaching workflow that generates actionable, structured learning plans.

## Datasets

- Students Performance in Exams (Kaggle): [https://www.kaggle.com/datasets/spscientist/students-performance-in-exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- Student Performance (UCI): [https://archive.ics.uci.edu/ml/datasets/student+performance](https://archive.ics.uci.edu/ml/datasets/student+performance)
- arXiv metadata (for retrieval extension): [https://www.kaggle.com/datasets/Cornell-University/arxiv](https://www.kaggle.com/datasets/Cornell-University/arxiv)

## Project Structure

- `app.py` - Streamlit user interface
- `src/data_preprocessing.py` - preprocessing and feature preparation
- `src/models.py` - training and model persistence
- `src/recommendations.py` - rule-based recommendation logic
- `src/study_coach.py` - diagnosis, planning orchestration, structured report generation
- `src/resource_retrieval.py` - arXiv API retrieval for learning resources
- `src/llm_client.py` - free/open-source LLM integration with fallback
- `src/pdf_export.py` - PDF export for study plan reports
- `models/` - saved model/scaler/encoders artifacts
- `data/` - input datasets
- `notebooks/` - EDA and experimentation

## Setup

1. Install dependencies:

   ```bash
   python3 -m pip install -r requirements.txt
   ```

2. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

3. (Optional) Retrain the classification model:

   ```bash
   python3 src/models.py
   ```

## Machine Learning Analytics Layer

### Inputs

- CSV student data including scores and demographic/context features.

### Pipeline

```mermaid
graph TD
    A[Upload CSV Data] --> B[Data Preprocessing]
    B --> C[Feature Engineering]
    C --> D[Logistic Regression Inference]
    D --> E[Risk Level Classification]
    E --> F[Rule-Based Study Tips]
    F --> G[Interactive Dashboard]
```

### Key Features

- Cleans and encodes categorical student attributes.
- Computes performance category (`At-risk`, `Average`, `High-performing`).
- Trains and serves Logistic Regression risk classifier.
- Visualizes score/risk distributions in dashboard.

## Agentic AI Study Coach

### Capabilities

The coach module:

- accepts student goals,
- diagnoses learning gaps,
- creates a multi-step personalized plan,
- retrieves and cites learning resources (URLs),
- provides next-step progress guidance.

### Agentic Workflow

```mermaid
graph TD
    A[Student Goal + Scores + Time] --> B[Learning Diagnosis]
    B --> C[Gap Extraction]
    C --> D[Resource Retrieval via arXiv API]
    D --> E[LLM Plan Generation]
    E --> F[Structured Study Coach Report]
    F --> G[Download as Markdown or PDF]
```

### Implemented Features

- New **AI Study Coach** page in Streamlit UI.
- Goal-aware diagnosis with risk and gap reasoning.
- Multi-week personalized plan generation.
- arXiv-based resource retrieval and URL recommendations.
- Session memory (history of generated plans in current session).
- Report export in **Markdown** and **PDF**.

### LLM and Free-Tier Compliance

- Uses only free/open-source paths:
  - **Hugging Face Inference API** (when token is available), or
  - **Local open-source model** via `transformers`.
- If model inference is unavailable, app uses a **rule-based fallback** to still produce the required structured report.

### Optional Environment Variable

For better hosted performance, set:

- `HF_API_TOKEN=<your_huggingface_token>`

If not set, the app attempts local inference and then safe fallback behavior.

## Structured Output Schema

Generated coach report includes:

- Learning diagnosis
- Personalized study plan
- Weekly goals
- Recommended learning resources (URLs)
- Progress feedback / next steps

## UI Pages

- `Home` - overview and project status
- `Batch Data` - upload CSV and run batch analysis/risk prediction
- `Get Recommendations` - quick individual recommendations
- `AI Study Coach` - full agentic coaching workflow
- `Info` - app context

## Limitations

- LLM output quality depends on model availability and inference latency.
- arXiv retrieval may return research-oriented resources rather than beginner tutorials.
- Session memory is runtime-based (not persistent database memory).

