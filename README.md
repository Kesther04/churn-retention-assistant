# churn-retention-assistant

An AI assistant that predicts which customers are likely to churn, explains why, and recommends how to keep them.

## The Problem

SaaS companies lose revenue to customers they could have kept. Most teams only notice churn after it happens, and even when they spot a risky customer, they do not know what to do about it.

This project closes both gaps. A machine learning model scores each customer's churn risk, and an AI agent turns that score into a clear explanation and a recommended action, grounded in a set of retention playbooks.

## What It Does

1. Takes a customer's details through a simple web interface.
2. Predicts the probability that the customer will churn.
3. Shows the top factors driving that risk.
4. Lets you ask the assistant questions like "Why is this customer at risk and what should I do?"
5. Returns a grounded answer and shows which playbook it came from.

## How It Works

The project has two connected parts.

**1. The prediction model (Machine Learning with Python)**

* Cleans and prepares the Telco Customer Churn dataset.
* Trains and compares logistic regression and random forest classifiers.
* Evaluates them with accuracy, precision, recall, and a confusion matrix.
* Saves the best model and exposes it as a function that returns a churn probability and the top risk factors.

**2. The retention assistant (AI Agents with RAG and LangChain)**

* A set of retention playbook documents is split into chunks, embedded, and stored in a vector database.
* A LangChain agent has two tools: a prediction tool that calls the saved model, and a retrieval tool that searches the playbooks.
* The agent predicts risk first, then retrieves relevant advice, then combines both into one answer.
* If the playbooks do not cover a question, the agent says it does not know instead of guessing.

## Tech Stack

* Python
* pandas, scikit learn, matplotlib, joblib
* LangChain
* Chroma (vector database)
* Streamlit

## Dataset

The Telco Customer Churn dataset from Kaggle. It contains customer records with details such as contract type, tenure, monthly charges, and services used, along with whether the customer churned.

## Project Structure

* data: raw and cleaned datasets
* notebooks: exploration and model training
* src: prediction function, RAG pipeline, and agent
* playbooks: retention playbook documents
* app: Streamlit interface
* requirements.txt: project dependencies
* .env.example: template for environment variables
* README.md: this file

## Getting Started

**1. Clone the repository**

```
git clone https://github.com/kesther04/churn-retention-assistant.git
cd churn-retention-assistant
```

**2. Create and activate a virtual environment**

```
python -m venv venv
source venv/bin/activate
```

On Windows, activate it with `venv\Scripts\activate` instead.

**3. Install the dependencies**

Install the packages listed in `requirements.txt` using pip.

**4. Add your model access**

Copy `.env.example` to `.env` and add your API key there. Never commit your `.env` file.

**5. Run the app**

```
streamlit run app/main.py
```

## Example Questions to Try

1. "Why is this customer at risk of churning?"
2. "What should I do to keep a price sensitive customer?"
3. "Which playbook applies to a customer in their first three months?"

## Limitations

1. The model is trained on one public telecom dataset, so it may not transfer directly to a different SaaS product.
2. The playbooks are a small, hand written set and do not cover every retention scenario.
3. The agent can occasionally pick the wrong tool or give a generic answer.
4. Churn probability is an estimate, not a certainty.

## Future Improvements

1. Add a regression model to estimate revenue at risk for each customer.
2. Add a third tool that drafts a personalized outreach message.
3. Support uploading a full customer list for batch scoring.
4. Retrain the model on a company's own data.

## About the Author

Built by Kesther Ogbu, a full stack developer who builds B2B SaaS products.

Portfolio: https://kesther.vercel.app

GitHub: kesther04