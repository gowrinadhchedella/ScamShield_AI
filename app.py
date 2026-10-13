import os
import streamlit as st
from google import genai
from typing import TypedDict
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="ScamShield AI", page_icon="🔐", layout="centered")

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

scam_knowledge = [
    """Bank impersonation scam: Fraudsters pretend to be a bank and claim an account is blocked,
    suspended, or compromised. Common signals include urgency, unofficial links, requests for
    OTPs, PINs, passwords, card details, or instructions to move money. Verify through official channels.""",

    """OTP and credential theft: Messages requesting an OTP, password, PIN, CVV, recovery code,
    or login credentials are high risk. Never share authentication codes in response to an unsolicited message.""",

    """Prize and lottery scam: The recipient is told they won a lottery, giveaway, reward, or
    expensive prize they did not knowingly enter. The scam may demand a fee, bank information,
    or identity documents before the supposed prize is released.""",

    """Job and recruitment scam: Warning signs include unusually easy jobs, very high salaries,
    upfront registration or training fees, requests to buy equipment, and requests for sensitive financial information.""",

    """Investment and crypto scam: Messages promise guaranteed returns, very high profits,
    secret trading methods, or risk-free cryptocurrency investments and pressure victims to act quickly.""",

    """Delivery and parcel scam: A message claims a package cannot be delivered because of an
    unpaid fee, incomplete address, or customs charge and may link to a fake courier website.""",

    """Tech support scam: The sender claims a device or account has a virus or security problem
    and asks the victim to call, install remote-access software, or provide credentials.""",

    """Government or authority impersonation: Scammers impersonate police, tax departments,
    courts, or other authorities and threaten arrest, fines, account freezing, or legal action
    unless money is paid immediately.""",

    """Gift card and unusual payment scam: Requests for payment through gift cards,
    cryptocurrency, wire transfers, or other hard-to-reverse methods are suspicious,
    especially when combined with secrecy or urgency.""",

    """Phishing link scam: Fraudulent messages often contain shortened, misspelled, look-alike,
    or unfamiliar links that lead to fake login pages. Navigate to the official site independently.""",

    """Family or emergency impersonation scam: A sender pretends to be a relative, friend,
    manager, or colleague in an emergency and urgently requests money or confidential information.
    Verify identity through a known trusted channel.""",

    """Low-risk characteristics: Expected context, no request for credentials, no suspicious
    links, no pressure, and no unusual payment request can indicate lower risk, but do not prove safety."""
]

documents = [Document(page_content=text) for text in scam_knowledge]

class TfidfEmbeddings(Embeddings):
    def __init__(self, texts):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.vectorizer.fit(texts)

    def embed_documents(self, texts):
        return self.vectorizer.transform(texts).toarray().tolist()

    def embed_query(self, text):
        return self.vectorizer.transform([text]).toarray()[0].tolist()

@st.cache_resource
def create_vector_database():
    texts = [doc.page_content for doc in documents]
    embeddings = TfidfEmbeddings(texts)
    return FAISS.from_documents(documents, embeddings)

vector_db = create_vector_database()

class ScamState(TypedDict):
    message: str
    context: str
    signals: str
    answer: str

def retrieve_patterns(state: ScamState):
    results = vector_db.similarity_search(state["message"], k=3)
    context = "\n\n".join(doc.page_content for doc in results)
    return {"context": context}

def extract_signals(state: ScamState):
    text = state["message"].lower()
    found = []

    rules = {
        "Urgency / pressure": ["urgent", "immediately", "now", "today", "act fast", "within 24"],
        "OTP / password request": ["otp", "password", "pin", "cvv", "verification code", "login code"],
        "Suspicious link": ["http://", "https://", "bit.ly", "tinyurl", "click here", "click the link"],
        "Money / payment request": ["pay", "payment", "fee", "transfer", "send money", "gift card", "crypto", "bitcoin"],
        "Threat / account warning": ["blocked", "suspended", "arrest", "legal action", "fine", "account closed"],
        "Prize / unrealistic reward": ["won", "winner", "lottery", "prize", "reward", "guaranteed return"]
    }

    for label, keywords in rules.items():
        if any(keyword in text for keyword in keywords):
            found.append(label)

    signals = ", ".join(found) if found else "No obvious rule-based warning signal detected"
    return {"signals": signals}

def analyze_risk(state: ScamState):
    prompt = f"""
You are ScamShield AI, a fraud-message analysis assistant.

Analyze the message using the retrieved scam-pattern knowledge and detected warning signals.

Return:
### Risk Level
Choose exactly one: LOW, MEDIUM, or HIGH

### Why
Explain the main reasons.

### Suspicious Signals
List the warning signs.

### Safe Next Steps
Give defensive actions. Never direct the user to contact a phone number, email, or link contained
inside the suspicious message. Recommend independent verification using official channels.

### Important Note
State that this is an AI screening result and cannot guarantee whether a message is genuine or fraudulent.

Do not claim certainty when evidence is insufficient.
Do not invent facts about the sender, website, phone number, or organization.

Retrieved Scam Knowledge:
{state["context"]}

Rule-based Signals:
{state["signals"]}

Message:
{state["message"]}
"""

    response = client.interactions.create(
        model="gemini-3.8-flash",
        input=prompt
    )
    return {"answer": response.output_text}

workflow = StateGraph(ScamState)
workflow.add_node("retrieve_patterns", retrieve_patterns)
workflow.add_node("extract_signals", extract_signals)
workflow.add_node("risk_analyzer", analyze_risk)
workflow.add_edge(START, "retrieve_patterns")
workflow.add_edge("retrieve_patterns", "extract_signals")
workflow.add_edge("extract_signals", "risk_analyzer")
workflow.add_edge("risk_analyzer", END)

scam_agent = workflow.compile()

st.title("🔐 ScamShield AI")
st.subheader("AI Scam & Fraud Message Analyzer")
st.write("Powered by Gemini, LangGraph, RAG, FAISS and TF-IDF Vectorization")
st.divider()

message = st.text_area(
    "Paste an SMS, email, or chat message:",
    placeholder="Example: URGENT! Your account is suspended. Click this link and enter your OTP...",
    height=180
)

if st.button("🔍 Analyze Message", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("Please paste a message first.")
    else:
        with st.spinner("ScamShield is analyzing the message..."):
            try:
                result = scam_agent.invoke({
                    "message": message,
                    "context": "",
                    "signals": "",
                    "answer": ""
                })

                st.success("Analysis complete")
                st.markdown(result["answer"])

                with st.expander("🚩 Detected Signals"):
                    st.write(result["signals"])

                with st.expander("🔎 RAG Knowledge Retrieved"):
                    st.write(result["context"])

            except Exception as e:
                st.error(f"Error: {str(e)}")

st.divider()
st.caption("ScamShield AI • Educational scam screening assistant • Verify important messages independently")
