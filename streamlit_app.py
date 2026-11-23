import os
import requests
import streamlit as st

st.set_page_config(page_title="Ticket Classification Assistant", page_icon="🎫", layout="centered")
st.title("🎫 AI-Powered Ticket Classification Assistant")
st.caption("E-commerce customer support • Module 05 project")
st.write("Enter a fictional customer support ticket. The assistant suggests a category and a factual summary for human review.")

api_url = os.getenv("API_URL", "http://127.0.0.1:8000")
with st.form("ticket_form"):
    ticket = st.text_area("Customer ticket", placeholder="Example: I was charged twice for the same order...", height=150)
    submitted = st.form_submit_button("Classify ticket", type="primary")

if submitted:
    if not ticket.strip():
        st.error("Please enter a ticket before submitting.")
    else:
        try:
            response = requests.post(f"{api_url}/analyze", json={"ticket": ticket}, timeout=20)
            if response.status_code == 200:
                result = response.json()
                st.success("Classification completed. Please review before taking action.")
                st.metric("Predicted category", result["category"])
                st.write("**Factual summary**")
                st.info(result["summary"])
                st.caption(f"Mode: {result['mode']} • Prompt: {result['prompt_version']}")
            elif response.status_code == 422:
                st.error("The ticket input was invalid. Please correct it and try again.")
            else:
                st.error(f"The analysis service returned an error ({response.status_code}). Please try again later.")
        except requests.RequestException:
            st.error("Cannot reach the FastAPI backend. Start the API and verify API_URL.")
st.divider()
st.caption("Safety: This tool classifies and summarizes only. It does not issue refunds, cancel orders or change accounts.")
