import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000/classify"

st.set_page_config(
    page_title="Mental Health Agentic AI Platform",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Mental Health Agentic AI Platform")

st.write("AI-powered emotional intelligence and wellness insights")


user_input = st.text_area(
    "Enter your thoughts or feelings:",
    height=150
)


if st.button("Analyze Emotion"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:

        payload = {
            "text": user_input
        }

        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:

            result = response.json()

            st.success("Analysis Complete")

            st.subheader("Detected Emotion")
            st.write(result["emotion"])

            st.subheader("Confidence Score")
            st.write(round(result["confidence"] * 100, 2), "%")

            st.subheader("All Predictions")

            for pred in result["all_predictions"]:
                st.write(
                    f"{pred['label']} → {round(pred['score'] * 100, 2)}%"
                )

        else:
            st.error("API request failed.")