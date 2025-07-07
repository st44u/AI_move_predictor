import streamlit as st
from predictor import UniversalStockPredictionAI  # Make sure this path is correct

# Title and intro
st.set_page_config(page_title="Stock Prediction AI", layout="wide")
st.title("📈 Universal Stock Prediction AI")
st.markdown("""
This AI understands natural language prompts and analyzes any stock or index.

**Examples**:  
- "Will NIFTY go up tomorrow?"  
- "What about Apple stock?"  
- "Predict Adani Power"  
- "Reliance outlook"

---
""")

# User input
user_prompt = st.text_input("🧠 Ask the AI:", placeholder="e.g. Tell me about Apple stock tomorrow")

if user_prompt:
    with st.spinner("🔍 Analyzing..."):
        ai = UniversalStockPredictionAI()
        response = ai.process_prompt(user_prompt)
        st.markdown("### AI Response")
        st.markdown(f"<pre style='white-space: pre-wrap;'>{response}</pre>", unsafe_allow_html=True)

# Optional: Run demo or test
with st.expander(" Run Demo / Test / Batch"):
    if st.button("Run Demo"):
        ai = UniversalStockPredictionAI()
        for prompt in [
            "NIFTY prediction", "Apple stock", "Adani Power", 
            "Tesla outlook", "Reliance stock", "Bank NIFTY tomorrow"
        ]:
            st.subheader(f"Prompt: {prompt}")
            response = ai.process_prompt(prompt)
            st.markdown(f"<pre style='white-space: pre-wrap;'>{response}</pre>", unsafe_allow_html=True)
            st.markdown("---")
