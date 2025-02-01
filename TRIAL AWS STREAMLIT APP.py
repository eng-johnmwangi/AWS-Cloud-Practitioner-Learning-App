import os
import streamlit as st
from openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List, Dict, Any
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client with OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

class OpenRouterLLM:
    def __init__(self, client: OpenAI):
        self.client = client

    def generate(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://aws-practitioner-app.com",
                "X-Title": "AWS Practitioner Learning App"
            },
            model="deepseek/deepseek-r1",
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content

llm = OpenRouterLLM(client=client)

# Initialize session state
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'current_question' not in st.session_state:
    st.session_state.current_question = None
if 'difficulty' not in st.session_state:
    st.session_state.difficulty = "Medium"

# Question Generation Prompt
question_prompt_template = """
Generate an AWS Cloud Practitioner exam question about {topic} with {difficulty} difficulty.
Include 4 multiple-choice options and provide the correct answer with explanation.

Format exactly like this:
Question: [question text]
Options:
A) [option A]
B) [option B]
C) [option C]
D) [option D]
Correct Answer: [letter]
Explanation: [detailed explanation]
"""

# Streamlit App Interface
st.title("AWS Cloud Practitioner Learning App")
st.write(f"**Score:** {st.session_state.score}")

# Difficulty Selector
st.session_state.difficulty = st.selectbox(
    "Select Difficulty",
    ["Easy", "Medium", "Hard"],
    index=1
)

# Topic Input
topic = st.text_input("Enter a topic (e.g., S3, EC2, IAM):")

if st.button("Generate Question"):
    if topic:
        with st.spinner("Generating question..."):
            try:
                prompt = question_prompt_template.format(
                    topic=topic,
                    difficulty=st.session_state.difficulty
                )
                response = llm.generate(prompt)
                
                # Parse the response
                question_data = {
                    "question": response.split("Question: ")[1].split("\nOptions:")[0].strip(),
                    "options": {},
                    "correct": response.split("Correct Answer: ")[1].split("\n")[0].strip(),
                    "explanation": response.split("Explanation: ")[1].strip()
                }
                
                # Parse options
                options_section = response.split("Options:")[1].split("Correct Answer:")[0]
                for line in options_section.strip().split('\n'):
                    if ')' in line:
                        key, value = line.split(')', 1)
                        question_data["options"][key.strip()] = value.strip()
                
                st.session_state.current_question = question_data
                
            except Exception as e:
                st.error(f"Error generating question: {str(e)}")
    else:
        st.warning("Please enter a topic")

# Display question if available
if st.session_state.current_question:
    question = st.session_state.current_question
    st.subheader("Generated Question")
    st.write(question["question"])
    
    # Display options
    options = list(question["options"].items())
    selected_option = st.radio(
        "Select your answer:",
        options=[f"{k}) {v}" for k, v in options],
        key="answer_selection"
    )
    
    if st.button("Submit Answer"):
        selected_letter = selected_option[0]
        if selected_letter == question["correct"]:
            st.success("Correct! 🎉")
            st.session_state.score += 1
        else:
            st.error(f"Incorrect ❌. The correct answer is {question['correct']})")
        
        # Show explanation
        st.subheader("Explanation")
        st.write(question["explanation"])
        
        # Clear for next question
        st.session_state.current_question = None
        st.rerun()