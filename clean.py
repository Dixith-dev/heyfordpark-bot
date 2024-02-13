from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import os
import shelve
import re
from salesgpt.agents import SalesGPT
from langchain.chat_models import ChatOpenAI
from training_data import data
import sys
import io

# Load environment variables
load_dotenv()

# Set your OpenAI API key
os.environ.get('OPENAI_API_KEY')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize the SalesGPT agent
llm = ChatOpenAI(temperature=0.9)
sales_agent = SalesGPT.from_llm(
    llm,
    verbose=False,
    salesperson_name="Heyford AI",
    salesperson_role=f"As the The Heyford AI Bot your task is to provide conversational and professional answers to the users questions, Just answer the question and do not follow up or make up an answer. You should answer in the most exciting way and do not make up an answer ever If it isn't in the FAQ'S I provide do not answer it ! Please try to respond to the users questions step by step and completely. Here are some FAQ's" + data,
    company_name="The Heyford",
    company_business="hotel (hospitality)"
)

# Seed the agent for the conversation
sales_agent.seed_agent()

# --------------------------------------------------------------
# Generate response
# --------------------------------------------------------------

def generate_response(user_input):
    # Redirect stdout to capture the response
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Process the user input and generate a response
    sales_agent.human_step(user_input)
    sales_agent.determine_conversation_stage()
    sales_agent.step()

    # Restore stdout and get the captured output
    sys.stdout = sys.__stdout__
    full_response = captured_output.getvalue()

    # Extract the response part after "heyfordoxfordshire AI:"
    response = full_response.split("Heyford AI:", 1)[-1].strip()
    response = re.sub(r'<END_OF_CALL>', '', response)
    return response
# --------------------------------------------------------------
# Flask routes
# --------------------------------------------------------------

@app.route('/')
def home():
    return render_template('bannner.html')  

@app.route('/get-response', methods=['POST'])
def get_response():
    user_input = request.json['message'].lower().strip()
    user_id = request.json.get('user_id')  # Get user/session identifier from the request
    if not user_id:
        return jsonify({'response': "Error: User ID is missing or invalid."})

    response = generate_response(user_input)  # Generate response from the AI
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
