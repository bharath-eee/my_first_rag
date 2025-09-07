import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict
from flask import Flask, request, jsonify, render_template
import webbrowser
import os
import logging
import re

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')

# List of forbidden keywords for jailbreaking or irrelevant queries
FORBIDDEN_KEYWORDS = [
    'jailbreak', 'hack', 'exploit', 'bypass', 'override', 'system prompt',
    'ignore instructions', 'root access', 'admin', 'malicious', 'code injection'
]

# Load the data
def load_data(file_path: str) -> List[Dict]:
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                logger.error(f"File is empty: {file_path}")
                raise ValueError(f"File is empty: {file_path}")
            data = json.loads(content)
        logger.debug(f"Loaded {len(data)} entries from {file_path}")
        return [entry['output'] for entry in data]
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise

# Convert district data to text for embedding
def data_to_text(data: Dict) -> str:
    return (
        f"District: {data['district']}, Latitude: {data['lat']}, Longitude: {data['lon']}, "
        f"Year: {data['year']}, Scenario: {data['scenario']}, Population: {data['population_estimate']}, "
        f"Rainfall: {data['avg_annual_rainfall_mm']} mm, Groundwater Level: {data['groundwater_level_m']} m"
    )

# Create embeddings and FAISS index
def create_vector_store(data: List[Dict]) -> tuple:
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        texts = [data_to_text(entry) for entry in data]
        embeddings = model.encode(texts, convert_to_numpy=True)
        logger.debug(f"Embeddings shape: {embeddings.shape}, type: {type(embeddings)}")
        
        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        logger.debug(f"FAISS index type: {type(faiss_index)}")
        faiss_index.add(embeddings)
        logger.debug(f"Added {faiss_index.ntotal} embeddings to FAISS index")
        
        return model, faiss_index, texts
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise

# Retrieve top-k relevant documents with similarity threshold
def retrieve_documents(query: str, model, faiss_index, texts: List[str], k: int = 3, threshold: float = 1.0) -> List[str]:
    try:
        logger.debug(f"Processing query: {query}")
        query_embedding = model.encode([query], convert_to_numpy=True)
        logger.debug(f"Query embedding shape: {query_embedding.shape}")
        logger.debug(f"FAISS index type before search: {type(faiss_index)}")
        if not hasattr(faiss_index, 'search'):
            logger.error(f"FAISS index does not have 'search' method. Type: {type(faiss_index)}")
            raise AttributeError("Invalid FAISS index")
        distances, indices = faiss_index.search(query_embedding, k)
        logger.debug(f"Retrieved indices: {indices}, distances: {distances}")
        valid_docs = [texts[i] for i, dist in zip(indices[0], distances[0]) if dist < threshold]
        return valid_docs if valid_docs else []
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        raise

# Generate response as HTML table and chart
def generate_response(query: str, retrieved_docs: List[str], data: List[Dict]) -> str:
    query_lower = query.lower().strip()
    # Handle generic queries
    if query_lower in ['hi', 'hello', 'hey']:
        return "Hello! I'm a chatbot that provides district-level groundwater and rainfall data. Try asking something like 'District names'."
    if query_lower == 'who are you':
        return "I'm a District Information Chatbot, designed to provide data on districts like rainfall, groundwater levels, and population. Ask me about a specific district or data point, e.g., 'District names'."
    
    # If no relevant documents, return a fallback message
    if not retrieved_docs:
        return f"Sorry, I couldn't find relevant data for '{query}'. Please try a query like 'Ariyalur rainfall in 2020'."

    # Parse retrieved documents into structured data
    table_rows = []
    chart_data = []
    is_ariyalur = 'ariyalur' in query_lower
    
    for doc in retrieved_docs:
        fields = {}
        for part in doc.split(", "):
            key, value = part.split(": ", 1)
            fields[key.lower().replace(" ", "_")] = value
        table_rows.append(fields)
        if is_ariyalur:
            chart_data.append({
                'label': f"{fields['year']} ({fields['scenario']})",
                'rainfall': float(fields['rainfall'].replace(" mm", ""))
            })
    
    # Create HTML table
    table_html = """
    <style>
        table { border-collapse: collapse; width: 100%; margin-top: 10px; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
        th { background-color: #007bff; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
    </style>
    <table>
        <tr>
            <th>District</th>
            <th>Year</th>
            <th>Scenario</th>
            <th>Rainfall (mm)</th>
            <th>Groundwater Level (m)</th>
            <th>Population</th>
        </tr>
    """
    for row in table_rows:
        table_html += f"""
        <tr>
            <td>{row['district']}</td>
            <td>{row['year']}</td>
            <td>{row['scenario']}</td>
            <td>{row['rainfall']}</td>
            <td>{row['groundwater_level']}</td>
            <td>{row['population']}</td>
        </tr>
        """
    table_html += "</table>"
    
    # Create Chart.js chart for Ariyalur rainfall
    chart_html = ""
    if is_ariyalur and chart_data:
        chart_config = {
            "type": "bar",
            "data": {
                "labels": [item['label'] for item in chart_data],
                "datasets": [{
                    "label": "Rainfall (mm)",
                    "data": [item['rainfall'] for item in chart_data],
                    "backgroundColor": ["#007bff", "#28a745", "#dc3545"],
                    "borderColor": ["#0056b3", "#218838", "#c82333"],
                    "borderWidth": 1
                }]
            },
            "options": {
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "title": {"display": True, "text": "Rainfall (mm)"}
                    },
                    "x": {
                        "title": {"display": True, "text": "Year (Scenario)"}
                    }
                },
                "plugins": {
                    "legend": {"display": True},
                    "title": {"display": True, "text": "Ariyalur Rainfall Comparison"}
                }
            }
        }
        chart_html = f"""
        <canvas id="rainfallChart" width="400" height="200"></canvas>
        <script>
            document.addEventListener('DOMContentLoaded', () => {{
                const ctx = document.getElementById('rainfallChart').getContext('2d');
                new Chart(ctx, {json.dumps(chart_config)});
            }});
        </script>
        """
    
    response = f"Based on your query '{query}', here is the relevant information:<br>{table_html}"
    if chart_html:
        response += f"<br>{chart_html}"
    
    return response

# Load data and initialize vector store
DATA_PATH = r"C:\Users\Bharath\Desktop\actual final\data.json"
try:
    data = load_data(DATA_PATH)
    model, faiss_index, texts = create_vector_store(data)
except Exception as e:
    logger.error(f"Failed to initialize chatbot: {str(e)}")
    raise

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        query = request.json.get('query')
        if not query:
            return jsonify({'response': 'Please provide a query.'}), 400
        
        # Check for forbidden keywords
        query_lower = query.lower().strip()
        if any(keyword in query_lower for keyword in FORBIDDEN_KEYWORDS):
            logger.warning(f"Forbidden query detected: {query}")
            return jsonify({'response': 'Wrong question you asked.'}), 400
        
        retrieved_docs = retrieve_documents(query, model, faiss_index, texts, k=3, threshold=1.0)
        response = generate_response(query, retrieved_docs, data)
        logger.debug(f"Response generated for query '{query}': {response[:100]}...")
        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Error in /chat endpoint: {str(e)}")
        return jsonify({'response': f'Error: {str(e)}'}), 500

# HTML template for chat interface
def create_html_template():
    html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>District Information Chatbot</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f2f5;
        }
        .chat-container {
            max-width: 600px;
            margin: auto;
            border: 1px solid #ccc;
            border-radius: 10px;
            background-color: white;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .chat-box {
            height: 400px;
            overflow-y: auto;
            padding: 15px;
            border-bottom: 1px solid #ccc;
        }
        .message {
            margin: 10px;
            padding: 10px;
            border-radius: 8px;
            max-width: 80%;
        }
        .user-message {
            background-color: #007bff;
            color: white;
            margin-left: auto;
            text-align: right;
        }
        .bot-message {
            background-color: #e9ecef;
            color: black;
            margin-right: auto;
        }
        .input-container {
            display: flex;
            padding: 10px;
        }
        input[type="text"] {
            flex-grow: 1;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
            font-size: 16px;
        }
        button {
            padding: 10px 20px;
            margin-left: 10px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-box" id="chatBox">
            <div class="message bot-message">Welcome to the District Information Chatbot! Ask about district data .</div>
        </div>
        <div class="input-container">
            <input type="text" id="userInput" placeholder="Type your query..." />
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>
    <script>
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const query = input.value.trim();
            if (!query) return;

            // Display user message
            const chatBox = document.getElementById('chatBox');
            const userMessage = document.createElement('div');
            userMessage.className = 'message user-message';
            userMessage.textContent = query;
            chatBox.appendChild(userMessage);

            // Clear input
            input.value = '';

            // Fetch response from server
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query })
                });
                const data = await response.json();
                const botMessage = document.createElement('div');
                botMessage.className = 'message bot-message';
                botMessage.innerHTML = data.response;
                chatBox.appendChild(botMessage);
                chatBox.scrollTop = chatBox.scrollHeight;
            } catch (error) {
                const botMessage = document.createElement('div');
                botMessage.className = 'message bot-message';
                botMessage.textContent = 'Error: Could not fetch response.';
                chatBox.appendChild(botMessage);
            }
        }

        // Allow pressing Enter to send message
        document.getElementById('userInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
    </script>
</body>
</html>
'''
    try:
        os.makedirs('templates', exist_ok=True)
        with open('templates/index.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.debug("Created HTML template")
    except Exception as e:
        logger.error(f"Error creating HTML template: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        logger.debug(f"Starting Flask app at {os.getcwd()}")
        create_html_template()
        webbrowser.open('http://127.0.0.1:5000')
        app.run(debug=False, port=5000)
    except Exception as e:
        logger.error(f"Failed to start Flask app: {str(e)}")
        raise