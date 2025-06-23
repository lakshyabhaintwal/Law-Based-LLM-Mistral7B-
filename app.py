from flask import Flask, request, jsonify, render_template
from rag_app import rag_chat  # your function from earlier

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_query = request.form.get("query")
    if not user_query:
        return "No input provided.", 400
    response = rag_chat(user_query)
    return render_template("index.html", query=user_query, response=response)

if __name__ == "__main__":
    app.run(debug=True)
