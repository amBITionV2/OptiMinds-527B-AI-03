from flask import Flask

# Create the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "ready"

if __name__ == '__main__':
    app.run(debug=True)
