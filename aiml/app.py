from flask import Flask

# Create the Flask app
app = Flask(__name__)

# Define a route
@app.route('/')
def home():
    return "ready"

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
