from flask import Flask, render_template, request
from app.main import main as generate_manga

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    generate_manga(prompt)
    return render_template('result.html', images=get_generated_images())

def get_generated_images():
    # Return list of generated images
    pass

if __name__ == '__main__':
    app.run(debug=True)
