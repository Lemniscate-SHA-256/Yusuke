from flask import Flask, render_template, request
from generator.art_generator import generate_art

app = Flask(__name__)

class MangaUI:
    def run(self):
        app.run(debug=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    title = request.form['title']
    story = generate_story(title)
    art = generate_art(story)
    return render_template('result.html', story=story, art=art)
