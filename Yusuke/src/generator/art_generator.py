import tensorflow as tf
from utils.image_processing import process_image

def generate_art(story):
    # Load model, generate image based on story
    model = tf.keras.models.load_model('path/to/model')
    art = model.predict(story)
    processed_art = process_image(art)
    return processed_art

