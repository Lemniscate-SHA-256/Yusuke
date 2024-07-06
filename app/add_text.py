from PIL import Image, ImageDraw, ImageFont

def add_text_to_image(image_path, text, output_path):
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    
    text_w, text_h = draw.textsize(text, font=font)
    width, height = image.size
    text_position = ((width - text_w) // 2, height - text_h - 10)

    bubble_radius = 10
    bubble_margin = 10
    bubble_box = [text_position[0] - bubble_margin, text_position[1] - bubble_margin, 
                  text_position[0] + text_w + bubble_margin, text_position[1] + text_h + bubble_margin]

    draw.rounded_rectangle(bubble_box, radius=bubble_radius, fill='white')
    draw.text(text_position, text, font=font, fill='black')
    
    image.save(output_path)
