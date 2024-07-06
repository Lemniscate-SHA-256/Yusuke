from .generate_story import generate_story
from .generate_panel import generate_panel
from .apply_style import apply_style
from .add_text import add_text_to_image

def main():
    story_prompt = "Once upon a time in a faraway land, there was a brave warrior named Hiro."
    story = generate_story(story_prompt)
    story_lines = story.split('. ')
    
    for i, line in enumerate(story_lines):
        panel_path = generate_panel()
        styled_panel_path = apply_style(panel_path)
        final_caption = f"{line}"
        output_path = f'final_image_with_text_{i}.png'
        add_text_to_image(styled_panel_path, final_caption, output_path)

if __name__ == "__main__":
    main()
