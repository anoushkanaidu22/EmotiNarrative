
import torch
from transformers import pipeline
import matplotlib.pyplot as plt

#analyze emotions in given text using pretrained transformer model
#[inputs] text (str): story prompt text to analyze
#[ouputs] dict of emotions and their scores
def analyze_emotions(text):

    emotion_classifier = pipeline('text-classification', 
                                 model='bhadresh-savani/distilbert-base-uncased-emotion', 
                                 return_all_scores=True)
    
    results = emotion_classifier(text)
    
    emotions = {item['label']: item['score'] for item in results[0]}
    
    sorted_emotions = dict(sorted(emotions.items(), key=lambda item: item[1], reverse=True))
    
    return sorted_emotions

#visualize emotions as bar chart
#[inputs] emotions (dict): dict of emotoins and their scores
def visualize_emotions(emotions):

    plt.figure(figsize=(10, 6))
    
    bars = plt.bar(emotions.keys(), emotions.values(), color='skyblue')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom', rotation=0)
    
    plt.xlabel('Emotions')
    plt.ylabel('Score')
    plt.title('Emotion Analysis of Story Prompt')
    plt.ylim(0, max(emotions.values()) + 0.1)  
    plt.tight_layout()
    plt.savefig('emotion_analysis.png') 
    plt.show()

#read multiline input from user
#[outputs] multi-line input as str
def read_multiline_input():

    # print("Enter your story prompt (press Enter twice to finish):")
    # lines = []
    # while True:
    #     line = input()
    #     if not line:
    #         break
    #     lines.append(line)
    # return '\n'.join(lines)
    text = "She stood frozen, her breath catching in her throat, each exhale forming fragile clouds in the cold morning air. Her outstretched hand trembled, fingers grasping at the empty space where moments ago, something—someone—had been. The sirens had faded, swallowed by the stillness of the street, leaving only the quiet creak of the wooden porch beneath her bare feet. The world felt too vast, too indifferent. The neighbors’ windows remained dark, their curtains drawn, as if the entire street had turned its back on her grief. The open door behind her yawned, a silent, hollow thing, whispering of absence. She could still hear the echoes of hurried voices, the rustling of stretcher wheels against the floor, the muffled plea of her own voice—“Please, wait. Just one more second.” But there were no more seconds. No more time. A gust of wind cut through the thin fabric of her nightshirt, but she barely felt it. Fear had rooted itself deep inside her chest, coiling like a living thing, tightening with every breath. It wasn’t just fear of what had happened. It was fear of what came next. Of walking back inside. Of facing the silence. Of realizing that the emptiness stretching out before her was not just in the house, not just in the street, but in her life itself."  
    return text

#main fn to analyze story prompt for emotions
def main():

    print("Story Prompt Emotion Analyzer")
    print("-----------------------------")
    
    story_prompt = read_multiline_input()
    
    if not story_prompt.strip():
        print("No input provided.")
        return
    
    try:

        emotions = analyze_emotions(story_prompt)
        
        print("\nEmotion Analysis Results:")
        for emotion, score in emotions.items():
            print(f"{emotion}: {score:.4f}")
        
        visualize_emotions(emotions)
        print("\nA visualization has been saved as 'emotion_analysis.png'")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Make sure you have installed the required packages:")
        print("pip install transformers torch matplotlib")

if __name__ == "__main__":
    main()
