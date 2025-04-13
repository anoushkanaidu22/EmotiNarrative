import os
import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import argparse

#analyzes emotions in faces found in given image
#[inputs] image_path (str): path to image file
#[output] list of dictionaries with emotion analysis for each face
def analyze_image_emotions(image_path):
    try:
        if not os.path.isfile(image_path):
            print(f"Error: file '{image_path}' not found.")
            return None
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: can't read image '{image_path}'.")
            return None
            
        #convert BGR to RGB since DeepFace works with RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        analysis_results = DeepFace.analyze(
            img_path=rgb_img,
            actions=['emotion'],
            enforce_detection=False, 
            detector_backend='opencv'
        )
        
        if isinstance(analysis_results, dict):
            analysis_results = [analysis_results]
            
        return analysis_results
        
    except Exception as e:
        print(f"error occurred: {e}")
        return None

#make visualization of detected faces and their emotions
#[inputs] image_path (str): path to image file, analysis_results (list): list of analysis results from DeepFace
#[outputs] none, saves & displays visualization
def visualize_emotions(image_path, analysis_results):

    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    for i, result in enumerate(analysis_results):
        #face region
        region = result.get('region', None)
        emotions = result.get('emotion', {})
        
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        dominant_emotion = sorted_emotions[0][0]
        
        if region:
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
            #rect around face
            draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=2)
            
            draw.text((x, y - 30), f"Face {i+1}: {dominant_emotion}", fill="red", font=font)
            
            y_offset = y + h + 10
            for emotion, score in sorted_emotions[:3]:  # Top 3 emotions
                draw.text((x, y_offset), f"{emotion}: {score:.1f}%", fill="blue", font=font)
                y_offset += 25
    
    #save
    output_path = "emotion_analysis_result.jpg"
    img.save(output_path)
    print(f"Visualization saved as '{output_path}'")
    
    #display
    plt.figure(figsize=(12, 10))
    plt.imshow(np.array(img))
    plt.axis('off')
    plt.tight_layout()
    plt.show()

#bar charts of emotions for each detected face
#[inputs] anlaysis_results (list): list of analysis results from DeepFace
#[outputs] none, displays charts
def visualize_emotion_chart(analysis_results):
    n_faces = len(analysis_results)
    
    if n_faces == 0:
        print("no faces detected for visualization.")
        return
        
    fig, axes = plt.subplots(1, n_faces, figsize=(6*n_faces, 6))
    if n_faces == 1:
        axes = [axes] 
        
    for i, (ax, result) in enumerate(zip(axes, analysis_results)):
        emotions = result.get('emotion', {})
        
        emotions = dict(sorted(emotions.items(), key=lambda x: x[1], reverse=True))
        
        bars = ax.bar(emotions.keys(), emotions.values(), color='skyblue')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', rotation=45)
        
        ax.set_title(f'Face {i+1} Emotions')
        ax.set_ylim(0, 105) 
        ax.set_ylabel('Confidence (%)')
        ax.set_xticklabels(emotions.keys(), rotation=45)
        
    plt.tight_layout()
    plt.savefig('emotion_charts.png')
    print("Emotion charts saved as 'emotion_charts.png'")
    plt.show()

#main fn to analyze emotions in image
def main():

    parser = argparse.ArgumentParser(description='Analyze emotions in an image.')
    parser.add_argument('image_path', type=str, help='Path to the image file')

    args = parser.parse_args()
    
    print(f"analyzing emotions in image: {args.image_path}")
    
    analysis_results = analyze_image_emotions(args.image_path)
    
    if analysis_results:
        print(f"\nFound {len(analysis_results)} face(s) in the image.")
        
        for i, result in enumerate(analysis_results):
            emotions = result.get('emotion', {})
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            
            print(f"\nFace {i+1}:")
            print(f"Dominant emotion: {dominant_emotion[0]} ({dominant_emotion[1]:.2f}%)")
            print("All emotions:")
            for emotion, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                print(f"  {emotion}: {score:.2f}%")
        
        visualize_emotions(args.image_path, analysis_results)
        visualize_emotion_chart(analysis_results)
    else:
        print("No analysis results to display.")

if __name__ == "__main__":
    main()
