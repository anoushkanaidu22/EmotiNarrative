import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import text_to_emotions

class TextEmotionModel:
    def __init__(self):
        self.version = "text_v1.0"
        self.correction_layer = None
        self.emotions = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    
    def analyze(self, text):
        #Analyze emotions in text and apply correction if available
        #get base model predictions
        try:
            base_predictions = text_to_emotions.analyze_emotions(text)
            
            if self.correction_layer is not None:
                #convert to feature vector
                features = np.array([[base_predictions.get(emotion, 0) for emotion in self.emotions]])
                
                #apply correction
                corrected = self.correction_layer.predict(features)[0]
                
                #back to dict
                corrected_predictions = {}
                for i, emotion in enumerate(self.emotions):
                    corrected_predictions[emotion] = max(0, min(1, corrected[i]))
                
                total = sum(corrected_predictions.values())
                if total > 0:
                    for emotion in corrected_predictions:
                        corrected_predictions[emotion] /= total
                
                return corrected_predictions
            else:
                return base_predictions
        except Exception as e:
            print(f"Error analyzing text: {e}")
            #return defualt vals
            return {emotion: 0.0 for emotion in self.emotions}
    
    def set_correction_layer(self, correction_layer):
        self.correction_layer = correction_layer
    
    def update_version(self, new_version):
        self.version = new_version