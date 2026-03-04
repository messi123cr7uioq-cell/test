#late_fusion

import tensorflow as tf      #type: ignore
import numpy as np           #type: ignore
import gradio as gr          #type: ignore
from tensorflow.keras.models import Model               #type: ignore
from tensorflow.keras.layers import Input, Lambda       #type: ignore
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification      #type: ignore

#Set up and loading models
TEXT_MODEL_PATH = r"C:\Users\user\Documents\E\report\trained_clinicalbert"
IMAGE_MODEL_PATH = "breast_cancer_finetuned.keras"

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model_text = TFAutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_PATH)
model_img = tf.keras.models.load_model(IMAGE_MODEL_PATH, compile=False)

LABELS = ['BI-RADS-1', 'BI-RADS-2', 'BI-RADS-3', 'BI-RADS-4', 'BI-RADS-5']

#define unified multimodal model 
def create_fused_model():
    img_input = Input(shape=(224, 224, 3), name="image_input")
    img_output = model_img(img_input)
    text_input = Input(shape=(5,), name="text_probabilities_input")
    
    fused = (0.40 * text_input) + (0.60 * img_output)
    unified_model = Model(inputs=[img_input, text_input], outputs=fused)
    return unified_model

# Save the unified structure
fused_system = create_fused_model()
fused_system.save("unified_birads_fusion_model.keras")
print("Unified Multimodal Model Saved!")

#UPDATED PREDICTION FUNCTION ---
def predict_birads(text_input, image_input):
   
    tokens = tokenizer(text_input, return_tensors="tf", truncation=True, padding=True, max_length=256)
    t_logits = model_text(**tokens).logits
    t_probs = tf.nn.softmax(t_logits, axis=-1).numpy()[0]

    img = tf.image.resize(image_input, (224, 224))
    img = tf.cast(img, tf.float32) / 255.0
    img = np.expand_dims(img, axis=0)
 
    fused_probs = fused_system.predict([img, np.expand_dims(t_probs, axis=0)])[0] 
    final_label = LABELS[np.argmax(fused_probs)]
    confidences = {LABELS[i]: float(fused_probs[i]) for i in range(len(LABELS))}
    return final_label, confidences

#improved UI (gradion blocks) 
with gr.Blocks(title="Advanced BI-RADS Multimodal AI") as demo:
    gr.Markdown("#Multimodal BI-RADS Diagnostic System")
    gr.Markdown("Combine clinical text reports with mammogram images for a high-accuracy prediction.")
    
    with gr.Row():
        with gr.Column():
            text_in = gr.Textbox(label="Clinical Report Analysis", lines=5, placeholder="Enter radiologist notes here...")
            img_in = gr.Image(label="Upload Mammogram/Ultrasound", type="numpy")
            submit_btn = gr.Button("Generate Fused Prediction", variant="primary")
            
        with gr.Column():
            label_out = gr.Textbox(label="Final BI-RADS Category")
            confidence_out = gr.Label(label="Fusion Confidence Breakdown")

    submit_btn.click(fn=predict_birads, inputs=[text_in, img_in], outputs=[label_out, confidence_out])

if __name__ == "__main__":
    demo.launch()