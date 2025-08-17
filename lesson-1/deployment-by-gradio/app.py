import gradio as gr
from fastai.vision.all import load_learner, PILImage

# Load FastAI model
learn = load_learner("model.pkl")

def predict(image):
    img = PILImage.create(image)
    pred, pred_idx, probs = learn.predict(img)
    return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}

with gr.Blocks(theme="soft") as demo:
    gr.Markdown("## 🐻 Bear Classifier")
    
    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            image_input = gr.Image(
                type="pil", 
                label="Upload an Image", 
                width=250,   
                height=250 

            )
            
            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                reset_btn = gr.Button("Reset", variant="secondary")
            
            output = gr.Label(num_top_classes=4, label="Prediction")
    
    submit_btn.click(fn=predict, inputs=image_input, outputs=output)
    reset_btn.click(fn=lambda: (None, None), inputs=None, outputs=[image_input, output])


demo.launch()
