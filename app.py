import os
import io
import json
import base64
import threading
from datetime import datetime
from flask import Flask, render_template, request, send_file
from PIL import Image
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for Flask
import matplotlib.pyplot as plt

from data.segmentation_model import BrainTumorSegmentationModel
from agents.xai_agent import XAIAgent
from utils.image_processing import load_and_preprocess_image, overlay_cam_on_image

# Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "temp_uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Model config
model_path = "models/best_model.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"


def image_to_base64(fig):
    """Convert Matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded_file = request.files.get("mri_image")
        if not uploaded_file:
            return render_template("index.html", error="No file uploaded")

        temp_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_file.filename)
        uploaded_file.save(temp_path)

        try:
            # --- Preprocess ---
            image_tensor, original_img = load_and_preprocess_image(temp_path, device)

            # --- Load segmentation model ---
            seg_model = BrainTumorSegmentationModel(model_path=model_path, device=device)
            prediction_mask = seg_model.predict(image_tensor).cpu().numpy().squeeze()

            # --- Tumor metrics ---
            seg_results = {
                "prediction_mask": prediction_mask,
                "metrics": {
                    "area_pixels": int((prediction_mask > 0.5).sum()),
                    "area_percentage": round(100 * (prediction_mask > 0.5).mean(), 2),
                    "centroid": {
                        "x": int(prediction_mask.shape[1] / 2),
                        "y": int(prediction_mask.shape[0] / 2)
                    },
                    "bounding_box": {
                        "x_min": 0,
                        "y_min": 0,
                        "x_max": prediction_mask.shape[1],
                        "y_max": prediction_mask.shape[0]
                    },
                    "contour_count": 1
                }
            }

            # --- Run XAI Agent with timeout ---
            xai_results = {}

            def run_xai_agent():
                nonlocal xai_results
                agent = XAIAgent(seg_model=seg_model, image_tensor=image_tensor, prediction_mask=prediction_mask)
                xai_results = agent.run()

            xai_thread = threading.Thread(target=run_xai_agent)
            xai_thread.start()
            xai_thread.join(timeout=300)  # 60-second timeout

            if not xai_results:
                xai_results = {
                    "gradcam_heatmap": prediction_mask,  # fallback
                    "explanation": "LLM explanation timed out or unavailable"
                }

            gradcam = xai_results.get("gradcam_heatmap", None)
            explanation = xai_results.get("explanation", "No explanation available")

            # --- Mask Plot ---
            fig_mask, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(prediction_mask, cmap='hot')
            ax.axis('off')
            mask_base64 = image_to_base64(fig_mask)
            plt.close(fig_mask)

            # --- Grad-CAM Plot ---
            fig_gradcam, ax = plt.subplots(figsize=(6, 6))
            if gradcam is not None:
                if isinstance(gradcam, torch.Tensor):
                    gradcam = gradcam.detach().cpu().numpy()
                ax.imshow(gradcam.squeeze(), cmap='jet')
            ax.axis('off')
            gradcam_base64 = image_to_base64(fig_gradcam)
            plt.close(fig_gradcam)

            # --- Overlay Visualization ---
            overlay_img = overlay_cam_on_image(
                original_img,
                gradcam if gradcam is not None else np.zeros_like(image_tensor[0].cpu().numpy())
            )
            overlay_img_pil = Image.fromarray((overlay_img * 255).astype(np.uint8))
            buf = io.BytesIO()
            overlay_img_pil.save(buf, format="PNG")
            overlay_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            # --- JSON Output ---
            output_data = {
                "timestamp": datetime.now().isoformat(),
                "image_filename": uploaded_file.filename,
                "tumor_analysis": seg_results["metrics"],
                "xai_explanation": {
                    "textual_justification": explanation
                }
            }
            output_filename = f"outputs/analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_filename, 'w') as f:
                json.dump(output_data, f, indent=2)

            return render_template(
                "results.html",
                mask_image=mask_base64,
                gradcam_image=gradcam_base64,
                overlay_image=overlay_base64,
                tumor_metrics=seg_results["metrics"],
                explanation=explanation,
                json_file=output_filename
            )

        except Exception as e:
            import traceback
            return render_template("index.html", error=str(e), traceback=traceback.format_exc())

    return render_template("index.html")


@app.route("/download/<path:filename>")
def download_file(filename):
    return send_file(filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
