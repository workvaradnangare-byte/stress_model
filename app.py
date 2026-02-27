from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load model once when app starts
model = tf.keras.models.load_model("stress_model.h5")

IMG_SIZE = 96


def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))
    return img


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":

        # Check if file exists in request
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["file"]

        # Check if file selected
        if file.filename == "":
            return render_template("index.html", error="No file selected")

        # Ensure static folder exists (IMPORTANT FOR RAILWAY)
        upload_folder = os.path.join(os.getcwd(), "static")
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        filepath = os.path.join(upload_folder, file.filename)
        file.save(filepath)

        img = preprocess_image(filepath)

        if img is None:
            return render_template("index.html", error="Invalid image file")

        prediction = model.predict(img)[0][0]

        if prediction > 0.3:
            result = "STRESSED"
            confidence = prediction * 100
        else:
            result = "NON-STRESSED"
            confidence = (1 - prediction) * 100

        return render_template(
            "index.html",
            result=result,
            confidence=round(confidence, 2),
            image=f"static/{file.filename}"
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)