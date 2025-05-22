from transformers import AutoProcessor, AutoModelForImageClassification
from PIL import Image
import cv2
import torch

model_name = "Falconsai/nsfw_image_detection"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)


def classify_image(path):
    # Open the image
    image = Image.open(path).convert("RGB")

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)

    # Get the predicted class and confidence
    predicted_class = probabilities.argmax(dim=1).item()
    confidence = probabilities[0, predicted_class].item()

    # Get the label (NSFW or SFW)
    labels = model.config.id2label
    predicted_label = labels[predicted_class]

    return predicted_label, confidence


# Function to blur the image
def blur_image(image_path, output_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (51, 51), 30)

    # Save the blurred image
    cv2.imwrite(output_path, blurred_image)


def classify_and_blur_image(image_path, output_path, blur_threshold=0.5):

    print(f"Processing image: {image_path}")

    # Classify the image
    label, confidence = classify_image(image_path)

    # Check if the image is NSFW
    if label == "NSFW" and confidence > blur_threshold:
        print(f"NSFW content detected! Confidence: {confidence:.2f}. Blurring the image...")
        blur_image(image_path, output_path)
        print(f"Blurred image saved at: {output_path}")
    else:
        print(f"Image classified as: {label} with confidence {confidence:.2f}")
        print("No blurring applied.")


if __name__ == "__main__":
    image_path = "~/Desktop/nsfw-test-data/image.png"

    output_path = "~/Desktop/nsfw-test-data/blurred_image.png"

    # Classify and blur the image if necessary
    classify_and_blur_image(image_path, output_path)
