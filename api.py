print("======== Running API from THIS FILE ========")
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os

app = Flask(__name__)
CORS(app)

model_brand = YOLO('Car_Brand/best.pt')
model_color = YOLO('Car_Color/best.pt')
model_plate = YOLO('Car_Plate/best.pt')

# قاموس الترجمة
labels_plate_brand = {
    'ALF': "أ", 'BA': "ب", 'THAA': "ت", 'THA': "ث", 'GEEM': "ج", 'HAA': "ح", 'KHA': "خ", 'DAL': "د", 'ZAL': "ذ", 'RAA': "ر",
    'ZAIN': "ز", 'SEEN': "س", 'SHEEN': "ش", 'SAAD': "ص", 'DAAD': "ض", 'TAH': "ط", 'ZAH': "ظ", 'AEN': "ع", 'GHEN': "غ", 'FAA': "ف",
    'KAAF': "ق", 'CAAF': "ك", 'LAM': "ل", 'MEEM': "م", 'NOON': "ن", 'HA': "ه", "WAW": "و", "YAA": "ي"
}

def extract_plate_labels(yolo_result, model):
    print(">> Using extract_plate_labels function")
    boxes_data = yolo_result[0].boxes.data
    letters = []
    numbers = []

    if boxes_data is not None and len(boxes_data) > 0:
        for row in boxes_data:
            x_center = (row[0].item() + row[2].item()) / 2
            class_id = int(row[5].item())
            english_label = model.names[class_id]
            translated_label = labels_plate_brand.get(english_label, english_label)

            if translated_label.isdigit():
                numbers.append((x_center, translated_label))
            else:
                letters.append((x_center, translated_label))

        letters_sorted = [item[1] for item in sorted(letters, key=lambda x: x[0], reverse=True)]
        
        numbers_sorted = [item[1] for item in sorted(numbers, key=lambda x: x[0])]

        final_plate = letters_sorted + numbers_sorted

        return " ".join(final_plate)
    return "NO DETECTION"


def extract_top_label(yolo_result, model):
    boxes = yolo_result[0].boxes
    if boxes and len(boxes.cls) > 0:
        top_idx = boxes.conf.argmax().item()
        class_id = int(boxes.cls[top_idx].item())
        return model.names[class_id]
    return "NO DETECTION"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image']
    image_path = os.path.join("uploads", image.filename)
    os.makedirs("uploads", exist_ok=True)
    image.save(image_path)

    brand_result = model_brand(image_path, conf=0.3)
    color_result = model_color(image_path, conf=0.3)
    plate_result = model_plate(image_path, conf=0.3)
    print("Detected plate items: ", [model_plate.names[int(row[5].item())] for row in plate_result[0].boxes.data])


    # طباعة للتأكد
    print("Detected classes in plate: ", [model_plate.names[int(row[5].item())] for row in plate_result[0].boxes.data])

    plate_label = extract_plate_labels(plate_result, model_plate)
    brand_label = extract_top_label(brand_result, model_brand)
    color_label = extract_top_label(color_result, model_color)

    response = {
        'car_plate': plate_label,
        'car_brand': labels_plate_brand.get(brand_label, brand_label),
        'car_color': color_label
    }

    os.remove(image_path)
    return jsonify(response)

if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

