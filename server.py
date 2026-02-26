from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
import cv2
import tensorflow as tf
import sympy # To safely do the math

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageData(BaseModel):
    image_base64: str

# 1. LOAD THE CUSTOM BRAIN!
print("Loading Custom Math Brain...")
model = tf.keras.models.load_model('math_brain.keras')
print("Brain Loaded!")

# 2. THE ALPHABETICAL FOLDER MAPPING
# This perfectly matches the folder order TensorFlow used during training
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'div', 'eq', 'minus', 'plus', 'times']

# Map the folder names to actual math symbols
symbol_map = {
    'div': '/', 'eq': '=', 'minus': '-', 'plus': '+', 'times': '*',
    '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
    '5': '5', '6': '6', '7': '7', '8': '8', '9': '9'
}

@app.post("/solve")
async def solve_math(data: ImageData):
    try:
        header, encoded = data.image_base64.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        
        # --- THE NEW CHOPPER LOGIC ---
        
        # 1. Dilation: Thicken the lines to connect loose strokes (like the two lines of an '=' sign)
        # A 5x5 kernel usually works well for canvas drawings, but you can tweak the size.
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        
        # 2. Find contours on the DILATED image, not the original
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bounding_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # 3. Smarter filtering: Ignore tiny noise specks, but keep valid strokes
            # We filter out boxes that are just a few pixels wide/tall
            if w > 15 and h > 15:
                bounding_boxes.append((x, y, w, h))
                
        # 4. Sort left-to-right based on the x-coordinate
        bounding_boxes = sorted(bounding_boxes, key=lambda box: box[0])
        
        equation_string = ""
        
        # THE BRAIN: Read each chopped box
        for (x, y, w, h) in bounding_boxes:
            # IMPORTANT: We crop from the ORIGINAL 'thresh' image, not the blurry 'dilated' one!
            # This ensures the AI sees the crisp lines it was trained on.
            pad = 10
            y1 = max(0, y - pad)
            y2 = min(thresh.shape[0], y + h + pad)
            x1 = max(0, x - pad)
            x2 = min(thresh.shape[1], x + w + pad)
            
            # 2. Resize to 28x28 (exactly what the AI expects)
            roi = thresh[y1:y2, x1:x2]
            h_roi, w_roi = roi.shape
            diff = abs(h_roi - w_roi)
            pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0

            if h_roi > w_roi:
                pad_left = diff // 2
                pad_right = diff - pad_left
            else:
                pad_top = diff // 2
                pad_bottom = diff - pad_top

            square_roi = cv2.copyMakeBorder(roi, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
            roi_resized = cv2.resize(square_roi, (28, 28), interpolation=cv2.INTER_AREA)

            # 3. Reshape for TensorFlow (1 image, 28x28, 1 color channel)
            roi_reshaped = np.reshape(roi_resized, (1, 28, 28, 1))
            
            # 4. Predict!
            prediction = model.predict(roi_reshaped)
            best_guess_index = np.argmax(prediction)
            best_guess_folder = class_names[best_guess_index]
            
            # 5. Add it to our equation string
            equation_string += symbol_map[best_guess_folder]
            
        # THE CALCULATOR: Solve the final string
        try:
            # sympy.sympify safely converts "5+7" into actual math and solves it
            answer = str(sympy.sympify(equation_string))
        except Exception:
            answer = "Syntax Error. Try writing clearer!"

        return {
            "latex": equation_string,
            "answer": answer
        }
        
    except Exception as e:
        return {"error": str(e)}