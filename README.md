🧮 Full-Stack AI Math SolverLive Demo: [YOUR_VERCEL_LINK_HERE]

An end-to-end, full-stack machine learning web application that allows users to draw mathematical equations on a digital canvas and instantly receive the calculated answer. Unlike basic wrappers around existing APIs, this project features a custom-trained TensorFlow Convolutional Neural Network (CNN) for handwritten character recognition and a custom computer vision pipeline for image processing.

🛠️ Tech Stack

Machine Learning: TensorFlow, Keras, Numpy
Computer Vision: OpenCV (cv2)
Backend: Python, FastAPI, Uvicorn, SymPy
Frontend: HTML, CSS, Vanilla JavaScript
Deployment: Render (Python API), Vercel (Static Frontend)

🧠 How It Works (The Pipeline)

Frontend Capture: A user draws an equation on the HTML canvas. The drawing is converted to a Base64 image and sent to the FastAPI backend.
Computer Vision (OpenCV): * The image is converted to grayscale and thresholded.
Dilation connects loose strokes (e.g., the two parallel lines of an = sign).
Contour Detection isolates individual symbols and creates bounding boxes.Aspect-Ratio 
Padding: Bounding boxes are mathematically padded into perfect squares to prevent distortion (the "squish" bug) before resizing them to the 28x28 pixel format required by the AI.
Symbols are sorted left-to-right based on spatial coordinates.
Machine Learning Inference: The processed 28x28 squares are fed into a custom-trained TensorFlow CNN, which predicts the mathematical symbol (0-9, +, -, *, /, =) with high accuracy.
Symbolic Computation: The predicted symbols are concatenated into a string and safely evaluated using SymPy, avoiding the security risks of native eval() functions. The final string and calculated answer are returned to the user interface.

🚀 Future RoadmapExpand the dataset to include alphanumeric characters (x, y, z) and advanced operators ($\int$, $\sum$, $\sqrt{}$).
Implement 2D spatial parsing to support fractions, exponents, and complex algebraic equations.








































