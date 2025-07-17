# ===============================
# تحميل البيانات ومعرفة شكلها
# ===============================
from sklearn.datasets import load_digits
digits = load_digits()
print("Number of images:", len(digits.images))
print("Shape of the first image:", digits.images[0].shape)

# ===============================
# عرض أول صورة
# ===============================
import matplotlib.pyplot as plt
plt.gray()
plt.matshow(digits.images[0])
# plt.show() 

# ===============================
# تقسيم البيانات إلى تدريب واختبار
# ===============================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

# ===============================
# تدريب النموذج MLPClassifier
# ===============================
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ===============================
# تقييم أداء النموذج
# ===============================
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the model:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# ===============================
# واجهة المستخدم باستخدام Gradio
# ===============================
import gradio as gr
import numpy as np
from PIL import Image

def predict_digit(data):
    try:
        img = data["composite"]
        img = Image.fromarray(img).convert("L")
        img = img.resize((8, 8))
        img = np.array(img)
        if img.mean() > 250:
            return "Please draw a darker digit!"
        img = 16 - (img / 255.0 * 16)
        img = img.flatten().reshape(1, -1)
        prediction = model.predict(img)[0]
        return f"Predicted Digit: {prediction}"
    except Exception as e:
        return f"Error: {str(e)}"

app = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(canvas_size=(100, 100)),
    outputs="text"
)

app.launch(share=True)
