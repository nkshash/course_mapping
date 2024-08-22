import joblib
import re
import gradio as gr
import os

# Load the models and vectorizers
model1 = joblib.load('model_course_type (1).pkl')
model2 = joblib.load('model_caution_type (1).pkl')
vectorizer1 = joblib.load('vectorizer_course_type (2).pkl')
vectorizer2 = joblib.load('vectorizer_caution_type (2).pkl')

# Define the caution list and direct mappings
caution = ["Criminal", "Investigation", "Community & social service management", "Journalism", 
           "Media", "Film", "Music", "TV", "Performing arts", "theatre", "Fashion design", 
           "Interior design", "Community integration", "Early childhood management", 
           "Early learning programmes", "Educational support", "social service worker", 
           "Education support"]

direct_mappings = {
    "design": "De",
    "management": "Mng",
    "dental": "De",
    "dentistry": "De",
    "engineering": "Eng",
    "science": "Sc",
    "Law":"L",
    "Laws":"L",
    "Medicine":"Me",
    "mathematics":"Mm",
    "Finance":"Mm",
    "Statistics":"Mm",
    "Statistical":"Mm",
    "Actuarial":"Mm",
    "Accounting":"Mng",
    "Business":"Mng",
    "Management":"Mng",
    "Analytics":"Mm",
    "Medicine":"Me",
    "Medical":"Me",
    "Nursing":"Me",
}

def predict_course_category(course_name):
    # Check for direct mappings first
    for keyword, category in direct_mappings.items():
        if keyword.lower() in course_name.lower():
            return category

    # If no direct mapping, preprocess and vectorize the course name
    course_name_processed = re.sub(r'[^\w\s]', '', course_name.lower())
    course_name_vectorized = vectorizer1.transform([course_name_processed])
    predicted_category = model1.predict(course_name_vectorized)[0]

    return predicted_category

def predict_caution_category(course_name):
    # Preprocess and vectorize the course name
    course_name_processed = re.sub(r'[^\w\s]', '', course_name.lower())
    course_name_vectorized = vectorizer2.transform([course_name_processed])
    predicted_category = model2.predict(course_name_vectorized)[0]
    return predicted_category

def check_caution_list(course_name):
    # Check if any caution word is in the course name
    for caution_word in caution:
        if caution_word.lower() in course_name.lower():
            return "Y"
    return None

def predict_course(course_name):
    # Predict course category with direct mapping
    predicted_category = predict_course_category(course_name)

    caution_mark = check_caution_list(course_name)

    if caution_mark == "Y":
        return f"Predicted category for '{course_name}': {predicted_category}\nCaution required for '{course_name}': Y"
    else:
        caution_category = predict_caution_category(course_name)
        return f"Predicted category for '{course_name}': {predicted_category}\nPredicted caution category for '{course_name}': {caution_category}"

# Create Gradio interface
iface = gr.Interface(
    fn=predict_course, 
    inputs=gr.Textbox(label="Enter Course Name"), 
    outputs=gr.Textbox(label="Predictions")
)
iface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 8080)))
iface.launch(share=True)
