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
caution = ["Criminal","childcare","community","theatre","theater","politics","Nursing","physiology","pHd","Philosophy","bachelor of design","bdes","bhms","bams","homeopathy","Ayurvedic","ayurveda", "Investigation", "Community & social service management", "Journalism","cordwainers","footwear",
          "Media", "Film", "Music", "TV", "Performing arts","fashion", "theatre", "Fashion design", 
          "Interior design", "Community integration", "Early childhood management", 
          "Early learning", "Educational support", "social service", 
          "Education support","home","Plumbing","Hospitality", "Hotel", "Sports", "Media", "Book keeping", 
          "Career development", "recreation", "leisure", "Community","Mphil","mst","literature","dphil","politics","Russian","slavonic","bphil","language","languages","medieval","islamic","wildlife","history","education","ecowild","Chinese","archaeology","life"]

direct_mappings = {
   "drug":"Me",
   "management": "Mng",
   "dental": "Dn",
   "dentistry": "Dn",
   "engineering": "Eng",
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
          "political":"O",
          "social":"O",
"literature":"O",
"theology":"O",
   "bs":"Sc",
   "ms":"Sc",
   "msc":"Sc",
   "bsc":"Sc",
   "mbbs":"Me",
   "Btech":"Eng",
   "Mtech":"Eng",
          "design": "De",
   "Science":"Sc",
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
       return f"Course type for {course_name}: {predicted_category}\nCaution type for {course_name}: Y"
   else:
       caution_category = predict_caution_category(course_name)
       return f"Course type for {course_name}: {predicted_category}\nCaution type for {course_name}: {caution_category}"

# HTML content for description
html_content = """
<div style="text-align:center; margin-bottom:20px;">
  <h2>Course Mapping Categories</h2>
</div>
<div style="display: flex">

  <div style="flex:1">
    <h3>Terminology for the course type</h3>
    <ul>
      <li>Dn : Dental course</li>
      <li>De : Design course</li>
      <li>Eng : Engineering course (STEM)</li>
      <li>L : Law course</li>
      <li>Mng : Management course</li>
      <li>Me : Medical course</li>
      <li>Mm : Mathematics course (STEM)</li>
      <li>Sc : Science course (STEM)</li>
      <li>Tech : Technology course (STEM)</li>
      <li>O : Others</li>
    </ul>
  </div>
  <div style="flex:1;">
    <h3>Terminology for the caution type</h3>
    <ul>
      <li>Y : Caution flag Yes, Proceed after confirmation from product team</li>
      <li>N : Caution Flag No, You can proceed</li>
    </ul>
  </div>
</div>
"""

iface = gr.Interface(
        fn=predict_course, 
        inputs=gr.Textbox(label="Enter Course Name"), 
        outputs=gr.Textbox(label="Predictions"), description=html_content
    )

iface.launch(server_name="0.0.0.0",server_port=int(os.environ.get("PORT", 8080)))
