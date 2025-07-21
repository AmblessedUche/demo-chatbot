import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=os.getenv("HF_TOKEN")
)

from transformers import pipeline
import difflib
import gradio as gr

# Load QA pipeline
qa_pipeline = pipeline("question-answering", model="deepset/bert-base-cased-squad2")

# Disease information
disease_contexts = {
    "malaria": """
Malaria is a mosquito-borne parasitic infection caused by Plasmodium species.
- Symptoms: High fever, chills, sweating, headaches, nausea, vomiting, fatigue, muscle pain.
- Prevention: Mosquito nets, insect repellents, eliminating breeding sites, antimalarial medication.
- Treatment: Artemisinin-based combination therapies (ACTs).
""",
    "diabetes": """
Diabetes is a chronic condition where the body can't produce or use insulin properly.
- Symptoms: Excessive thirst, frequent urination, increased hunger, fatigue, blurred vision.
- Management: Diet, exercise, medication, insulin therapy, and blood sugar monitoring.
""",
    "hypertension": """
Hypertension (high blood pressure) occurs when the force of blood against artery walls is too high.
- Causes: Poor diet (especially salty foods), obesity, inactivity, stress, kidney disease, hormonal disorders.
- Risks: Heart disease, stroke, kidney failure.
- Management: Reduce salt intake, regular exercise, stress control, medication.
""",
    "covid": """
COVID-19 is a viral respiratory illness caused by the SARS-CoV-2 virus.
- Symptoms: Fever, dry cough, shortness of breath, fatigue, loss of taste or smell, sore throat.
- Prevention: Vaccination, masks, social distancing, and hygiene.
- Treatment: Supportive care, antiviral meds, oxygen therapy.
""",
    "asthma": """
Asthma is a chronic lung disease that inflames and narrows the airways.
- Symptoms: Wheezing, shortness of breath, chest tightness, coughing.
- Management: Avoid triggers, inhalers, medications like corticosteroids.
""",
    "tuberculosis": """
Tuberculosis (TB) is a bacterial infection that mainly affects the lungs.
- Symptoms: Persistent cough, weight loss, night sweats, fever.
- Treatment: Long course of antibiotics.
""",
    "typhoid": """
Typhoid fever is caused by Salmonella Typhi bacteria.
- Symptoms: High fever, weakness, stomach pain, headache, loss of appetite.
- Prevention: Vaccination, safe drinking water, proper sanitation.
- Treatment: Antibiotics.
"""
}

# Find disease from question
def find_closest_disease(question):
    diseases = list(disease_contexts.keys())
    matches = difflib.get_close_matches(question.lower(), diseases, n=1, cutoff=0.4)
    if matches:
        return matches[0]
    for d in diseases:
        if d in question.lower():
            return d
    return None

# Get chatbot response
def get_response(question):
    disease = find_closest_disease(question)
    context = disease_contexts.get(disease, "\n".join(disease_contexts.values()))
    result = qa_pipeline(question=question, context=context)
    return result["answer"]

# Gradio UI
iface = gr.Interface(fn=get_response, inputs="text", outputs="text", title="🩺 Medical Q&A Bot")
iface.launch(share=True)

