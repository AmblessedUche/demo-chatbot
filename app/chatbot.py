from transformers import pipeline
import difflib

# Load the QA pipeline
qa_pipeline = pipeline("question-answering", model="deepset/bert-base-cased-squad2")
print("Device set to use", qa_pipeline.device)

# Disease contexts dictionary with detailed info
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

# Fuzzy match helper
def find_closest_disease(question):
    diseases = list(disease_contexts.keys())
    matches = difflib.get_close_matches(question.lower(), diseases, n=1, cutoff=0.4)
    if matches:
        return matches[0]
    for d in diseases:
        if d in question.lower():
            return d
    return None

# QA logic without fallback
def get_response(question):
    disease = find_closest_disease(question)
    if disease:
        context = disease_contexts[disease]
    else:
        # If disease not found, just combine all contexts as fallback context
        context = "\n".join(disease_contexts.values())

    result = qa_pipeline(question=question, context=context)
    return result["answer"]

# Terminal chat
if __name__ == "__main__":
    print("🤖 Chatbot ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break
        response = get_response(user_input)
        print(f"Chatbot: {response}")
