from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="deepset/bert-base-cased-squad2")

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

def get_response(question):
    q_lower = question.lower()
    context = None
    for disease in disease_contexts:
        if disease in q_lower:
            context = disease_contexts[disease]
            break
    if context is None:
        context = "\n".join(disease_contexts.values())
    result = qa_pipeline(question=question, context=context)
    return result["answer"]

if __name__ == "__main__":
    print("Type 'exit' to end the chat.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = get_response(user_input)
        print(f"Chatbot: {response}")
