from transformers import pipeline

# Load available translation pipelines
translate_en_to_ig = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ig")
translate_en_to_ha = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ha")
# translate_en_to_yo = pipeline("translation", model="Helsinki-NLP/opus-mt-en-yo")  # Not available

translate_ig_to_en = pipeline("translation", model="Helsinki-NLP/opus-mt-ig-en")
translate_ha_to_en = pipeline("translation", model="Helsinki-NLP/opus-mt-ha-en")
translate_yo_to_en = pipeline("translation", model="Helsinki-NLP/opus-mt-yo-en")

def translate(text, source_lang, target_lang):
    if source_lang == "en" and target_lang == "ig":
        return translate_en_to_ig(text)[0]['translation_text']
    elif source_lang == "en" and target_lang == "ha":
        return translate_en_to_ha(text)[0]['translation_text']
    elif source_lang == "en" and target_lang == "yo":
        return "[Translation to Yoruba not available]"
    elif source_lang == "ig" and target_lang == "en":
        return translate_ig_to_en(text)[0]['translation_text']
    elif source_lang == "ha" and target_lang == "en":
        return translate_ha_to_en(text)[0]['translation_text']
    elif source_lang == "yo" and target_lang == "en":
        return translate_yo_to_en(text)[0]['translation_text']
    else:
        return text
