"""
Farmer Communication Module — Marathi Text + Audio Output
============================================================
Takes the diagnosis result (from gradcam_explainability.py) and converts
it into simple Marathi text, plus an optional spoken audio file — so a
farmer with low literacy can still understand the result.

Install dependencies:
    pip install gTTS --break-system-packages

Note: gTTS (Google Text-to-Speech) requires an internet connection to
generate audio. If you're offline, it will just skip audio and print
the Marathi text instead.
"""

from gtts import gTTS
import os
import concurrent.futures

# ----------------------------------------------------------------------
# MARATHI TRANSLATIONS — pre-written by disease class (not machine
# translated) so the wording is accurate, natural, and farmer-friendly.
# Edit/improve these with a native speaker's review before your
# conclave presentation.
# ----------------------------------------------------------------------
DISEASE_NAME_MARATHI = {
    "Bacterial_spot": "जिवाणूजन्य ठिपके रोग (बॅक्टेरियल स्पॉट)",
    "Early_blight": "लवकर येणारा करपा रोग (अर्ली ब्लाइट)",
    "Late_blight": "उशिरा येणारा करपा रोग (लेट ब्लाइट)",
    "Leaf_Mold": "पानावरील बुरशी रोग (लीफ मोल्ड)",
    "Septoria_leaf_spot": "सेप्टोरिया पानांवरील ठिपके रोग",
    "Spider_mites Two-spotted_spider_mite": "कोळी किडीचा प्रादुर्भाव (स्पायडर माइट)",
    "Target_Spot": "लक्ष्य ठिपके रोग (टार्गेट स्पॉट)",
    "Tomato_Yellow_Leaf_Curl_Virus": "पिवळा पर्णगुच्छ विषाणू रोग",
    "Tomato_mosaic_virus": "मोझॅक विषाणू रोग",
    "healthy": "झाड निरोगी आहे",
    "powdery_mildew": "भुरी रोग (पावडरी मिल्ड्यू)",
}

RECOMMENDATION_MARATHI = {
    "Bacterial_spot": "तांबेयुक्त बुरशीनाशक (कॉपर ऑक्सिक्लोराईड, २.५ ग्रॅम प्रति लिटर पाणी) फवारणी करा. वरून पाणी देणे टाळा, कारण त्यामुळे रोग पसरतो.",
    "Early_blight": "मॅन्कोझेब ७५% डब्ल्यूपी (०.२५%) फवारणी करा आणि खालची जुनी, संक्रमित पाने काढून टाका.",
    "Late_blight": "हा रोग वेगाने पसरतो. कॉपर ऑक्सिक्लोराईड (२.५ ग्रॅम प्रति लिटर) प्रतिबंधात्मक फवारणी करा; तीव्र प्रादुर्भाव असल्यास मेटालॅक्सिल + मॅन्कोझेब वापरा आणि कृषी तज्ञांचा सल्ला घ्या.",
    "Leaf_Mold": "झाडांमधील अंतर वाढवून हवा खेळती ठेवा आणि क्लोरोथॅलोनिल किंवा मॅन्कोझेब फवारणी करा.",
    "Septoria_leaf_spot": "मॅन्कोझेब ७५% डब्ल्यूपी (०.२५%) फवारणी करा, संक्रमित पाने काढून टाका आणि ३-४ वर्षांनी पीक फेरपालट करा.",
    "Spider_mites Two-spotted_spider_mite": "कडुलिंब तेल किंवा योग्य माइटनाशक (मिटिसाइड) फवारणी करा. सामान्य बुरशीनाशकाने कोळी किडे नियंत्रित होत नाहीत.",
    "Target_Spot": "मॅन्कोझेब किंवा क्लोरोथॅलोनिल फवारणी करा आणि पीक फेरपालट करा.",
    "Tomato_Yellow_Leaf_Curl_Virus": "यावर थेट औषध नाही. पांढरी माशी (व्हाईटफ्लाय) नियंत्रित करणे हाच उपाय आहे. इमिडाक्लोप्रिड सारखे कीटकनाशक वापरा आणि संक्रमित झाडे उपटून नष्ट करा.",
    "Tomato_mosaic_virus": "यावर उपचार नाही. संक्रमित झाडे त्वरित उपटून नष्ट करा आणि हात व अवजारे साबणाने स्वच्छ करा, कारण स्पर्शाने रोग पसरतो.",
    "healthy": "तुमचे पीक निरोगी आहे. काळजीपूर्वक निरीक्षण सुरू ठेवा.",
    "powdery_mildew": "गंधकयुक्त (सल्फर) बुरशीनाशक फवारणी करा.",
}

# ----------------------------------------------------------------------
# IMPORTANT SAFETY NOTE — read before using in a real deployment
# ----------------------------------------------------------------------
# These recommendations are based on published ICAR and agricultural
# research literature, using ACTIVE INGREDIENT names (not brand names),
# since exact branded products, availability, and legal registration
# vary by Indian state. Before this reaches real farmers:
#   1. Have a licensed agronomist or your guide review every entry.
#   2. Exact dosages can vary by local pest resistance, weather, and
#      crop stage — these are general starting points, not prescriptions.
#   3. Always tell farmers to confirm with their local Krishi Vigyan
#      Kendra (KVK) or agricultural extension officer before purchase,
#      and to follow the product label exactly (safety gear, re-entry
#      interval, pre-harvest interval).
#   4. Some products/dosages require specific certification to
#      recommend commercially in India — this is general educational
#      guidance, not a substitute for professional agronomic advice.
DISCLAIMER_MARATHI = (
    "सूचना: ही शिफारस सर्वसाधारण मार्गदर्शनासाठी आहे. फवारणीपूर्वी कृपया "
    "आपल्या जवळच्या कृषी विज्ञान केंद्र (KVK) किंवा कृषी अधिकाऱ्यांचा सल्ला "
    "अवश्य घ्या आणि औषधाच्या लेबलवरील सूचनांचे पालन करा."
)


def build_marathi_message(predicted_class, confidence, yield_loss_range):
    disease_mr = DISEASE_NAME_MARATHI.get(predicted_class, predicted_class)
    recommendation_mr = RECOMMENDATION_MARATHI.get(predicted_class, "")
    confidence_pct = int(confidence * 100)
    loss_low, loss_high = yield_loss_range

    if predicted_class == "healthy":
        message = (
            f"निदान: {disease_mr}. "
            f"विश्वासार्हता: {confidence_pct} टक्के. "
            f"{recommendation_mr}"
        )
    else:
        message = (
            f"निदान: तुमच्या झाडाला {disease_mr} झाला आहे. "
            f"विश्वासार्हता: {confidence_pct} टक्के. "
            f"उपचार न केल्यास अंदाजे {loss_low} ते {loss_high} टक्के उत्पादन कमी होऊ शकते. "
            f"सल्ला: {recommendation_mr} "
            f"{DISCLAIMER_MARATHI}"
        )

    return message


def speak_marathi(message, save_path="diagnosis_audio_marathi.mp3", timeout_seconds=10):
    """Generate Marathi audio via gTTS, bounded by a hard timeout.

    gTTS calls an external Google endpoint with no built-in timeout of
    its own. On some cloud hosts (e.g. Render's datacenter IPs), that
    call can hang indefinitely instead of failing quickly, which — left
    unbounded — eventually triggers gunicorn's own worker timeout and
    kills the whole request, truncating the response the browser was
    waiting on. Bounding it here means a slow/blocked TTS call fails
    fast and the rest of the diagnosis (text, heatmap) still returns
    successfully, just without audio.
    """
    def _generate():
        tts = gTTS(text=message, lang="mr")
        tts.save(save_path)
        return save_path

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_generate)
            result = future.result(timeout=timeout_seconds)
            print(f"Audio saved to {save_path}")
            return result
    except concurrent.futures.TimeoutError:
        print(f"gTTS call exceeded {timeout_seconds}s and was abandoned (likely blocked/slow on this host).")
        print("Continuing without audio — text result is still available.")
        return None
    except Exception as e:
        print(f"Could not generate audio: {e}")
        print("Text result is still available below.")
        return None


# ----------------------------------------------------------------------
# COMBINED PIPELINE — call this after diagnose_leaf() from
# gradcam_explainability.py
# ----------------------------------------------------------------------
def communicate_result(predicted_class, confidence, yield_loss_range, speak=True):
    message = build_marathi_message(predicted_class, confidence, yield_loss_range)

    print("\n=== शेतकऱ्यांसाठी संदेश (Farmer Message - Marathi) ===")
    print(message)

    if speak:
        speak_marathi(message)

    return message


# ----------------------------------------------------------------------
# EXAMPLE USAGE (standalone test, without running the full model)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example: simulate a diagnosis result
    example_result = communicate_result(
        predicted_class="Early_blight",
        confidence=0.94,
        yield_loss_range=(20, 50),
        speak=True,
    )
