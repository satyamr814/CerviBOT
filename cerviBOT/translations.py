"""
Translation dictionary for English and Hindi
"""
TRANSLATIONS = {
    'en': {
        # UI Elements
        'title': 'Cervical Health Risk Assessment',
        'subtitle': 'Answer a few questions to assess your risk level',
        'lang_button': 'हिंदी',
        'hello': "Hello! I'm here to help assess your cervical health risk. Let's start with a few questions.",
        'analyzing': "Thanks for your answers! Analyzing your risk...",
        'risk_assessment_result': 'Risk Assessment Result',
        'probability': 'Probability',
        'label': 'Label',
        'confidence': 'Confidence',
        'why_prediction': 'Why this prediction?',
        'next': 'Next',
        'question_of': 'Question {current} of {total}',
        
        # Questions
        'q_age': 'What is your age?',
        'q_partners': 'How many sexual partners have you had?',
        'q_first_sex': 'At what age did you first have sexual intercourse?',
        'q_pregnancies': 'How many pregnancies have you had?',
        'q_smoking': 'How many years have you been smoking? (Enter 0 if you don\'t smoke)',
        'q_hormonal': 'Are you currently using hormonal contraceptives?',
        'q_hormonal_years': 'If yes, for how many years? (Enter 0 if not applicable)',
        'q_hiv': 'What is your HIV status?',
        'q_pain': 'Do you experience pain during intercourse?',
        'q_discharge_type': 'What type of vaginal discharge do you experience?',
        'q_discharge_color': 'What color is your vaginal discharge?',
        'q_bleeding': 'When do you experience abnormal vaginal bleeding?',
        
        # Options
        'opt_yes': 'Yes',
        'opt_no': 'No',
        'opt_none': 'None',
        'opt_watery': 'watery',
        'opt_thick': 'thick',
        'opt_bloody': 'bloody',
        'opt_normal': 'normal',
        'opt_pink': 'pink',
        'opt_pale': 'pale',
        'opt_between_periods': 'Between periods',
        'opt_after_sex': 'After sex',
        'opt_after_menopause': 'After menopause',
        'opt_enter_number': 'Enter a number',
        
        # Risk Levels
        'risk_low': 'Low Risk',
        'risk_medium': 'Medium Risk',
        'risk_high': 'High Risk',
        
        # Advice
        'advice_low': 'Low risk — routine screening as per local guidelines is recommended.',
        'advice_medium': 'Medium risk — consider scheduling a clinical check-up and follow-up screening.',
        'advice_high': 'High risk — seek urgent clinical evaluation and further diagnostic testing.',
        
        # Errors
        'error_network': 'Network or server error: {error}',
        'error_explanation': 'Error getting explanation: {error}',
    },
    'hi': {
        # UI Elements
        'title': 'गर्भाशय ग्रीवा स्वास्थ्य जोखिम मूल्यांकन',
        'subtitle': 'अपने जोखिम स्तर का आकलन करने के लिए कुछ प्रश्नों के उत्तर दें',
        'lang_button': 'English',
        'hello': 'नमस्ते! मैं आपके गर्भाशय ग्रीवा स्वास्थ्य जोखिम का आकलन करने में मदद करने के लिए यहां हूं। आइए कुछ प्रश्नों से शुरू करें।',
        'analyzing': 'आपके उत्तरों के लिए धन्यवाद! आपके जोखिम का विश्लेषण किया जा रहा है...',
        'risk_assessment_result': 'जोखिम मूल्यांकन परिणाम',
        'probability': 'संभावना',
        'label': 'लेबल',
        'confidence': 'आत्मविश्वास',
        'why_prediction': 'यह भविष्यवाणी क्यों?',
        'next': 'अगला',
        'question_of': 'प्रश्न {current} में से {total}',
        
        # Questions
        'q_age': 'आपकी उम्र क्या है?',
        'q_partners': 'आपके कितने यौन साथी रहे हैं?',
        'q_first_sex': 'आपने पहली बार किस उम्र में यौन संबंध बनाए?',
        'q_pregnancies': 'आपकी कितनी गर्भावस्थाएं रही हैं?',
        'q_smoking': 'आप कितने वर्षों से धूम्रपान कर रहे हैं? (यदि आप धूम्रपान नहीं करते हैं तो 0 दर्ज करें)',
        'q_hormonal': 'क्या आप वर्तमान में हार्मोनल गर्भनिरोधक उपयोग कर रहे हैं?',
        'q_hormonal_years': 'यदि हाँ, तो कितने वर्षों से? (यदि लागू नहीं होता तो 0 दर्ज करें)',
        'q_hiv': 'आपकी HIV स्थिति क्या है?',
        'q_pain': 'क्या आपको संभोग के दौरान दर्द होता है?',
        'q_discharge_type': 'आपको किस प्रकार का योनि स्राव होता है?',
        'q_discharge_color': 'आपके योनि स्राव का रंग क्या है?',
        'q_bleeding': 'आपको असामान्य योनि रक्तस्राव कब होता है?',
        
        # Options
        'opt_yes': 'हाँ',
        'opt_no': 'नहीं',
        'opt_none': 'कोई नहीं',
        'opt_watery': 'पानी जैसा',
        'opt_thick': 'गाढ़ा',
        'opt_bloody': 'खूनी',
        'opt_normal': 'सामान्य',
        'opt_pink': 'गुलाबी',
        'opt_pale': 'पीला',
        'opt_between_periods': 'मासिक धर्म के बीच',
        'opt_after_sex': 'संभोग के बाद',
        'opt_after_menopause': 'रजोनिवृत्ति के बाद',
        'opt_enter_number': 'एक संख्या दर्ज करें',
        
        # Risk Levels
        'risk_low': 'कम जोखिम',
        'risk_medium': 'मध्यम जोखिम',
        'risk_high': 'उच्च जोखिम',
        
        # Advice
        'advice_low': 'कम जोखिम — स्थानीय दिशानिर्देशों के अनुसार नियमित जांच की सिफारिश की जाती है।',
        'advice_medium': 'मध्यम जोखिम — नैदानिक जांच और अनुवर्ती जांच शेड्यूल करने पर विचार करें।',
        'advice_high': 'उच्च जोखिम — तत्काल नैदानिक मूल्यांकन और आगे की नैदानिक जांच कराएं।',
        
        # Errors
        'error_network': 'नेटवर्क या सर्वर त्रुटि: {error}',
        'error_explanation': 'स्पष्टीकरण प्राप्त करने में त्रुटि: {error}',
    }
}

def translate(key: str, lang: str = 'en', **kwargs) -> str:
    """Get translation for a key in specified language."""
    lang_dict = TRANSLATIONS.get(lang, TRANSLATIONS['en'])
    text = lang_dict.get(key, key)
    
    # Replace placeholders
    if kwargs:
        try:
            text = text.format(**kwargs)
        except:
            pass
    
    return text

