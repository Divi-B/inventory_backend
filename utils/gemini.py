import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_gemini_summary(predictions: list[dict]) -> str:
    prompt = (
        "You are an expert in inventory forecasting.\n"
        "Here are some prediction results:\n"
        f"{predictions}\n\n"
        "Please summarize key insights from these results."
    )
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text
