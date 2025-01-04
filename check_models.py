import google.generativeai as genai

def list_available_models():
    try:
        # Configure Gemini API
        genai.configure(api_key='AIzaSyB_-GoxMD_Myomyk161VloCIVCqGRYsjSE')  # Replace with your API key
        
        # List available models
        models = genai.list_models()
        
        print("Available Models:")
        for model in models:
            print(f"Model Name: {model.name}")
    
    except Exception as e:
        print(f"Error listing models: {str(e)}")

if __name__ == "__main__":
    list_available_models()
