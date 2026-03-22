from app import VoiceAnalysisResponse, get_raw_analysis


if __name__ == "__main__":
    path = input("Enter path to audio file (for example, test.wav): ").strip().strip('"')
    print("Please wait, analyzing audio...")
    try:
        result_text = get_raw_analysis(path)
        print(VoiceAnalysisResponse(analysis=result_text).model_dump_json(ensure_ascii=False, indent=2))
    except Exception as error:
        print(f"Error: {error}")