
from pipeline import QAPipeline

if __name__ == "__main__":
    try:
        pipeline = QAPipeline(log_level="DEBUG")
        
        question = input("Please input your question: ")
        
        print("Processing question:", question)
        answer = pipeline.answer_question(question)
        print("\nAnswer:", answer)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")