
from pipeline import QAPipeline

if __name__ == "__main__":
    try:
        pipeline = QAPipeline(log_level="WARNING")
        
        # Example question
        question = "SQL注入漏洞是什么含义，有哪些攻击手段，什么时候发现的，影响面有多大。"
        
        print("Processing question:", question)
        answer = pipeline.answer_question(question)
        print("\nAnswer:", answer)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")