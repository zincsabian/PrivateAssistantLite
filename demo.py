from pipeline import QAPipeline

if __name__ == "__main__":
    try:
        pipeline = QAPipeline(log_level="DEBUG")

        while True:
            question = input("Please input your question: ")
            print("User:\n", question)
            answer = pipeline.answer_question(question)
            print("Assistant:\n", answer)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
