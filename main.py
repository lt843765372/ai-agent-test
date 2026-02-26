# def main():
#     print("Hello from ai-agent-test!")
#
#
# if __name__ == "__main__":
#     main()

from langchain_ollama import ChatOllama

if __name__ == "__main__":
    llm = ChatOllama(
        model="deepseek-r1:7b"
    )
    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]
    # ai_msg = llm.invoke(messages)
    ai_msg = llm.stream(messages)
    for msg in ai_msg:
        print(msg)
    print(ai_msg)
