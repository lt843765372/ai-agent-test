
import asyncio

from langchain_ollama import ChatOllama

llm = ChatOllama(model="deepseek-r1:7b")

async def test_ainvoke():
    prompt = "生成一个简短的python函数注释示例"
    response = await llm.ainvoke(prompt)
    print("异步完整结果123", response.content)


async def test_astream():
    prompt = "解释异步编程的核心概念"
    stream = llm.astream(prompt)
    print("异步流式结果456:",stream, end="")
    async for line in stream:
        print(line.content,end="")


asyncio.run(test_astream())
