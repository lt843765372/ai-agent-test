import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_core.prompts import PromptTemplate
load_dotenv()

llm = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=SecretStr(os.getenv("DASHSCOPE_API_KEY")),
    streaming=True,
)
# SecretStr() 输出 *******

p = PromptTemplate.from_template("今天{something}真不错")
prompt = p.format(something="天气")
print(f"prompt:{prompt}")

# resp = llm.stream("300+200等于多少")
resp = llm.stream(prompt)
# 需要复用提示词部分
# 就用提示词框架

for item in resp:
    print(item.content, end="")