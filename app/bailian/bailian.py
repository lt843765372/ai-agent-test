import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


print(os.getenv("DASHSCOPE_API_KEY"))
print(os.getenv("OPENAI_API_KEY"))


client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    # 11.在pycharm中修改-当前文件-文件运行配置-环境变量-添加DASHSCOPE_API_KEY=xxx
    # 111.用.env文件+dotenv 多个文件集中管理，+.gitignore
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    model="qwen-plus",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁？"},
    ]
)
print(completion.model_dump_json())