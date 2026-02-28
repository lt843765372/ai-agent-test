from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()
# 定义对话图
workflow = StateGraph(state_schema=MessagesState)

# 初始化模型（修复：移除冗余的 SecretStr，规范 API Key 传入）
model = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 直接传入字符串，无需 SecretStr
    streaming=False,  # 若不需要流式输出，改为 False；需要则保留 True 并改用 stream 调用
)

# 修复：正确更新 MessagesState（累加消息而非覆盖）
def call_model(state: MessagesState):
    # 调用模型，传入历史消息
    response = model.invoke(state["messages"])
    # 关键：返回的是「新增的消息列表」，而非直接覆盖 messages
    return {"messages": [response]}

# 添加节点和边（修复：补全图的闭环逻辑）
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")  # 起始节点指向 model
# 单个节点场景：model 节点作为结束，无需额外边

# 添加内存持久化
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# 配置对话线程 ID
config = {"configurable": {"thread_id": "abc123"}}

# 测试对话
query = "Hi! I'm Bob."
input_messages = [HumanMessage(content=query)]  # 注意：HumanMessage 需要指定 content 参数（原代码未显式写，低版本可能报错）
output = app.invoke({"messages": input_messages}, config)

# 输出最后一条消息（AI 回复）
output["messages"][-1].pretty_print()