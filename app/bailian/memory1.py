import os
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()
# 这段代码实现了一个 “带记忆的 AI 对话功能”：它用 LangGraph 搭了一个最简单的对话流程，每次你输入问题，代码会调用通义千问模型生成回答，
# #同时 MemorySaver 会自动把对话历史存起来，下次提问时模型就能参考之前的内容，不会像普通单次调用那样 “失忆”。

# langchain# 线性工具链
# langgraph# 有状态的图结构、是chain的升级，相当于给chain加了决策分支、记忆留存的能力。不仅能按顺序执行任务，还能根据不同结果走不同流程，甚至能记住之前的操作来回溯或循环，做复杂的多步骤任务会更灵活。
# LangGraph里的 “决策分支” 功能吧。举个例子，你可以在流程里加一个判断：如果用户问的是天气，就调用天气
# API；如果问的是知识，就直接让模型回答。这个判断的过程就是决策，LangGraph
# 能轻松实现这种分支逻辑，而# LangChain# 就很难做到。


# 记住 “StateGraph 搭框架、add_node 加功能、add_edge 连流程” 这三步，
# #就能轻松用 LangGraph 搭建出复杂的 AI 应用了。



# **********定义一个新的图**********
workflow = StateGraph(state_schema=MessagesState)
# StateGraph 是 LangGraph 库的核心类，专门用来构建有状态、可回溯的工作流。你可以把它理解成一个 “智能流程图画布”：你能往里面添加 “节点”，比如调用模型、调用工具、做逻辑判断，还能定义节点之间的跳转关系，比如
# “调用工具后如果拿到结果就整理回答，如果没拿到就重新调用”。它最强大的地方是自带 “状态管理”，能自动记住每个节点的运行结果，方便你随时回溯或修改流程

model = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=SecretStr(os.getenv("DASHSCOPE_API_KEY")),
    streaming=True,
)

# 定义调用模型的函数
def call_model(state: MessagesState):
    response = model.invoke(state["messages"]) #是把对话存在单次运行的内存里，程序一关就没了
    return {"messages": response}
# MessagesState 类型注解


# **********定义图中的（单个）节点**********
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# 添加内存持久化
memory = MemorySaver()
# MemorySaver 是把对话持久化保存下来，下次启动程序还能找回之前的聊天记录
# MemorySaver 不需要你手动传对话内容，因为它和 LangGraph 的工作流是绑定的 —— 每次工作流里的 state 有更新，比如新增了用户提问或模型回复，LangGraph 会自动把最新的 state 同步给 MemorySaver 保存起来，整个过程是框架自动完成的，不需要你额外写代码

app = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}

query = "Hi! I'm Bob."

input_messages = [HumanMessage(query)]
# AIMessage、SystemMessage、FunctionMessage 这些类型，分别用来封装 AI 的回复、系统的提示指令和工具调用的结果
# 这是 LangChain 生态里固定的结构化消息格式，不同的消息类型会有不同的处理逻辑。比如 SystemMessage 会被模型当作系统指令优先执行，FunctionMessage 会触发工具调用，而 HumanMessage 和 AIMessage 则是普通的对话内容
output = app.invoke({"messages": input_messages}, config)
# 这是考虑到代码上线后的实际使用场景，比如把这个对话功能做成网站或 APP，就会有很多人同时用，这时候 thread_id 就像每个用户的 “专属聊天房间号”，能保证每个人的聊天记录独立不串线。你自己练习的时候用不到多用户，但加上这个参数会让代码更规范、更接近真实项目。
# 你可以给每个用户分配唯一的 ID，比如用用户的手机号、用户名或者随机生成的 UUID 作为 thread_id，把它存在用户的会话信息里，每次用户发消息时就用对应的 ID 来调用 app.invoke，这样就能区分不同用户的聊天记录了。

# output["messages"][-1].pretty_print()  # 输出包含了所有消息的状态；栈结构-后进先出 输出最新的一次回复

# 6. 持续对话循环（终端交互）
if __name__ == "__main__":
    # 这行代码是Python 脚本的 “启动开关”。当你直接运行这个.py 文件时，开关打开，下面的代码会自动执行；但如果别的项目只是想借用这个文件里的函数或工具，开关会关闭，不会触发自动运行的逻辑，避免干扰其他项目。
    print("✨ 开始对话（输入「退出」结束）✨")
    while True: # while True 会创建一个无限循环
        try:
            # 获取终端输入
            user_input = input("你说：")
            # 除了 input ()，还有 sys 模块的 sys.stdin 可以读取多行输入，适合处理大段文本；print () 的进阶用法，比如加 end 参数来取消换行，或者用 sep 来指定分隔符；另外还有 getpass 模块的 getpass ()，可以输入密码而不显示明文，适合做简单的登录验证。

            # 退出逻辑
            if user_input.strip() in ["退出", "exit", "q"]:
                print("👋 对话结束！")
                break

            # ✅ 核心修复：输入格式是字典，key为messages，值是消息列表
            input_data = {"messages": [HumanMessage(content=user_input)]}

            # 调用模型（传入正确格式的输入和config）
            result = app.invoke(input_data, config=config)

            # 提取并打印AI最新回复（取最后一条AIMessage）
            ai_reply = result["messages"][-1].content
            print("AI：", ai_reply)

        except Exception as e:
            print(f"❌ 出错了：{str(e)}")
            continue


