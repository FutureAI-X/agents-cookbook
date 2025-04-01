from typing import TypedDict, Annotated, List, Any, Optional
from langchain_core.messages import AnyMessage
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
import base64
from IPython.display import Image, display

# Step1 初始化大模型
llm = ChatOpenAI(
    base_url="https://api-inference.modelscope.cn/v1/",
    api_key="954a12f1-1c3f-4756-b133-248866adfd7b",
    model="Qwen/Qwen2.5-32B-Instruct",
    temperature=0
)

# Step2 定义状态：所谓状态指在整个流程中需要跟踪的信息
class AgentState(TypedDict):
    # The input document
    input_file:  Optional[str]  # Contains file path, type (PNG)

    # 这个状态比我们之前见过的稍微复杂些。 AnyMessage 是来自 langchain 的类，用于定义消息，而 add_messages 是一个操作符，它会添加最新消息而不是覆盖现有状态。
    messages: Annotated[list[AnyMessage], add_messages]

# Step3 准备工具并绑定
vision_llm = ChatOpenAI(
    base_url="https://api-inference.modelscope.cn/v1/",
    api_key="954a12f1-1c3f-4756-b133-248866adfd7b",
    model="Qwen/Qwen2.5-VL-72B-Instruct",
    temperature=0
)
def extract_text(img_path: str) -> str:
    """
    Extract text from an image file using a multimodal model.

    Args:
        img_path: A local image file path (strings).

    Returns:
        A single string containing the concatenated text extracted from each image.
    """
    all_text = ""
    try:
       
        # Read image and encode as base64
        with open(img_path, "rb") as image_file:
            image_bytes = image_file.read()

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Prepare the prompt including the base64 image data
        message = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Extract all the text from this image. "
                            "Return only the extracted text, no explanations."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                ]
            )
        ]

        # Call the vision-capable model
        response = vision_llm.invoke(message)

        # Append extracted text
        all_text += response.content + "\n\n"

        return all_text.strip()
    except Exception as e:
        # You can choose whether to raise or just return an empty string / error message
        error_msg = f"Error extracting text: {str(e)}"
        print(error_msg)
        return ""

def divide(a: int, b: int) -> float:
    """Divide a and b."""
    return a / b

tools = [
    divide,
    extract_text
]
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

# Step4 定义节点
def assistant(state: AgentState):
    # System message
    textual_description_of_tool="""
extract_text(img_path: str) -> str:
    Extract text from an image file using a multimodal model.

    Args:
        img_path: A local image file path (strings).

    Returns:
        A single string containing the concatenated text extracted from each image.
divide(a: int, b: int) -> float:
    Divide a and b
"""
    image=state["input_file"]
    sys_msg = SystemMessage(content=f"You are an helpful agent that can analyse some images and run some computatio without provided tools :\n{textual_description_of_tool} \n You have access to some otpional images. Currently the loaded images is : {image}")


    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])],"input_file":state["input_file"]}

# Step5 创建 StateGraph 并定义边（将所有内容连接在一起）
# 创建 graph
builder = StateGraph(AgentState)

# 添加 nodes
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

# 编译 graph
react_graph = builder.compile()

# Step6 案例测试
def math():
    messages = [HumanMessage(content="Divide 6790 by 5")]
    messages = react_graph.invoke({"messages": messages, "input_file": None})
    for m in messages['messages']:
        m.pretty_print()

def image_handle():
    messages = [HumanMessage(content="根据提供的图片，显示的日期和时间是？")]
    messages = react_graph.invoke({"messages": messages,"input_file":"/home/futureai/code/agents-cookbook/langgraph/date.png"})
    for m in messages['messages']:
        m.pretty_print()

if __name__ == "__main__":
    image_handle()