from fastapi import FastAPI
from llama_cpp import Llama
import uvicorn
from pydantic import BaseModel
from typing import Dict, List
import time

# 存储多会话对话历史
conversation_histories: Dict[str, List[dict]] = {}

class ChatRequest(BaseModel):
    user_input: str
    session_id: str = "default"  # 默认会话ID

app = FastAPI()

# 模型配置
llm = Llama(
    model_path="../models/MentaLLaMA-chat-7B.Q8_0.gguf",
    n_ctx=1024,        # 上下文长度
    n_gpu_layers=33,
    n_threads=6,       # CPU线程数
    n_batch=256,       # 批处理优化
    use_mlock=True     # 内存锁定
)

@app.post("/chat")
async def chat(request: ChatRequest):
    # 初始化或获取对话历史
    if request.session_id not in conversation_histories:
        conversation_histories[request.session_id] = []

    history = conversation_histories[request.session_id]

    # 添加系统消息：要求直接回答，且不使用任何编号或分点列举
    if not history:  # 仅在会话开始时添加系统消息
        history.append({
            "role": "system",
            "content": (
                "You are a professional psychological counselor. Respond in a friendly, patient, and understanding manner. "
                "Keep your answer concise and direct without repeating the user's question. Do not use numbered lists, bullet points, "
                "or separate suggestions. Provide your answer in a single, continuous paragraph without enumerations."
            )
        })

    # 添加用户消息（带时间戳）
    history.append({
        "role": "user",
        "content": request.user_input,
        "timestamp": str(time.time())
    })

    # 只保留最近10条对话历史
    if len(history) > 10:
        history = history[-10:]

    try:
        # 调用模型生成回复
        response = llm.create_chat_completion(
            messages=history,
            temperature=0.2,
            max_tokens=50,
            stop=["\n###"]
        )

        ai_message = response['choices'][0]['message']
        ai_content = ai_message['content']

        # 将AI回复添加到历史记录中
        history.append({
            "role": "assistant",
            "content": ai_content,
            "timestamp": str(time.time())
        })

        return {
            "response": ai_content,
            "session_id": request.session_id,
            "history_length": len(history)
        }

    except Exception as e:
        return {
            "error": str(e),
            "detail": "生成回复时发生错误"
        }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=600
    )
