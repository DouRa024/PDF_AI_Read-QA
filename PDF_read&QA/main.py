import streamlit as st
from langchain.memory import ConversationBufferMemory
from utils import qa_agent

st.title('📑爱吃PDF，拉出回答给你💩')

with st.sidebar:
    api_key = st.text_input("请输入DeepSeek API密钥：", type="password")
    st.markdown("[获取DeepSeek API key](https://platform.deepseek.com/)")

# 初始化会话状态
if 'memory' not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key='chat_history',
        output_key='answer'
    )

if 'chat_history' not in st.session_state:
    st.session_state["chat_history"] = []

uploaded_file = st.file_uploader("<给我吃好吃的pdf：", type="pdf")
question = st.text_input("提问我任何关于pdf 的问题吧", disabled=not uploaded_file)

if uploaded_file and question and not api_key:
    st.info("没有api_key,想白嫖啊？")

if uploaded_file and question and api_key:
    with st.spinner("别急，在想"):
        response = qa_agent(api_key, st.session_state["memory"], uploaded_file, question)

    # 处理响应 - 使用更健壮的方式获取答案
    if 'answer' in response:
        answer = response['answer']
    elif 'text' in response:  # 尝试其他可能的键
        answer = response['text']
    elif 'result' in response:
        answer = response['result']
    else:
        st.error(f"无法获取答案，响应结构: {list(response.keys())}")
        answer = "抱歉，无法获取答案"

    st.write('### 有的，兄弟有的')
    st.write(answer)

    # 更新聊天历史
    if 'chat_history' in response:
        st.session_state["chat_history"] = response['chat_history']
    else:
        # 手动更新聊天历史
        st.session_state["chat_history"].append({"role": "user", "content": question})
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})

# 显示聊天历史
if 'chat_history' in st.session_state and st.session_state["chat_history"]:
    with st.expander("想知道之前你问过什么吗"):
        for i in range(0, len(st.session_state["chat_history"]), 2):
            if i < len(st.session_state["chat_history"]):
                human_message = st.session_state["chat_history"][i]
                st.write(f"**你:** {human_message.content if hasattr(human_message, 'content') else human_message}")

            if i + 1 < len(st.session_state["chat_history"]):
                ai_message = st.session_state["chat_history"][i + 1]
                st.write(f"**AI:** {ai_message.content if hasattr(ai_message, 'content') else ai_message}")

            if i < len(st.session_state["chat_history"]) - 2:
                st.divider()