from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings  # 使用本地嵌入模型
import os
import logging
from transformers import AutoModel, AutoTokenizer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def qa_agent(api_key, memory, uploaded_file, question):
    try:
        # 初始化聊天模型
        model = ChatOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
            temperature=0.3
        )

        # 处理PDF文件
        file_content = uploaded_file.read()
        temp_file_path = 'temp.pdf'
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(file_content)

        # 加载PDF
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

        # 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=['\n\n', '。', '！', '？', '，', '、', '']
        )
        texts = text_splitter.split_documents(docs)

        # # 使用本地嵌入模型 (避免API问题)
        # embeddings_model = HuggingFaceEmbeddings(
        #     model_name="BAAI/bge-large-zh-v1.5",
        #     model_kwargs={'device': 'cpu'},
        #     encode_kwargs={'normalize_embeddings': True}
        # )
        embeddings_model = HuggingFaceEmbeddings(
            model_name="text2vec-base-local",
            model_kwargs={'device': 'cpu',"local_files_only": True},
            encode_kwargs={'normalize_embeddings': True}
        )




        # 创建向量数据库
        db = FAISS.from_documents(texts, embeddings_model)
        retriever = db.as_retriever(search_kwargs={"k": 3})

        # 创建问答链 - 确保正确设置输出键
        qa = ConversationalRetrievalChain.from_llm(
            llm=model,
            retriever=retriever,
            memory=memory,
            return_source_documents=False,
            output_key="answer"  # 明确设置输出键
        )

        # 执行查询
        response = qa({"question": question})

        # 确保响应包含所有必要字段
        if 'chat_history' not in response:
            response['chat_history'] = memory.chat_history.messages

        return response

    except Exception as e:
        logger.error(f"处理过程中出错: {str(e)}")
        return {
            "answer": f"处理问题时出错: {str(e)}",
            "chat_history": memory.chat_history.messages if hasattr(memory, 'chat_history') else []
        }

    finally:
        # 确保删除临时文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)