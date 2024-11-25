from langchain_community.vectorstores import FAISS
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document
import pandas as pd
import os
import openai
from openai import RateLimitError

# OpenAI APIキーの設定
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY 環境変数が設定されていません。")
os.environ["OPENAI_API_KEY"] = openai_api_key  # 必要に応じて環境変数を設定

# 1. CSVファイルの読み込みと前処理
csv_path = "./data/rise_slot.csv"  # 読み込むCSVファイルのパス
df = pd.read_csv(csv_path, encoding="utf-8")  # データフレームとして読み込む

# 各行を独立したドキュメントとして扱う
# 各行が1つの文書としてリストに格納
documents = []
for idx, row in df.iterrows():
    doc_text = " ".join(row.astype(str))  # 行内の全ての列を文字列として結合
    documents.append(Document(page_content=doc_text))  # Document型に変換して追加

# テキストの分割 - 必要に応じて文書を分割（例: 1つの文書が大きすぎる場合）
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=50)  # chunk_sizeを小さく設定
texts = text_splitter.split_documents(documents)

# ベクターDBの作成
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)

# モデルとトークン数制限の設定
llm = ChatOpenAI(model="gpt-4", temperature=0, max_tokens=550)  # 出力トークンを100に設定

# プロンプトの定義
template = """
以下の情報を参考にして質問に答えてください。分からないことには答えないでください。

文脈:
{context}

質問:
{question}

回答:
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# 6. 関連ドキュメント数の制限（k = 関連する情報の数）
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# プロンプトとチェーンの設定
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# 質問の処理とエラーハンドリング
query = "上位AT「ツラヌキ」にはどのように突入するのですか？"
try:
    # invoke メソッドを使用して結果を取得
    result = qa_chain.invoke({"query": query})

    # 結果から必要な情報を抽出
    print(f"質問:{query}",)
    print("\n回答:", result['result'])
    print("\n関連するソースドキュメント:")
    for doc in result['source_documents']:
        print(doc.page_content)

except RateLimitError as e:
    print(f"Rate limit error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
