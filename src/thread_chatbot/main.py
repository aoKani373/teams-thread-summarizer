import os
import html
import json
import pandas as pd
import streamlit as st
from openai import OpenAI
import plotly.express as px
from datetime import timedelta
from dotenv import load_dotenv, find_dotenv
from prompt import TEMPLATE

load_dotenv(find_dotenv())

LLM_NAME = os.getenv("LLM_NAME", "gpt-5-nano")
RERANKER_NAME = os.getenv("RERANKER_NAME", "cl-nagoya/ruri-v3-reranker-310m")

import torch
from sentence_transformers import CrossEncoder


@st.cache_resource
def load_reranker():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return CrossEncoder(
        RERANKER_NAME, device=device, max_length=512
    )

client = OpenAI()
reranker = load_reranker()

def stream(stream):
    for event in stream:
        if event.type == 'response.output_text.delta':
            yield event.delta
        elif event.type == 'response.output_text.done':
            if messages := st.session_state.get("messages"):
                messages.append({"role": "assistant", "content": event.text})

# """
# - id: メッセージID
# - body_content: メッセージ本文
# - from_user_displayName: 送信者
# - createdDateTime: 送信日時

# - body_plain: メッセージ本文（プレーンテキスト）
# - createdDateTime_jst: 送信日時（日本時間）
# - rerank_score: 質問とメッセージの類似度
# """

def show_gpu_stats():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3) # GB
        reserved = torch.cuda.memory_reserved() / (1024 ** 3) # GB
        
        total_vram: float = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) # GB

        st.sidebar.subheader("GPU Memory Monitor")
        st.sidebar.info(f"Device: {torch.cuda.get_device_name(0)}")
        
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Allocated", f"{allocated:.2f} GB")
        col2.metric("Reserved", f"{reserved:.2f} GB")
        
        usage_pct = reserved / total_vram
        st.sidebar.progress(usage_pct, text=f"VRAM Usage ({reserved:.1f}/{total_vram:.1f} GB)")
    else:
        st.sidebar.warning("CUDA is not available. Running on CPU.")

def show_timeline(df: pd.DataFrame, step_delta = timedelta(minutes=1)):
    min_date = df["createdDateTime_jst"].min().to_pydatetime()
    max_date = df["createdDateTime_jst"].max().to_pydatetime()

    start_date, end_date = st.slider(
        "フィルタ範囲を選択してください",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        step=step_delta,
        format="MM/DD HH:mm"
    )

    df_filtered = df[(df["createdDateTime_jst"] >= start_date) & (df["createdDateTime_jst"] <= end_date)]
    st.write("Record count:", len(df_filtered))

    fig = px.scatter(
        df,
        x="createdDateTime_jst",
        y="from_user_displayName",
        color="from_user_displayName",
        hover_data=["body_plain"],
        title="メッセージ送信タイムライン",
        labels={"createdDateTime_jst": "送信日時", "from_user_displayName": "送信者"}
    )

    fig.add_vline(x=pd.Timestamp(start_date).timestamp() * 1000, line_width=2, line_dash="dash", line_color="red")
    fig.add_vline(x=pd.Timestamp(end_date).timestamp() * 1000, line_width=2, line_dash="dash", line_color="red")

    fig.add_vrect(x0=min_date, x1=start_date, fillcolor="gray", opacity=0.2, line_width=0)
    fig.add_vrect(x0=end_date, x1=max_date, fillcolor="gray", opacity=0.2, line_width=0)

    fig.update_traces(marker=dict(size=12, opacity=0.7))
    fig.update_layout(yaxis={"categoryorder": "total ascending"})

    st.plotly_chart(fig, width="stretch")

    return df_filtered

def intro():
    show_gpu_stats()

    if uploaded_file := st.file_uploader("Upload Teams messages file", type="json"):
        data = json.load(uploaded_file)
        messages = [data["parentMessage"]] + data["replies"]

        del data

        df = pd.json_normalize(messages, sep="_").sort_values("createdDateTime", ascending=True)
        
        df["createdDateTime_jst"] = pd.to_datetime(df["createdDateTime"]).dt.tz_convert("Asia/Tokyo")
        df['body_plain'] = df['body_content'].str.replace(r'<[^>]*>', '', regex=True).apply(html.unescape)

        df_cleaned = df.dropna(axis=1, how="any")

        st.write(f"表示中の列数: {len(df_cleaned.columns)}（全 {len(df.columns)}列中）")

        st.dataframe(df_cleaned, width="stretch")

        # ---
        st.divider()
        df_filtered: pd.DataFrame = show_timeline(df)

        # ---
        st.divider()
        col1, col2 = st.columns(2)

        selection = col1.segmented_control(
            "Action", ["Rerank", "Generate"], selection_mode="multi", default="Rerank"
        )
        top_k = col2.number_input(
            "Top-K", value=5, min_value=1, max_value=20
        )

        if prompt := st.chat_input("Type your message here..."):
            st.chat_message("user").write(prompt)

            if "Rerank" in selection:
                messages_reranked = reranker.rank(
                    query=prompt, 
                    documents=df_filtered["body_plain"].to_list(), 
                    top_k=top_k, 
                    return_documents=False
                )
                df_results = pd.DataFrame(messages_reranked)
                df_top_k: pd.DataFrame = df_filtered.iloc[df_results["corpus_id"]].copy()
                df_top_k["rerank_score"] = df_results["score"].values
            else:
                df_top_k: pd.DataFrame = df_filtered.tail(top_k)
            
            for _, row in df_top_k.iterrows():
                with st.chat_message(row["from_user_displayName"]):
                    badges_text = f":green-badge[Date: {row["createdDateTime_jst"]}] :gray-badge[Sender: {row["from_user_displayName"]}]"
                    if "rerank_score" in row:
                        badges_text = f":blue-badge[Score: {row["rerank_score"]:.4f}] " + badges_text

                    st.markdown(badges_text)
                    st.html(row["body_content"])

            if "Generate" in selection:
                final_string = (
                    "[" + df_top_k["createdDateTime_jst"].dt.strftime("%Y-%m-%d %H:%M").astype(str) + "]" +
                    df_top_k["from_user_displayName"] + ": " + 
                    df_top_k["body_plain"]
                ).str.cat(sep="\n")

                prompt_template: str = TEMPLATE["generate"]

                query = prompt_template.format(
                    user_query=prompt, teams_messages=final_string
                )

                response = client.responses.create(
                    model=LLM_NAME, input=query, stream=True
                )

                with st.chat_message("assistant"):
                    st.write_stream(stream(response))

intro()