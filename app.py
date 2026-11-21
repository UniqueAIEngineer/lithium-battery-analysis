import streamlit as st
import pandas as pd
import joblib

model, FEATS, P = joblib.load("queen.joblib")

st.title("バッテリ劣化判定ツール v6")
st.caption("内部抵抗（Re・Rct）から劣化を判定します。容量は使いません")
st.caption("学習データの範囲：Re 0.036〜0.107 Ω、Rct 0.060〜0.139 Ω（範囲外の値は判定の信頼性が下がります）")

re_first = st.number_input("最初のRe（Ω）",min_value=0.001,value=0.050,format="%.4f")
re_now = st.number_input("今のRe（Ω）",min_value=0.001,value=0.055,format="%.4f")
rct = st.number_input("今のRct（Ω）",min_value=0.001,value=0.083,format="%.4f")

if st.button("判定"):
    re_ratio = re_now/re_first
    x = pd.DataFrame([[re_ratio, rct]], columns=FEATS)
    prob = model.predict_proba(x)[0,1]

    st.write("Reの倍率:", round(re_ratio, 3))
    st.write("劣化の確率:", round(prob, 2))
    st.progress(float(prob))

    conf = max(prob, 1 - prob)
    if conf > 0.9:
        st.write("確信度：非常に高い")
    elif conf > 0.7:
        st.write("確信度：高い")
    elif conf > 0.6:
        st.write("確信度：中程度")
    else:
        st.write("確信度：要追加調査")
       
    if prob >= P:
        st.error("劣化の可能性あり")
    else:
        st.success("正常です")

with st.expander("判定基準の説明"):
    st.markdown("""
- 劣化の確率が 0.2 以上なら「劣化の可能性あり」
- 線を 0.2 と低めにしているのは、劣化の見逃しを減らすため
- 確信度：90%以上 非常に高い／70%以上 高い／60%以上 中程度／それ未満 要追加調査
""")