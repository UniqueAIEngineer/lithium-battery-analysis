import streamlit as st
import pandas as pd
import joblib

model, FEATS, P = joblib.load("queen.joblib")

st.title("バッテリ劣化判定ツール v6")
st.caption("内部抵抗（Re・Rct）から劣化を判定します。容量は使いません")

re_first = st.number_input("最初のRe（Ω）",value=0.050,format="%.4f")
re_now = st.number_input("今のRe（Ω）",value=0.055,format="%.4f")
rct = st.number_input("今のRct（Ω）",value=0.080,format="%.4f")

if st.button("判定"):
    re_ratio = re_now/re_first
    x = pd.DataFrame([[re_ratio, rct]], columns=FEATS)
    prob = model.predict_proba(x)[0,1]

    st.write("Reの倍率:", round(re_ratio, 3))
    st.write("劣化の確率:", round(prob, 2))
    st.progress(float(prob))

    if prob >= P:
        st.error("劣化の可能性あり")
    else:
        st.success("正常です")
