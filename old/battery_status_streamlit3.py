import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

def get_status_info(prediction_result):
    normal_prob = float(prediction_result['正常確率'])
    degraded_prob = float(prediction_result['劣化確率'])
    
    # 確信度レベルの詳細な判定
    if max(normal_prob, degraded_prob) > 0.9:
        confidence = '非常に高い確信度'
        confidence_color = 'blue'
    elif max(normal_prob, degraded_prob) > 0.7:
        confidence = '高い確信度'
        confidence_color = 'green'
    elif max(normal_prob, degraded_prob) > 0.6:
        confidence = '中程度の確信度'
        confidence_color = 'orange'
    else:
        confidence = '要追加調査'
        confidence_color = 'red'
    
    # 状態の判定
    if normal_prob > 0.6:
        status = '正常です'
        icon = '✅'
        color = 'green'
    elif degraded_prob > 0.7:
        status = '劣化の可能性があります'
        icon = '⚠️'
        color = 'red'
    else:
        status = '注意が必要です'
        icon = '⚜️'
        color = 'orange'
    
    return {
        'status': status,
        'icon': icon,
        'color': color,
        'confidence': confidence,
        'confidence_color': confidence_color
    }

def display_prediction_result(row_num, data, prediction_result):
    with st.container():
        st.markdown(f"### 行 {row_num} の結果")
        
        # 状態情報の取得
        status_info = get_status_info(prediction_result)
        
        # ステータスと確信度
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(
                f"<h4 style='color: {status_info['color']}'>{status_info['icon']} {status_info['status']}</h4>", 
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"<p style='color: {status_info['confidence_color']}'>確信度: {status_info['confidence']}</p>",
                unsafe_allow_html=True
            )
        
        # 確率バー
        st.markdown("#### 確率分布")
        col3, col4 = st.columns(2)
        with col3:
            st.progress(float(prediction_result['正常確率']))
            st.markdown(f"正常: {prediction_result['正常確率']}")
        with col4:
            st.progress(float(prediction_result['劣化確率']))
            st.markdown(f"劣化: {prediction_result['劣化確率']}")
        
        # 詳細データと判定
        st.markdown("#### 測定データと判定")
        temp = data['Temperature']
        voltage = data['Voltage']
        capacity = data['Capacity']
        
        # 各パラメータの判定
        temp_status = "🟢 正常" if 20 <= temp <= 30 else "🔴 注意"
        voltage_status = "🟢 正常" if voltage >= 3.5 else "🔴 注意"
        capacity_status = "🟢 十分" if capacity > 2000 else "🔴 低下"
        
        st.markdown(f"🌡 温度: {temp:.1f}℃ ({temp_status})")
        st.markdown(f"⚡ 電圧: {voltage:.2f}V ({voltage_status})")
        st.markdown(f"🔋 容量: {capacity:.0f}mAh ({capacity_status})")
        
        # 判定基準の説明
        with st.expander("判定基準の説明"):
            st.markdown("""
            - 温度: 20-30℃が正常範囲
            - 電圧: 3.5V以上が正常範囲
            - 容量: 2000mAh以上が十分
            - 確信度: 
                - 90%以上: 非常に高い
                - 70-90%: 高い
                - 60-70%: 中程度
                - 60%未満: 追加調査推奨
            """)
        
        st.markdown("---")

def main():
    st.title('バッテリー劣化予測ツール v5')
    st.caption('※注意：このバージョンはCSVファイル選択と予測が可能です。')
    
    # ファイル選択機能の追加
    st.sidebar.header('データファイル選択')
    
    # アップロードかデフォルトファイルかの選択
    file_option = st.sidebar.radio(
        "データソースの選択:",
        ['デフォルトファイル (battery_data2.csv)', 'ファイルをアップロード']
    )
    
    try:
        if file_option == 'ファイルをアップロード':
            uploaded_file = st.sidebar.file_uploader("CSVファイルを選択", type=['csv'])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success('ファイルのアップロードに成功しました')
            else:
                st.sidebar.warning('ファイルをアップロードしてください')
                return
        else:
            # デフォルトファイルの読み込み
            default_file = Path('battery_data2.csv')
            if default_file.exists():
                df = pd.read_csv(default_file)
                st.sidebar.success('デフォルトファイルの読み込みに成功しました')
            else:
                st.sidebar.error('デフォルトファイルが見つかりません')
                return
        
        # データの基本情報表示
        st.sidebar.markdown("### データ概要")
        st.sidebar.markdown(f"総レコード数: {len(df)}")
        
        # モデルの読み込み
        model_path = Path('model_data.joblib')
        if not model_path.exists():
            st.error('予測モデルファイルが見つかりません')
            return
            
        model_tuple = joblib.load(model_path)
        model = model_tuple[0]
        
        # 複数行選択
        st.write('行を選択:')
        selected_rows = st.multiselect('', options=df.index, format_func=lambda x: f'Row {x}')
        
        if st.button('予測'):
            if not selected_rows:
                st.warning('行を選択してください。')
                return
            
            for row in selected_rows:
                data = df.iloc[row]
                prediction = model.predict_proba([data])[0]
                
                result = {
                    '予測結果': '正常です' if prediction[0] > prediction[1] else '劣化の可能性があります',
                    '正常確率': f'{prediction[0]:.2f}',
                    '劣化確率': f'{prediction[1]:.2f}',
                }
                
                display_prediction_result(row, data, result)
                
    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
        st.error("データの読み込みまたは処理中にエラーが発生しました。ファイル形式を確認してください。")
        
if __name__ == "__main__":
    main()