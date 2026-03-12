import streamlit as st
import pandas as pd
import numpy as np

# ==========================================
# データ処理関数 (変更なし)
# ==========================================
def func1_fft(df, max_freq_mhz=None, pad_factor=1):
    time = df.iloc[:, 0].values
    pressure = df.iloc[:, 1].values
    
    dt = time[1] - time[0]
    
    N_orig = len(pressure)
    N_padded = N_orig * pad_factor
    
    F = np.fft.rfft(pressure, n=N_padded)
    freqs_mhz = np.fft.rfftfreq(N_padded, d=dt) / 1e6
    
    if max_freq_mhz is not None:
        F[freqs_mhz > max_freq_mhz] = 0
        
    out_df = pd.DataFrame({
        'Frequency_MHz': freqs_mhz,
        'Real': np.real(F),
        'Imaginary': np.imag(F),
        'Amplitude': np.abs(F)
    })
    return out_df

def func2_fft_filter(df, filter_type='cut_below', cut_freq_mhz=1.0, center_freq_mhz=5.0, bandwidth_mhz=2.0, max_freq_mhz=None, pad_factor=1):
    time = df.iloc[:, 0].values
    pressure = df.iloc[:, 1].values
    
    dt = time[1] - time[0]
    
    N_orig = len(pressure)
    N_padded = N_orig * pad_factor
    
    F = np.fft.rfft(pressure, n=N_padded)
    freqs_mhz = np.fft.rfftfreq(N_padded, d=dt) / 1e6
    
    if filter_type == 'cut_below':
        F[freqs_mhz <= cut_freq_mhz] = 0
    elif filter_type == 'cut_above':
        F[freqs_mhz >= cut_freq_mhz] = 0
    elif filter_type == 'bandpass':
        lower_bound = center_freq_mhz - (bandwidth_mhz / 2.0)
        upper_bound = center_freq_mhz + (bandwidth_mhz / 2.0)
        F[(freqs_mhz < lower_bound) | (freqs_mhz > upper_bound)] = 0
        
    if max_freq_mhz is not None:
        F[freqs_mhz > max_freq_mhz] = 0
        
    out_df = pd.DataFrame({
        'Frequency_MHz': freqs_mhz,
        'Real': np.real(F),
        'Imaginary': np.imag(F),
        'Amplitude': np.abs(F)
    })
    return out_df

def func3_ifft(df):
    freqs_mhz = df['Frequency_MHz'].values
    real = df['Real'].values
    imag = df['Imaginary'].values
    
    F = real + 1j * imag
    pressure_reconstructed = np.fft.irfft(F)
    
    N_time = len(pressure_reconstructed)
    max_freq_mhz = freqs_mhz[-1]
    max_freq_hz = max_freq_mhz * 1e6
    dt = 1.0 / (2.0 * max_freq_hz)
    
    time_reconstructed_s = np.arange(0, N_time) * dt
    time_reconstructed_us = time_reconstructed_s * 1e6
    
    out_df = pd.DataFrame({
        'Time_us': time_reconstructed_us,
        'Pressure': pressure_reconstructed
    })
    return out_df

# ==========================================
# Streamlit UI構築
# ==========================================
st.title("音圧波形 FFT/逆FFT 解析ツール")
st.write("CSV形式のデータをアップロードして、処理を実行します。")

if 'current_feature' not in st.session_state:
    st.session_state.current_feature = "機能1: 任意範囲のFFT"

st.sidebar.header("機能選択")
feature = st.sidebar.radio(
    "実行する処理を選んでください",
    ("機能1: 任意範囲のFFT", "機能2: 指定周波数カット ＋ FFT", "機能3: 逆FFT (波形復元)"),
    key="feature_radio"
)

if st.session_state.current_feature != feature:
    st.session_state.current_feature = feature
    if 'result_df' in st.session_state:
        del st.session_state['result_df']

pad_options = {"1倍 (そのまま)": 1, "2倍": 2, "3倍": 3, "5倍": 5, "10倍": 10}

uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=["csv"], key="uploader")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    if feature in ["機能1: 任意範囲のFFT", "機能2: 指定周波数カット ＋ FFT"]:
        auto_dt = df.iloc[1, 0] - df.iloc[0, 0]
        st.info(f"💡 CSVから読み取ったサンプリング間隔 (dt): {auto_dt:.2e} 秒")
        st.markdown("---")

    # ==========================================
    # 機能1: 任意範囲のFFT
    # ==========================================
    if feature == "機能1: 任意範囲のFFT":
        st.header("機能1: 任意範囲のFFT")
        
        col1, col2 = st.columns(2)
        with col1:
            max_freq = st.number_input("抽出・表示する最大周波数 (MHz) ※0で制限なし", min_value=0.0, value=10.0, step=0.1, key="max_freq1")
        with col2:
            pad_choice = st.selectbox("周波数分解能の倍率 (ゼロパディング)", options=list(pad_options.keys()), key="pad1")
            pad_factor = pad_options[pad_choice]
            
        if st.button("FFTを実行", key="btn_run1"):
            with st.spinner('処理中...'):
                limit = max_freq if max_freq > 0 else None
                st.session_state['result_df'] = func1_fft(df, max_freq_mhz=limit, pad_factor=pad_factor)
                st.session_state['limit'] = limit
                st.success("処理が完了しました！")

        if 'result_df' in st.session_state:
            out_df = st.session_state['result_df']
            limit = st.session_state.get('limit', None)
            plot_df = out_df[out_df['Frequency_MHz'] <= limit] if limit is not None else out_df
            st.line_chart(plot_df.set_index('Frequency_MHz')['Amplitude'])
            csv = out_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="out1.csv をダウンロード", data=csv, file_name='out1.csv', mime='text/csv')

    # ==========================================
    # 機能2: 周波数カットFFT
    # ==========================================
    elif feature == "機能2: 指定周波数カット ＋ FFT":
        st.header("機能2: 指定周波数カット ＋ FFT")
        
        st.markdown("### ① カットするフィルターの指定")
        filter_label = st.radio(
            "フィルターの種類",
            ("指定MHz【以下】をカット (ハイパス)", "指定MHz【以上】をカット (ローパス)", "【中心周波数】を指定して抽出 (バンドパス)"),
            key="filter_type_radio"
        )
        
        cut_freq = 1.0
        center_freq = 5.0
        bandwidth = 2.0
        
        if filter_label == "指定MHz【以下】をカット (ハイパス)":
            filter_type = 'cut_below'
            cut_freq = st.number_input("カットする基準の周波数 (MHz)", min_value=0.0, value=1.0, step=0.1)
        elif filter_label == "指定MHz【以上】をカット (ローパス)":
            filter_type = 'cut_above'
            cut_freq = st.number_input("カットする基準の周波数 (MHz)", min_value=0.0, value=10.0, step=0.1)
        else:
            filter_type = 'bandpass'
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                center_freq = st.number_input("中心周波数 (MHz)", min_value=0.1, value=5.0, step=0.1)
            with col_b2:
                bandwidth = st.number_input("抽出する帯域幅 (MHz)", min_value=0.1, value=2.0, step=0.1)
            st.caption(f"※ {center_freq - bandwidth/2:.2f} MHz 〜 {center_freq + bandwidth/2:.2f} MHz の範囲のみを残します。")
        
        st.markdown("---")
        st.markdown("### ② 最終的に抽出・表示するFFTの範囲")
        col1, col2 = st.columns(2)
        with col1:
            max_freq = st.number_input("結果に残す最大周波数 (MHz) ※0で全範囲", min_value=0.0, value=10.0, step=0.1, key="max_freq2")
        with col2:
            pad_choice = st.selectbox("周波数分解能の倍率 (ゼロパディング)", options=list(pad_options.keys()), key="pad2")
            pad_factor = pad_options[pad_choice]
        
        if st.button("FFTを実行", key="btn_run2"):
            with st.spinner('処理中...'):
                limit = max_freq if max_freq > 0 else None
                st.session_state['result_df'] = func2_fft_filter(
                    df, filter_type=filter_type, cut_freq_mhz=cut_freq, 
                    center_freq_mhz=center_freq, bandwidth_mhz=bandwidth, 
                    max_freq_mhz=limit, pad_factor=pad_factor
                )
                st.session_state['limit'] = limit
                st.success("処理が完了しました！")

        if 'result_df' in st.session_state:
            out_df = st.session_state['result_df']
            limit = st.session_state.get('limit', None)
            plot_df = out_df[out_df['Frequency_MHz'] <= limit] if limit is not None else out_df
            st.line_chart(plot_df.set_index('Frequency_MHz')['Amplitude'])
            csv = out_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="out2.csv をダウンロード", data=csv, file_name='out2.csv', mime='text/csv')

    # ==========================================
    # 機能3: 逆FFT
    # ==========================================
    elif feature == "機能3: 逆FFT (波形復元)":
        st.header("機能3: 逆FFT (波形復元)")
        st.info("💡 機能1または機能2でダウンロードしたCSV (out1.csv, out2.csv) をアップロードしてください。")
        
        if {'Frequency_MHz', 'Real', 'Imaginary'}.issubset(df.columns):
            
            # 【追加】表示する最大時間 (μsec) の入力欄
            max_time = st.number_input("表示する最大時間 (μsec) ※0で全範囲", min_value=0.0, value=5.0, step=0.1, key="max_time3")
            
            if st.button("逆FFTを実行", key="btn_run3"):
                with st.spinner('波形を復元中...'):
                    st.session_state['result_df'] = func3_ifft(df)
                    st.session_state['max_time'] = max_time if max_time > 0 else None
                    st.success("波形の復元が完了しました！")
                    
            if 'result_df' in st.session_state:
                out_df = st.session_state['result_df']
                limit_time = st.session_state.get('max_time', None)
                
                # 【追加】グラフ表示用に指定された時間でデータを切り出す
                plot_df = out_df[out_df['Time_us'] <= limit_time] if limit_time is not None else out_df
                
                st.line_chart(plot_df.set_index('Time_us')['Pressure'])
                
                csv = out_df.to_csv(index=False).encode('utf-8')
                st.download_button(label="out3.csv をダウンロード", data=csv, file_name='out3.csv', mime='text/csv')
        else:
            st.error("【エラー】アップロードされたファイルが異なります。")