import streamlit as st
import pandas as pd
import numpy as np

# ==========================================
# データ処理関数
# ==========================================
def func1_fft(df, max_freq_mhz=None, pad_factor=1):
    time = df.iloc[:, 0].values
    pressure = df.iloc[:, 1].values
    dt = time[1] - time[0]
    N_padded = len(pressure) * pad_factor
    F = np.fft.rfft(pressure, n=N_padded)
    freqs_mhz = np.fft.rfftfreq(N_padded, d=dt) / 1e6
    if max_freq_mhz is not None:
        F[freqs_mhz > max_freq_mhz] = 0
    return pd.DataFrame({'Frequency_MHz': freqs_mhz, 'Real': np.real(F), 'Imaginary': np.imag(F), 'Amplitude': np.abs(F)})

def func2_fft_filter(df, filter_type='cut_below', cut_freq_mhz=1.0, center_freq_mhz=5.0, bandwidth_mhz=2.0, max_freq_mhz=None, pad_factor=1):
    time = df.iloc[:, 0].values
    pressure = df.iloc[:, 1].values
    dt = time[1] - time[0]
    N_padded = len(pressure) * pad_factor
    F = np.fft.rfft(pressure, n=N_padded)
    freqs_mhz = np.fft.rfftfreq(N_padded, d=dt) / 1e6
    
    if filter_type == 'cut_below':
        F[freqs_mhz <= cut_freq_mhz] = 0
    elif filter_type == 'cut_above':
        F[freqs_mhz >= cut_freq_mhz] = 0
    elif filter_type == 'bandpass':
        lower = center_freq_mhz - (bandwidth_mhz / 2.0)
        upper = center_freq_mhz + (bandwidth_mhz / 2.0)
        F[(freqs_mhz < lower) | (freqs_mhz > upper)] = 0
        
    if max_freq_mhz is not None:
        F[freqs_mhz > max_freq_mhz] = 0
    return pd.DataFrame({'Frequency_MHz': freqs_mhz, 'Real': np.real(F), 'Imaginary': np.imag(F), 'Amplitude': np.abs(F)})

def func3_ifft(df):
    freqs_mhz = df['Frequency_MHz'].values
    F = df['Real'].values + 1j * df['Imaginary'].values
    pressure_reconstructed = np.fft.irfft(F)
    dt = 1.0 / (2.0 * freqs_mhz[-1] * 1e6)
    time_us = np.arange(0, len(pressure_reconstructed)) * dt * 1e6
    return pd.DataFrame({'Time_us': time_us, 'Pressure': pressure_reconstructed})

# 新規追加：機能4（滑らかなフィルター）
def func4_fft_smooth_filter(df, filter_target='lowpass', cut_freq_mhz=1.0, transition=1.0, window_type='hann', max_freq_mhz=None, pad_factor=1):
    time = df.iloc[:, 0].values
    pressure = df.iloc[:, 1].values
    dt = time[1] - time[0]
    N_padded = len(pressure) * pad_factor
    F = np.fft.rfft(pressure, n=N_padded)
    freqs_mhz = np.fft.rfftfreq(N_padded, d=dt) / 1e6

    # 乗数（減衰率）を1.0で初期化
    multiplier = np.ones_like(freqs_mhz, dtype=float)

    if window_type == 'hann':
        if filter_target == 'lowpass': # 指定MHz以上を徐々にカット
            idx = (freqs_mhz > cut_freq_mhz) & (freqs_mhz < cut_freq_mhz + transition)
            multiplier[idx] = 0.5 * (1 + np.cos(np.pi * (freqs_mhz[idx] - cut_freq_mhz) / transition))
            multiplier[freqs_mhz >= cut_freq_mhz + transition] = 0
        else: # highpass: 指定MHz以下を徐々にカット
            idx = (freqs_mhz > cut_freq_mhz - transition) & (freqs_mhz < cut_freq_mhz)
            multiplier[idx] = 0.5 * (1 - np.cos(np.pi * (freqs_mhz[idx] - (cut_freq_mhz - transition)) / transition))
            multiplier[freqs_mhz <= cut_freq_mhz - transition] = 0

    elif window_type == 'butterworth':
        order = transition # パラメータを次数として扱う
        if filter_target == 'lowpass':
            multiplier = 1.0 / np.sqrt(1.0 + (freqs_mhz / cut_freq_mhz)**(2 * order))
        else:
            safe_freqs = np.where(freqs_mhz == 0, 1e-10, freqs_mhz) # 0割り回避
            multiplier = 1.0 / np.sqrt(1.0 + (cut_freq_mhz / safe_freqs)**(2 * order))
            multiplier[freqs_mhz == 0] = 0

    elif window_type == 'gaussian':
        if filter_target == 'lowpass':
            idx = freqs_mhz > cut_freq_mhz
            multiplier[idx] = np.exp(-0.5 * ((freqs_mhz[idx] - cut_freq_mhz) / transition)**2)
        else:
            idx = freqs_mhz < cut_freq_mhz
            multiplier[idx] = np.exp(-0.5 * ((freqs_mhz[idx] - cut_freq_mhz) / transition)**2)

    F_filtered = F * multiplier

    if max_freq_mhz is not None:
        F_filtered[freqs_mhz > max_freq_mhz] = 0

    return pd.DataFrame({'Frequency_MHz': freqs_mhz, 'Real': np.real(F_filtered), 'Imaginary': np.imag(F_filtered), 'Amplitude': np.abs(F_filtered)})


# ==========================================
# Streamlit UI構築
# ==========================================
st.title("音圧波形 FFT/逆FFT 解析ツール")

if 'current_feature' not in st.session_state:
    st.session_state.current_feature = "機能1: 任意範囲のFFT"

st.sidebar.header("機能選択")
feature = st.sidebar.radio(
    "実行する処理を選んでください",
    ("機能1: 任意範囲のFFT", "機能2: 指定周波数カット (急峻)", "機能3: 逆FFT (波形復元)", "機能4: 滑らかな周波数カット (窓関数)"),
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
    
    if feature in ["機能1: 任意範囲のFFT", "機能2: 指定周波数カット (急峻)", "機能4: 滑らかな周波数カット (窓関数)"]:
        auto_dt = df.iloc[1, 0] - df.iloc[0, 0]
        st.info(f"💡 CSVから読み取ったサンプリング間隔 (dt): {auto_dt:.2e} 秒")
        st.markdown("---")

    # ----- 機能1 -----
    if feature == "機能1: 任意範囲のFFT":
        col1, col2 = st.columns(2)
        with col1:
            max_freq = st.number_input("抽出・表示する最大周波数 (MHz) ※0で制限なし", min_value=0.0, value=10.0, step=0.1)
        with col2:
            pad_factor = pad_options[st.selectbox("分解能の倍率", options=list(pad_options.keys()))]
            
        if st.button("FFTを実行"):
            with st.spinner('処理中...'):
                limit = max_freq if max_freq > 0 else None
                st.session_state['result_df'] = func1_fft(df, max_freq_mhz=limit, pad_factor=pad_factor)
                st.session_state['limit'] = limit
                st.success("完了しました！")

        if 'result_df' in st.session_state:
            out_df = st.session_state['result_df']
            limit = st.session_state.get('limit', None)
            plot_df = out_df[out_df['Frequency_MHz'] <= limit] if limit is not None else out_df
            st.line_chart(plot_df.set_index('Frequency_MHz')['Amplitude'])
            st.download_button("out1.csv をダウンロード", data=out_df.to_csv(index=False).encode('utf-8'), file_name='out1.csv', mime='text/csv')

    # ----- 機能2 -----
    elif feature == "機能2: 指定周波数カット (急峻)":
        filter_type = 'cut_below' if st.radio("方法", ("指定MHz【以下】をカット", "指定MHz【以上】をカット")) == "指定MHz【以下】をカット" else 'cut_above'
        cut_freq = st.number_input("基準の周波数 (MHz)", min_value=0.0, value=1.0, step=0.1)
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            max_freq = st.number_input("結果に残す最大周波数 (MHz)", min_value=0.0, value=10.0, step=0.1)
        with col2:
            pad_factor = pad_options[st.selectbox("分解能の倍率", options=list(pad_options.keys()))]
        
        if st.button("FFTを実行"):
            with st.spinner('処理中...'):
                limit = max_freq if max_freq > 0 else None
                st.session_state['result_df'] = func2_fft_filter(df, filter_type=filter_type, cut_freq_mhz=cut_freq, max_freq_mhz=limit, pad_factor=pad_factor)
                st.session_state['limit'] = limit
                st.success("完了しました！")

        if 'result_df' in st.session_state:
            out_df = st.session_state['result_df']
            limit = st.session_state.get('limit', None)
            plot_df = out_df[out_df['Frequency_MHz'] <= limit] if limit is not None else out_df
            st.line_chart(plot_df.set_index('Frequency_MHz')['Amplitude'])
            st.download_button("out2.csv をダウンロード", data=out_df.to_csv(index=False).encode('utf-8'), file_name='out2.csv', mime='text/csv')

    # ----- 機能3 -----
    elif feature == "機能3: 逆FFT (波形復元)":
        if {'Frequency_MHz', 'Real', 'Imaginary'}.issubset(df.columns):
            max_time = st.number_input("表示する最大時間 (μsec) ※0で全範囲", min_value=0.0, value=5.0, step=0.1)
            
            if st.button("逆FFTを実行"):
                with st.spinner('復元中...'):
                    st.session_state['result_df'] = func3_ifft(df)
                    st.session_state['max_time'] = max_time if max_time > 0 else None
                    st.success("完了しました！")
                    
            if 'result_df' in st.session_state:
                out_df = st.session_state['result_df']
                limit_time = st.session_state.get('max_time', None)
                plot_df = out_df[out_df['Time_us'] <= limit_time] if limit_time is not None else out_df
                st.line_chart(plot_df.set_index('Time_us')['Pressure'])
                st.download_button("out3.csv をダウンロード", data=out_df.to_csv(index=False).encode('utf-8'), file_name='out3.csv', mime='text/csv')
        else:
            st.error("【エラー】機能1, 2, 4で出力したCSVをアップロードしてください。")

    # ----- 機能4 (新規) -----
    elif feature == "機能4: 滑らかな周波数カット (窓関数)":
        st.header("機能4: 滑らかな周波数カット (窓関数)")
        st.write("周波数を急にゼロにせず、滑らかに減衰させることで波形の歪み（リンギング）を防ぎます。")
        
        filter_target = 'lowpass' if st.radio("減衰の方向", ("指定MHz【以上】を徐々にカット", "指定MHz【以下】を徐々にカット")) == "指定MHz【以上】を徐々にカット" else 'highpass'
        cut_freq = st.number_input("減衰を開始(または終了)する基準周波数 (MHz)", min_value=0.0, value=5.0, step=0.1)
        
        window_choice = st.selectbox("窓関数（減衰カーブ）の種類", ("ハニングテーパー (直感的)", "バターワース型 (自然な減衰)", "ガウス型 (波打ち最小)"))
        
        if window_choice == "ハニングテーパー (直感的)":
            window_type = 'hann'
            transition = st.number_input("完全にゼロになるまでの帯域幅 (MHz)", min_value=0.1, value=2.0, step=0.1)
            st.caption(f"※ {cut_freq} MHz から始まり、{cut_freq + transition if filter_target=='lowpass' else cut_freq - transition} MHz で完全に 0 になります。")
        elif window_choice == "バターワース型 (自然な減衰)":
            window_type = 'butterworth'
            transition = st.number_input("フィルタの次数 (数値が大きいほど急峻にカット)", min_value=1, value=4, step=1)
        else:
            window_type = 'gaussian'
            transition = st.number_input("減衰の緩やかさ (標準偏差 MHz)", min_value=0.1, value=1.0, step=0.1)
            
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            max_freq = st.number_input("結果に残す最大周波数 (MHz)", min_value=0.0, value=10.0, step=0.1)
        with col2:
            pad_factor = pad_options[st.selectbox("分解能の倍率", options=list(pad_options.keys()))]

        if st.button("FFTを実行"):
            with st.spinner('処理中...'):
                limit = max_freq if max_freq > 0 else None
                st.session_state['result_df'] = func4_fft_smooth_filter(
                    df, filter_target=filter_target, cut_freq_mhz=cut_freq, 
                    transition=transition, window_type=window_type, 
                    max_freq_mhz=limit, pad_factor=pad_factor
                )
                st.session_state['limit'] = limit
                st.success("完了しました！")

        if 'result_df' in st.session_state:
            out_df = st.session_state['result_df']
            limit = st.session_state.get('limit', None)
            plot_df = out_df[out_df['Frequency_MHz'] <= limit] if limit is not None else out_df
            st.line_chart(plot_df.set_index('Frequency_MHz')['Amplitude'])
            st.download_button("out4.csv をダウンロード", data=out_df.to_csv(index=False).encode('utf-8'), file_name='out4.csv', mime='text/csv')

