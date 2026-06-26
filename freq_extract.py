import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq

# ==========================================
# 1. 參數與檔案路徑設定
# ==========================================
image_path = '00336.jpg'         # 您的單張無人機測試圖
gyro_csv_path = 'RawGyro.csv'    # 您的陀螺儀數據

# 假設這張圖片的觸發時間戳 (依據 GroundTruthAGM 或是影像檔名對應)
# 這裡設定為 RawGyro 中的某個起始時間，加上 30ms (30,000 微秒) 的捲簾快門曝光時間
image_timestamp_start = 7089907  
exposure_time_us = 30000         
image_timestamp_end = image_timestamp_start + exposure_time_us

# ==========================================
# 2. 圖片空間頻率分析 (2D FFT)
# ==========================================
# 讀取圖片並轉灰階
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"找不到圖片 {image_path}")

# 執行 2D FFT
f_transform = np.fft.fft2(img)
f_shift = np.fft.fftshift(f_transform) # 將零頻率移到中心
# 取對數以壓縮振幅範圍，方便視覺化
magnitude_spectrum_2d = 20 * np.log(np.abs(f_shift) + 1)

# ==========================================
# 3. 陀螺儀時間頻率分析 (1D FFT 於 30ms 曝光窗內)
# ==========================================
# 讀取陀螺儀 CSV (忽略頭尾可能的空白或錯誤格式)
df_gyro = pd.read_csv(gyro_csv_path, skipinitialspace=True)
# 確保欄位名稱正確 (去除可能的多餘空白)
df_gyro.columns = [col.strip() for col in df_gyro.columns]

# 篩選出這張圖片「曝光 30ms 期間」的陀螺儀數據
mask = (df_gyro['Timpstemp'] >= image_timestamp_start) & (df_gyro['Timpstemp'] <= image_timestamp_end)
df_exposure = df_gyro.loc[mask]

if df_exposure.empty:
    print("[警告] 在指定的 30ms 曝光窗內找不到陀螺儀數據，請確認時間戳。")
    # 為了展示代碼邏輯，若找不到則強制取前 50 筆數據模擬 30ms
    df_exposure = df_gyro.head(50)

# 取出 Y 軸 (Roll) 或 X 軸 (Pitch) 的角速度作為分析範例
gyro_y = df_exposure['y'].values
N = len(gyro_y)
# 計算採樣頻率 (假設數據點均勻分佈於 30ms 內)
T = (exposure_time_us / 1e6) / N if N > 0 else 0.001 

# 執行 1D FFT
yf = fft(gyro_y)
xf = fftfreq(N, T)[:N//2]
magnitude_spectrum_1d = 2.0/N * np.abs(yf[0:N//2])

# ==========================================
# 4. 繪製對比圖表
# ==========================================
plt.figure(figsize=(16, 10))

# [子圖 1] 原始圖片
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('1. Original UAV Image (Spatial Domain)')
plt.axis('off')

# [子圖 2] 圖片的 2D 頻譜
plt.subplot(2, 2, 2)
plt.imshow(magnitude_spectrum_2d, cmap='jet')
plt.title('2. Image 2D FFT Spectrum (Shows Blur Directionality)')
plt.axis('off')

# [子圖 3] 曝光 30ms 內的原始陀螺儀波形
plt.subplot(2, 2, 3)
plt.plot(df_exposure['Timpstemp'], df_exposure['x'], label='Omega_x (Pitch)')
plt.plot(df_exposure['Timpstemp'], df_exposure['y'], label='Omega_y (Roll)')
plt.plot(df_exposure['Timpstemp'], df_exposure['z'], label='Omega_z (Yaw)')
plt.title('3. Gyroscope Data during 30ms Exposure (Time Domain)')
plt.xlabel('Timestamp (us)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()
plt.grid(True)

# [子圖 4] 陀螺儀數據的 1D 頻譜
plt.subplot(2, 2, 4)
plt.plot(xf, magnitude_spectrum_1d, color='red')
plt.title('4. Gyroscope 1D FFT Spectrum (Shows Vibration Frequencies)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()