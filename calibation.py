import cv2
import numpy as np
import glob
import csv
import os

# ==========================================
# 參數設定 (請根據您的真實棋盤格內角點數量修改)
# ==========================================
# 注意：是「內角點」的數量，即黑白交界點。如果格子是 10x7，內角點通常是 9x6
CHECKERBOARD = (8, 6) 

# 設定亞像素精確化停止準則
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 準備 3D 空間點 (Object Points)，格子邊長嚴格設為 1.0
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = [] # 儲存所有圖片的 3D 空間點
imgpoints = [] # 儲存所有圖片的 2D 像素點

# 讀取資料夾內所有的校正圖片 (假設副檔名為 .jpg，可自行修改)
# 建議將 30 張圖片放在同一個資料夾，例如名為 'calib_images'
image_folder = '/home/r14942135/data/r14942135/AGZ_subset/MAV Images Calib' 
images = glob.glob(os.path.join(image_folder, '*.png'))

print(f"找到 {len(images)} 張校正圖片，開始偵測角點...")

success_count = 0
gray_shape = None

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_shape = gray.shape[::-1]

    # 1. 尋找粗略的 2D 棋盤格角點
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # 2. 如果成功找到，進行高精度亞像素精確化並收集點
    if ret == True:
        success_count += 1
        objpoints.append(objp) # 壓入邊長為 1.0 的 3D 坐標

        # 亞像素精確化 (學術研究必做，能將角點精確度提升到小數點後兩位像素)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        print(f"[{success_count}/{len(images)}] 成功提取角點: {os.path.basename(fname)}")
    else:
        print(f"[-] 無法提取角點（已自動跳過）: {os.path.basename(fname)}")

print(f"\n角點提取結束。成功：{success_count} 張，失敗：{len(images) - success_count} 張。")

# ==========================================
# 3. 執行多張圖片的聯合相機標定
# ==========================================
if success_count >= 10: # 確保有足夠的有效圖片才進行標定
    print("開始執行相機聯合最佳化標定...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)
    
    # 提取關鍵幾何參數
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    cx = mtx[0, 2]
    cy = mtx[1, 2]
    
    k1, k2, p1, p2, k3 = dist[0][:5]
    
    print("\n=== 標定成功 ===")
    print(f"重投影誤差 (RMS Error): {ret:.4f} 像素 (通常 < 0.5 代表極度精確)")
    print(f"內參矩陣 K:\n{mtx}")
    print(f"畸變係數 D:\n{dist}")

    # ==========================================
    # 4. 將參數寫入 CSV 檔案
    # ==========================================
    csv_filename = "uav_camera_intrinsics.csv"
    
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        
        # 寫入表頭
        writer.writerow(['Parameter', 'Value', 'Description'])
        
        # 寫入內參 Matrix K 元素
        writer.writerow(['fx', fx, 'Horizontal focal length in pixels'])
        writer.writerow(['fy', fy, 'Vertical focal length in pixels'])
        writer.writerow(['cx', cx, 'Horizontal optical center (principal point x)'])
        writer.writerow(['cy', cy, 'Vertical optical center (principal point y)'])
        
        # 寫入畸變係數 Distortion Coefficients
        writer.writerow(['k1', k1, 'Radial distortion coefficient 1'])
        writer.writerow(['k2', k2, 'Radial distortion coefficient 2'])
        writer.writerow(['p1', p1, 'Tangential distortion coefficient 1'])
        writer.writerow(['p2', p2, 'Tangential distortion coefficient 2'])
        writer.writerow(['k3', k3, 'Radial distortion coefficient 3'])
        
        # 寫入整體標定質量指標
        writer.writerow(['rms_error', ret, 'Root Mean Square re-projection error in pixels'])

    print(f"\n[SUCCESS] 所有幾何內參已成功寫入：{csv_filename}")

else:
    print("\n[ERROR] 有效圖片數量不足（少於10張），標定終止。請檢查 CHECKERBOARD 的內角點尺寸是否數錯。")