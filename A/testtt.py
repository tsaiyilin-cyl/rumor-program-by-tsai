import cv2
import numpy as np

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# 鼠标框选 ROI
r = cv2.selectROI("Select Target", frame, False)
cv2.destroyWindow("Select Target")

x, y, w, h = map(int, r)
track_window = (x, y, w, h)
roi = frame[y:y+h, x:x+w]
# 计算 ROI 的 HSV 颜色直方图
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [16], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# 终止条件：10 次迭代或中心移动 < 1 像素
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 反向投影得到概率图
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # 应用 CamShift 跟踪
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)

    # 绘制跟踪结果（椭圆）
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    cv2.polylines(frame, [pts], True, (0,255,0), 2)

    cv2.imshow('CamShift Tracking', frame)
    if cv2.waitKey(30) & 0xFF == 27:   # ESC 退出
        break

cap.release()
cv2.destroyAllWindows()