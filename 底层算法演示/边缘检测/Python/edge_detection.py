"""按 README：灰度转换 -> 高斯降噪 -> Sobel / Laplacian / Canny 边缘检测，输出三路视频。"""
import cv2

SRC = "../1789298663603..mp4"
OPERATORS = ("sobel", "laplacian", "canny")

cap = cv2.VideoCapture(SRC)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writers = {name: cv2.VideoWriter(f"../{name}.mp4", fourcc, fps, (w, h)) for name in OPERATORS}

while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.convertScaleAbs(cv2.magnitude(gx, gy))
    laplacian = cv2.convertScaleAbs(cv2.Laplacian(blur, cv2.CV_64F))
    canny = cv2.Canny(blur, 100, 200)

    for name, edge in zip(OPERATORS, (sobel, laplacian, canny)):
        writers[name].write(cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR))

cap.release()
for w_ in writers.values():
    w_.release()
print("done:", ", ".join(f"{name}.mp4" for name in OPERATORS))
