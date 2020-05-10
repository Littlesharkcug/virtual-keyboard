
import cv2
import numpy as np
import time



def DistortCamera(img1):
    img2 = np.zeros(img1.shape, np.uint8) #创建一个空图像
    a=[1063.52854201920 ,  0    ,975.973148837204]
    b=[ 0.0,1062.77067424982,   527.294121972490]
    c=[0.0 , 0.0  ,1]
    IntrinsicMatrix=np.array([a,b,c]) #内参数矩阵
    DistortionCoeffs= np.array([ -0.416859697151604	,0.243400173184379	,-0.000386511315523013,	-0.000442537250503539, -0.0948327836627917])
    cv2.undistort(img1,IntrinsicMatrix,DistortionCoeffs,img2)
    return img2

#寻找手指轮廓与坐标
# def Fincounters(img):





cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
cap.set(cv2.CAP_PROP_EXPOSURE,-8)  #调整相机曝光 用AmCap软件测试 这个数值比较可以用
num = 1
while cap.isOpened():
     read,frame=cap.read()
     if not read:
       break

     starttime = time.time()
     # 原图像
     # cv2.namedWindow('1', cv2.WINDOW_NORMAL)    # 窗口大小可以改变
     # cv2.resizeWindow('1',960, 540)
     # cv2.imshow("1",frame)


     #图像处理部分
     img = DistortCamera(frame)
     imggray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#转灰度图
     gaussianResult = cv2.GaussianBlur(imggray,(5,5),1.5) #高斯滤波
     ret,mask =cv2.threshold(gaussianResult,200,255,cv2.THRESH_BINARY)#二值化

     kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
     opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) #开运算

     contours,hierarchy=cv2.findContours(opened,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
     # cv2.drawContours(img,contours,-1,(0,0,255),3)

     # if len(contours) != 1:#轮廓总数


     M = cv2.moments(contours[0]) # 计算第一条轮廓的各阶矩,字典形式
     center_x = int(M["m10"] / M["m00"])
     center_y = int(M["m01"] / M["m00"])
     # h1,w1 = opened.shape
     # image = np.zeros([h1, w1], dtype=opened.dtype)
     cv2.drawContours(img,contours,-1,(0,0,255),3)
     # cv2.drawContours(img, contours, 0, 255, -1)#绘制轮廓，填充
     cv2.circle(img, (center_x, center_y), 7, 128, -1)#绘制中心点
     print("center_x, center_y",center_x, center_y)


     #计算FPS
     endtime = time.time()
     seconds = endtime - starttime
     fps = 1/seconds
     print("fps:",fps)

     cv2.namedWindow('2', cv2.WINDOW_NORMAL)    # 窗口大小可以改变
     cv2.resizeWindow('2',960, 540)
     cv2.imshow("2",img)

     if cv2.waitKey(5)&0xFF==ord('q'):
      break
#释放相关资源
cap.release()
cv2.destroyAllWindows()


