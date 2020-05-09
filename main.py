
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


cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
cap.set(cv2.CAP_PROP_EXPOSURE,-8)  #调整相机曝光 用AmCap软件测试 这个数值比较可以用

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
     frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#转灰度图



     endtime = time.time()
     seconds = endtime - starttime
     fps = 1/seconds
     print("消耗时间",seconds)
     print("fps:",fps)
     cv2.namedWindow('2', cv2.WINDOW_NORMAL)    # 窗口大小可以改变
     cv2.resizeWindow('2',960, 540)
     cv2.imshow("2",img)
     print("2",img.shape)

     if cv2.waitKey(5)&0xFF==ord('q'):
      break
#释放相关资源
cap.release()
cv2.destroyAllWindows()


# while True:
#     path = "E:\keyboards\picture\WIN_20200508_20_35_46_Pro.jpg"
#     img1 = cv2.imread(path)
#     print("图像的大小",img1.shape)
#
#     img2 = DistortCamera(img1)
#     cv2.namedWindow('2', cv2.WINDOW_NORMAL)    # 窗口大小可以改变
#     cv2.resizeWindow('2',960, 540)
#     cv2.imshow("2",img2)
#     print("2",img2.shape)
#
#     if cv2.waitKey(5)&0xFF==ord('q'):
#       break
# #释放相关资源
#
# cv2.destroyAllWindows()
