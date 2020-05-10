
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

px1 = -3.426e-05
px2 = 0.03892
px3= 21.87
py1 =  5.297e-05
py2 =  -0.1072
py3 =  52.37

#键值 字典
keyboard_dit = {'q':(9.5,9.5),'w':(13,9.5),'e':(16.5,9.6),'r':(20,9.6),'t':(22.5,9.6),'y':(26,9.3),'u':(29.2,9.3),'i':(32.1,9.3),'o':(35.2,9.2),'p':(38.5,9),
       'a':(10.7,11.2),'s':(14,11.2),'d':(17.2,11.2),'f':(20.5,11.2),'g':(24,11.1),'h':(26.5,11),'j':(30,11),'k':(33,11),'l':(35.5,11),
       'z':(13,13.1),'x':(16,13.1),'c':(19,13.1),'v':(22.3,13),'b':(25.2,13),'n':(28,13),'m':(31,13)}
k=[]  #空数组 用来装 欧氏距离

cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
cap.set(cv2.CAP_PROP_EXPOSURE,-8)  #调整相机曝光 用AmCap软件测试 这个数值比较可以用
num = 0

while cap.isOpened():
     read,frame=cap.read()
     if not read:
       break

     k=[]  #需要清零 否则会在上一轮基础上增加
     starttime = time.time()
     # 原图像
     # cv2.namedWindow('1', cv2.WINDOW_NORMAL)    # 窗口大小可以改变
     # cv2.resizeWindow('1',960, 540)
     # cv2.imshow("1",frame)


     #图像处理部分
     img = DistortCamera(frame)
     imggray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#转灰度图
     gaussianResult = cv2.GaussianBlur(imggray,(5,5),1.5) #高斯滤波
     ret,mask =cv2.threshold(gaussianResult,175,255,cv2.THRESH_BINARY)#二值化

     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
     opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) #开运算

     contours,hierarchy = cv2.findContours(opened,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
     #cv2.drawContours(img,contours,-1,(0,0,255),3)
     # print('contours[0]',contours[0])
     if len(contours) != 0:#轮廓总数
         M = cv2.moments(contours[0]) # 计算第一条轮廓的各阶矩,字典形式
         center_x = int(M["m10"] / M["m00"])
         center_y = int(M["m01"] / M["m00"])
         # h1,w1 = opened.shape
         # image = np.zeros([h1, w1], dtype=opened.dtype)
         cv2.drawContours(img,contours,-1,(0,0,255),3)      # cv2.drawContours(img, contours, 0, 255, -1)#绘制轮廓，填充
         cv2.circle(img, (center_x, center_y), 7, 128, -1)  #绘制中心点
         # print("center_x, center_y",center_x, center_y)

     fx = px1*pow(center_x,2)+px2*center_x+px3
     fy = py1*pow(center_y,2)+py2*center_y+py3
     print("fx , fy: ",fx ,fy)

     #计算FPS
     endtime = time.time()
     seconds = endtime - starttime
     fps = 1/seconds
     #print("fps:",fps)

     cv2.namedWindow('2', cv2.WINDOW_NORMAL)    # 窗口大小可以改变
     cv2.resizeWindow('2',960, 540)
     cv2.imshow("2",img)

     #计算键值
     for values in keyboard_dit.values():
        distance=pow(values[0]-fx,2)+pow(values[1]-fy,2)
        k.append(distance)

     min_distance= min(k)
     min_distance_index=k.index(min_distance)
     print(list(keyboard_dit.items())[min_distance_index][0])

     if cv2.waitKey(5)&0xFF==ord('q'):
      break
#释放相关资源
cap.release()
cv2.destroyAllWindows()


