import cv2

cascade_filename = 'haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(cascade_filename)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

age_net = cv2.dnn.readNetFromCaffe(
    'deploy_age.prototxt',
    'age_net.caffemodel')

gender_net = cv2.dnn.readNetFromCaffe(
    'deploy_gender.prototxt',
    'gender_net.caffemodel')

age_list = ['(0 ~ 2)','(4 ~ 6)','(8 ~ 12)','(15 ~ 20)',
            '(25 ~ 32)','(38 ~ 43)','(48 ~ 53)','(60 ~ 100)']

gender_list = ['Male', 'Female']

def main():
    camera = cv2.VideoCapture(-1)
    camera.set(3,640)
    camera.set(4,480)
        
    while( camera.isOpened() ):
    
        _,img = camera.read()
        
        img = cv2.resize(img,dsize=None,fx=1.0,fy=1.0)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        results = cascade.detectMultiScale(gray,            # 입력 이미지
                                           scaleFactor= 1.1,# 이미지 피라미드 스케일 factor
                                           minNeighbors=5,  # 인접 객체 최소 거리 픽셀
                                           minSize=(20,20)  # 탐지 객체 최소 크기
                                           )

        for box in results:
            x, y, w, h = box
            face = img[int(y):int(y+h),int(x):int(x+h)].copy()
            blob = cv2.dnn.blobFromImage(face, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_preds.argmax()
            
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_preds.argmax()
            
            info = gender_list[gender] +' '+ age_list[age]

            cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), thickness=2)
            cv2.putText(img,info,(x,y-15),0, 0.5, (0, 255, 0), 1)

        cv2.imshow('facenet',img)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
