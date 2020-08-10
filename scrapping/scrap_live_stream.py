import cv2
import time
from datetime import datetime
import os


def getLiveStreamImageFrame():

    #Ponte Dom Tito Buss
    url = 'rtsp://pontedonardelli.ddns-intelbras.com.br:554/user=furb&password=rio101&channel=1&stream=0.sdp?'
    #Ponte Pamplona - não funciona
    #url = "rtsp://pontedopamplona.ddns-intelbras.com.br:554/user=furb&password=rio101&channel=1&stream=0.sdp?"
    #Ponte da XV
    #url = 'rtsp://pontedaxv.ddns-intelbras.com.br:554/user=furb&password=rio101&channel=1&stream=0.sdp?'
    playurl = url

    cap = cv2.VideoCapture(playurl)

    ret, frame = cap.read()

    cap.release()
    #cv2.destroyAllWindows()


    return frame        
#if not cv2.imwrite('images\\' + str(datetime.now()).replace(":","_") + ".jpg", frame):
#    raise Exception('NOPS')
    #cv2.imshow('frame', frame)

