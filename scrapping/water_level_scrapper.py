import scrap_live_stream
import scrap_mesurament
import cv2
import time
import boto3
import tinys3
import os, os.path

AWS_KEY = 'XXX'
AWS_SECRET = 'XX'
BUCKET = 'XX'


def upload_to_aws(local_file, bucket, s3_file):
    s3 = boto3.client('s3', aws_access_key_id=AWS_KEY,
                      aws_secret_access_key=AWS_SECRET)

    try:
        s3.upload_file(local_file, bucket, s3_file)
        print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False


def getFileCount():
    path, dirs, files = next(os.walk("images\\"))
    return len(files)

if __name__ == "__main__":
    lastLevel = 0
    print('starting cycle..')
    while(True):
        time.sleep(60 * 5)
        hour, level, difference = scrap_mesurament.findFirstWaterLevel()
        print(hour)
        if True:
            lastLevel = level

            file_count = getFileCount()
            
            hour = hour.replace(':','_').replace('/','_')
            img = scrap_live_stream.getLiveStreamImageFrame()
            file_location = 'images\\' + str(level) + '_' + str(file_count) + '.jpg'
            try:
                if cv2.imwrite(file_location , img):
                    #uploaded = upload_to_aws(file_location, BUCKET, file_location.replace('images\\',''))
                    print('Captured!')
            except:
                print('ERROU')
            #    raise Exception("Could not write image")
            
            #conn.upload(file_location, f, BUCKET)
            #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            

