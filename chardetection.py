import json
from subprocess import call
def imagetotext():
    f = open("detectedplate.txt", "w")
    x=call(["aws","rekognition","detect-text","--image","S3Object={Bucket='alprdata',Name='img2.png'}"],stdout=f)
    with open('detectedplate.txt') as json_file:
        data = json.load(json_file)
        for p in data['TextDetections']:
           if(len(p['DetectedText'])==10):
               numplate=p['DetectedText']
               break
           elif(len(p['DetectedText'])>=10):
               numplate=p['DetectedText']
               numplate=numplate.replace(" ","")
               numplate=numplate.replace("IND","")
               numplate=numplate.replace(".","")
               break
    return numplate
