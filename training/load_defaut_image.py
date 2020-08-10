

def load():
    import json
    from PIL import Image, ImageDraw 
    import matplotlib.pyplot as plt
    import numpy as np
    
    json = json.load(open('C:\\Users\\Wollinger\\Desktop\\TCC\\river_annotation\\ds\\ann\\0.88_215.jpg.json'))    
    
    objects = json['objects']
    points = objects[0]['points']
    exterior = points['exterior']
      
    polygons = []
    shape = (1920, 1080)
    
    for i in range(len(exterior)):
        polygons.append(tuple(exterior[i]))
    
    img = Image.new('L', shape, 0)
    ImageDraw.Draw(img).polygon(polygons, outline=1, fill=1)
    mask = np.array(img)
    arr = mask.astype('bool')
    
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if(mask[i][j] == 0):
                arr[i][j] = False
            else:
                arr[i][j] = True

    return arr.reshape((1080,1920,1))