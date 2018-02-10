import numpy as np  
import keras
import cv2
import math
from scipy.spatial import distance
#from vector import *
#import bresenham

# pocetak
# kod je pronadjen na gitu prošlogodišnje generacije, postavio ju je profesor Obradović
def dot(v, w):
    x, y = v
    X, Y = w
    return x * X + y * Y


def length(v):
    x, y = v
    return math.sqrt(x * x + y * y)


def vector(b, e):
    x, y = b
    X, Y = e
    return (X - x, Y - y)


def unit(v):
    x, y = v
    mag = length(v)
    return (x / mag, y / mag)


def distance(p0, p1):
    return length(vector(p0, p1))


def scale(v, sc):
    x, y = v
    return (x * sc, y * sc)


def add(v, w):
    x, y = v
    X, Y = w
    return (x + X, y + Y)


# Given a line with coordinates 'start' and 'end' and the
# coordinates of a point 'pnt' the proc returns the shortest
# distance from pnt to the line and the coordinates of the
# nearest point on the line.
#
# 1  Convert the line segment to a vector ('line_vec').
# 2  Create a vector connecting start to pnt ('pnt_vec').
# 3  Find the length of the line vector ('line_len').
# 4  Convert line_vec to a unit vector ('line_unitvec').
# 5  Scale pnt_vec by line_len ('pnt_vec_scaled').
# 6  Get the dot product of line_unitvec and pnt_vec_scaled ('t').
# 7  Ensure t is in the range 0 to 1.
# 8  Use t to get the nearest location on the line to the end
#    of vector pnt_vec_scaled ('nearest').
# 9  Calculate the distance from nearest to pnt_vec_scaled.
# 10 Translate nearest back to the start/end line.
# Malcolm Kesson 16 Dec 2012

def pnt2line(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0 / line_len)
    t = dot(line_unitvec, pnt_vec_scaled)
    r = 1
    if t < 0.0:
        t = 0.0
        r = -1
    elif t > 1.0:
        t = 1.0
        r = -1
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return (dist, (int(nearest[0]), int(nearest[1])), r)

#kraj

blueSum = 0
greenSum = 0

# def euclidean4(vector1, vector2):
#     ''' use scipy to calculate the euclidean distance. '''
#     dist = distance.euclidean(vector1, vector2)
#     return dist

def euclidean3(vector1, vector2):
    ''' use numpy.linalg.norm to calculate the euclidean distance. '''
    vector1, vector2 = list_to_npArray(vector1, vector2)
    distance = np.linalg.norm(vector1-vector2, 2, 0) # the third argument "0" means the column, and "1" means the line.
    return distance

def list_to_npArray(vector1, vector2):
    '''convert the list to numpy array'''
    if type(vector1) == list:
        vector1 = np.array(vector1)
    if type(vector2) == list:
        vector2 = np.array(vector2)
    return vector1, vector2


def findLongest(lines):
    temp = 0
    longest = lines[0]
    for line in lines:
        x1,y1,x2,y2 = line[0]
        vector1 = [x1,y1]
        vector2 = [x2,y2]
        dist = euclidean3(vector1,vector2)
        if(dist > temp):
            temp = dist
            longest = line
    
    return longest

#prvobitna metoda za tracking - ne radi kako treba
def generateNumbers(coords):
    for coord in coords:
        flag = False
        x = coord.x
        y = coord.y
        w = coord.w
        h = coord.h

        xw = x+w
        yh = y+h
        for broj in brojevi:
            vector1 = [broj.x+broj.w, broj.y+broj.h]
            vector2 = [xw, yh]
            dist = distance((broj.x+broj.w,broj.y+broj.h),(xw,yh))
            if(dist<20):
                broj.x = x
                broj.y = y
                broj.w = w
                broj.h = h
                flag = True
                
        if(flag == False):
            brojevi.append(ImgCoord(x,y,w,h,False,False,coord.img))
            
    
def generateNumbers2(kontura):
    x = kontura[0]
    y = kontura[1]
    w = kontura[2]
    h = kontura[3]

    xw = x+w
    yh = y+h

    noviBroj = ImgCoord(x,y,w,h,False,False,None)

    flag = False

    for broj in brojevi:
        dist = distance((broj.x+broj.w,broj.y+broj.h),(xw,yh))
        if(dist<20):
            return broj
                
    return None


class ImgCoord:

    def __init__(self,x,y,w,h,pb,pg,img):
        self.x = x
        self.y = y
        self.h = h
        self.w = w
        self.pb = False
        self.pg = False
        self.img = img

    def modify_values(x,y,h,w):
        self.x = x
        self.x = y

    def find_mid(x,y,h,w):
        return x+w/2 , w+h/2



def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret,image_bin = cv2.threshold(image_gs, 160, 255, cv2.THRESH_BINARY)
    return image_bin

def HoughLines(img_orig):
    gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150)
    gblur = cv2.GaussianBlur(edges,(7,7),1)

    lines = cv2.HoughLinesP(gblur,1,np.pi/180,100,100,30)
    

    return lines

def HoughLinesGRAY(img_orig):
    edges = cv2.Canny(img_orig,50,150)
    gblur = cv2.GaussianBlur(edges,(7,7),1)

    lines = cv2.HoughLinesP(gblur,1,np.pi/180,100,100,30)
    

    return lines

def calculateAreas(slike,blu_lines,gre_lines,coords,img,img_bin):
    
    X1b,Y1b,X2b,Y2b = blu_lines[0]
    X1g,Y1g,X2g,Y2g = gre_lines[0]

    P1g = [X1g,Y1g]
    P2g = [X2g,Y2g]
    P1b = [X1b,Y1b]
    P2b = [X2b,Y2b]


    for coord in coords:
        x = coord.x + coord.w
        y = coord.y + coord.h
        dist,nearest,r =pnt2line((x,y),(X1b,Y1b),(X2b,Y2b))
        if(dist <= 7 and coord.pb == False):
            coord.pb = True
            cv2.rectangle(img,(coord.x,coord.y),(x,y),(0,255,0),2)
            image = img_bin[coord.y-7:y+7,coord.x-7:x+7]
            #
            resized = cv2.resize(image, (28, 28), interpolation = cv2.INTER_NEAREST)
            scale = resized / 255
            mVector = scale.flatten()
            mColumn = np.reshape(mVector, (1, 784))
            imgNM = np.array(mColumn, dtype=np.float32)
            # 
            
            nmb = model_NM.predict(imgNM)
            global blueSum
            blueSum+=np.argmax(nmb)

        distg,nearestg,rg=pnt2line((x,y),(X1g,Y1g),(X2g,Y2g))
        if(distg <= 7 and coord.pg == False):
            coord.pg = True
            cv2.rectangle(img,(coord.x,coord.y),(x,y),(0,255,0),2)
            image = img_bin[coord.y-7:y+7,coord.x-7:x+7]
            #
            resized = cv2.resize(image, (28, 28), interpolation = cv2.INTER_NEAREST)
            scale = resized / 255
            mVector = scale.flatten()
            mColumn = np.reshape(mVector, (1, 784))
            imgNM = np.array(mColumn, dtype=np.float32)
            # 
            
            nmb = model_NM.predict(imgNM)
            blueSum-=np.argmax(nmb)


#prvobitna zamisao deljenja linija
def differLines(lines):
    suma = 0
    green_lines = []
    blue_lines = []
    for line in lines:
        x1,y1,x2,y2 = line[0]
        suma += y2
    
    mid = suma / len(lines)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        if(y2 > mid):
            green_lines.append(line)
        else:
            blue_lines.append(line)

    return green_lines,blue_lines





def resize_region(region):
    '''Transformisati selektovani region na sliku dimenzija 28x28'''
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)

def select_roi(image_orig, image_bin):
    '''Oznaciti regione od interesa na originalnoj slici. (ROI = regions of interest)
        Za svaki region napraviti posebnu sliku dimenzija 28 x 28. 
        Za označavanje regiona koristiti metodu cv2.boundingRect(contour).
        Kao povratnu vrednost vratiti originalnu sliku na kojoj su obeleženi regioni
        i niz slika koje predstavljaju regione sortirane po rastućoj vrednosti x ose
    '''
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = [] # lista sortiranih regiona po x osi (sa leva na desno)
    regions_array = []
    regions_coords = []
    for contour in contours: 
        x,y,w,h = cv2.boundingRect(contour) #koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        
        if h > 9:
            imali = generateNumbers2(cv2.boundingRect(contour))

            if imali is None:
                novi = ImgCoord(x,y,w,h,False,False,None)
                brojevi.append(novi)       
            else:
                imali.x = x
                imali.y = y
                imali.w = w
                imali.h = h
            
            region = image_bin[y:y+h,x:x+w]
            regions_array.append(resize_region(region))
            regions_coords.append(ImgCoord(x,y,w,h,False,False,resize_region(region)))
    
    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return image_orig, regions_array, regions_coords

def scale_to_range(image): # skalira elemente slike na opseg od 0 do 1
    ''' Elementi matrice image su vrednosti 0 ili 255. 
        Potrebno je skalirati sve elemente matrica na opseg od 0 do 1
    '''
    return image/255

def matrix_to_vector(image):
    '''Sliku koja je zapravo matrica 28x28 transformisati u vektor sa 784 elementa'''
    return image.flatten()


def prepare_for_ann(regions):
    '''Regioni su matrice dimenzija 28x28 čiji su elementi vrednosti 0 ili 255.
        Potrebno je skalirati elemente regiona na [0,1] i transformisati ga u vektor od 784 elementa '''
    ready_for_ann = []
    for region in regions:
        # skalirati elemente regiona 
        # region sa skaliranim elementima pretvoriti u vektor
        # vektor dodati u listu spremnih regiona
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
        
    return ready_for_ann

model_NM = keras.models.load_model("trainedNN/keras_mnist.h5")
cap = cv2.VideoCapture("genproj3/video-4.avi")

flag,ss = cap.read()
b,g,r = cv2.split(ss)
b_blur = cv2.GaussianBlur(b,(5,5),1)
lines = HoughLines(ss)
#g_lines,b_lines = differLines(lines)
blu_lines = HoughLinesGRAY(b_blur)
gre_lines = HoughLinesGRAY(g)
longestB = findLongest(blu_lines)
longestG = findLongest(gre_lines)


brojevi = []

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if(ret == True):

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_bin = image_bin(gray)
        img, slike, coords = select_roi(frame,img_bin)
        
        calculateAreas(slike,longestB,longestG,brojevi,frame,img_bin)
        
        x1,y1,x2,y2 = longestB[0]
        cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
        
        x1,y1,x2,y2 = longestG[0]
        cv2.line(frame,(x1,y1),(x2,y2),(255,255,0),2)
        

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the capture
#print(len(brojevi))
print(blueSum)
cap.release()
cv2.destroyAllWindows()