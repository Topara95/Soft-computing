import cv2
import sys
import keras


tracker = cv2.TrackerMIL_create()
tracker_type = "MIL"
video = cv2.VideoCapture("C:/Users/Topolic/Desktop/Jovan/soft/projekat/genproj3/video-0.avi")

if not video.isOpened():
        print ("Could not open video")
        sys.exit()

ok, frame = video.read()
if not ok:
    print ('Cannot read video file')
    sys.exit()


# Define an initial bounding box
    #bbox = (287, 23, 86, 320)
 
# Uncomment the line below to select a different bounding box
bbox = cv2.selectROI(frame, False)
 
# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)


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

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin



number = 0
model_NM = keras.models.load_model("keras_mnist.h5")
while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
         
        # Start timer
        timer = cv2.getTickCount()
 
        # Update tracker
        ok, bbox = tracker.update(frame)
 
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
 
        # Draw bounding box
        if ok:
            
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            #region = frame[bbox[0]-2:bbox[1]+bbox[3]+2,bbox[0]-2:bbox[0]+bbox[2]+2]
            #reg = prepare_for_ann(region)
            #number = model_NM.predict(reg)
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
 
        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
     
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
 
        # Display result
        cv2.imshow("Tracking", frame)
 
        # Exit if ESC pressed
        k = cv2.waitKey(40) & 0xff
        if k == 27 : break