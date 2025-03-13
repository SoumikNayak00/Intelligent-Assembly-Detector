from tkinter import *
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import datetime
from tkinter import filedialog
import os
import numpy as np
from tkinter.scrolledtext import ScrolledText
import cv2
from tkinter import *
from PIL import Image, ImageTk
import threading
import cv2 as cv
from tkinter import *
from PIL import Image, ImageTk
from ultralytics import YOLO
import time
# Monkey patch for getargspec
import inspect
inspect.getargspec = inspect.getfullargspec
from pyfirmata import Arduino #needs to be installed
## StandardFirmata must be uploaded onto the  arduino board, before using it
## for the first time (can be found in standard arduino ide example codes)


######################################### Arduino Part


global start_needle_0
global start_needle_1
global start_needle_2
start_needle_0 = False
start_needle_1 = False
start_needle_2 = False
global start_needle_bearing
global stop_needle_bearing

# Load YOLO model
model_t = YOLO('needle_bearing_v9\\content\\runs\\segment\\train\\weights\\best.pt')
######## define functions for "camera for calibration tab"###################
#model_ass=YOLO('needle_bearing_assembly_v2\\content\\runs\\detect\\train\\weights\\best.pt')
#model_ass=YOLO('gear_after_needle\\content\\runs\\detect\\train\\weights\\best.pt')
model_ass=YOLO('gear_after_needle_v2\\content\\runs\\segment\\train\\weights\\best.pt')
#######ROLLER MODEL
model_roller_roi = YOLO('gear_roi_3\\content\\runs\\detect\\train\\weights\\best.pt')
model_test_balls = YOLO('detecting_rollers_v3\\content\\runs\\segment\\train2\\weights\\best.pt')
## ###############################setting up the geometry of main GUI, creating frames

root = Tk()
root.title('Cross Member Dimension Checker')
root.geometry("2000x1400")
import datetime
    
# using now() to get current time
current_time = datetime.datetime.now()
print(type(current_time.date))

if current_time.year==2025 and current_time.month<4 :
    
    
    notebook = ttk.Notebook(root)
    notebook.pack(pady=15)
    f1 = Frame(notebook, width=1950, height=1350, bg="lightSkyBlue2")
    f2 = Frame(notebook, width=1950, height=1350, bg="lightSkyBlue2")
    #f3 = Frame(notebook, width=1950, height=1350, bg="lightSkyBlue2")
    #f4 = Frame(notebook, width=1950, height=1350, bg="lightSkyBlue2")
    f1.pack(fill="both", expand=1)
    f2.pack(fill="both", expand=1)
    #f3.pack(fill="both", expand=1)
    #f4.pack(fill="both", expand=1)
    f1.pack(fill="both", expand=1)
    f2.pack(fill="both", expand=1)
    #f3.pack(fill="both", expand=1)
    #f4.pack(fill="both", expand=1)
    notebook.add(f1, text="Needle Cage detection")
    notebook.add(f2, text="Roller Bearing Detection")
    #notebook.add(f3, text="Measurement")
    #notebook.add(f4, text="Manual")
    style = ttk.Style()

# Modify the padding and font size of the tabs
    style.configure("TNotebook.Tab",padding=[10, 5], font=("Arial", 14, "bold")) 

    ####some drop down menu function for arduino


    options = [f"COM{i}" for i in range(1, 15)]  # Generates COM1 to COM9

    # Create a variable to track the selected option
    selected_option = StringVar()
    selected_option.set(options[0])  # Default to COM1



    # Create the dropdown menu
    com_selector = OptionMenu(f1, selected_option, *options)
    com_selector.config(font=('Times 14'), bg='white', fg='black')
    com_selector['menu'].config(font=('Times', 18)) 
    com_selector.place(x=1000, y=910)

    # Define global variables
    board = None
    pin_12 = None

    # Initialize Arduino connection
    def initialize_arduino():
        global board, pin_12
        selected_com = selected_option.get()  # Get the selected COM port

        try:
            if 'board' in globals() and board:
                board.exit()  # Close the previous connection
                print("Previous connection closed.")
        except Exception as e:
            print(f"Error while closing the previous connection: {e}")


        try:
            # Initialize Arduino
            board = Arduino(selected_com)
            time.sleep(1)  # Allow the connection to stabilize
            print(f"Connected to Arduino on {selected_com}")
            com_selector.config(font=('Times 14'), bg='green', fg='white')           
            # Initialize pin 12 as a digital output
            pin_12 = board.get_pin('d:12:o')
            print("Pin 12 initialized as digital output")
        except Exception as e:
            print(f"Error connecting to {selected_com}: {e}")
            com_selector.config(font=('Times 14'), bg='red', fg='white')   

    # Function to turn on the relay
    def turn_on():
        if pin_12:
            pin_12.write(1)
            print("Relay ON")
        else:
            print("Arduino is not initialized. Please initialize first.")

    # Function to turn off the relay
    def turn_off():
        if pin_12:
            pin_12.write(0)
            print("Relay OFF")
        else:
            print("Arduino is not initialized. Please initialize first.")


    front=PhotoImage(file= "needle.png",master= f1)
    img= PhotoImage( master= f1)
    img_label_1= Label(f1,image=front)
    img_label_1.place(x=2, y=0)
    img_label_2= Label(f2,image=img)
    img_label_2.place(x=0, y=0)
    #img_label_3= Label(f3,image=img)
    #img_label_3.place(x=0, y=0)



    #Button(f2, text="Start Tab 2", font=("Arial", 14), command=activate_tab2, bg="green", fg="white").pack(pady=20)
    ############setting 2nd tab##$#########


    frame4 = Frame(master=f2, width=800, height=500, bg="black")  # Increased width for better visibility
    #frame4.pack_propagate(False)
    frame4.place(x=50, y=100)  # Positioned towards the left
    label4 = Label(frame4)
    label4.pack()

    # Labels with instructions on the right side
    label41 = Label(master=f2, text="Camera Feed", font=('Times 14'), width=30, height=2)
    label41.place(x=1200, y=150)  # Adjusted x and y to place it on the right side
    label411 = Label(master=f2, text="ROI detected", font=('Times 14'), width=30, height=2)
    label411.place(x=1200, y=250)
    label412 = Label(master=f2, text="No Thrust Bearing", font=('Times 14'), width=30, height=2)
    label412.place(x=1200, y=350)
    label4s = Label(master=f2, text="Station-0", font=('Times 14'), width=15, height=1)
    label4s.place(x=250, y=65)



    ##############For tab 1############
    frame1 = Frame(master=f1, width=250, height=240, bg="black")
    frame1.place(x=2, y=100)
    label1 = Label(frame1)
    label1.pack()

    # Labels with instructions
    label11 = Label(master=f1, text="Camera 1 Feed", font=('Times 14'), width=30, height=2)
    label11.place(x=100, y=650)
    label111 = Label(master=f1, text="Gear detected", font=('Times 14'), width=30, height=2)
    label111.place(x=100, y=750)
    label112 = Label(master=f1, text="Count: ", font=('Times 14'), width=30, height=2)
    label112.place(x=100, y=850)
    label1s = Label(master=f1, text="Station-1", font=('Times 14'), width=15, height=1)
    label1s.place(x=200, y=65)

    frame2 = Frame(master=f1, width=250, height=240, bg="black")
    frame2.place(x=650, y=100)
    label2 = Label(frame2)
    label2.pack()
    label2s = Label(master=f1, text="Station-2", font=('Times 14'), width=15, height=1)
    label2s.place(x=900, y=65)
    def start_needle_bearing():
        global thread1, thread2, thread3
        initialize_arduino()
        thread1 = threading.Thread(target=video_stream, args=(0, label1, label11, label111,label112))
        thread2 = threading.Thread(target=video_stream, args=(1, label2,label21 ,label211,label212))
        thread3 = threading.Thread(target=video_stream, args=(2, label3,label31, label311,label312))
        #thread4 = threading.Thread(target=video_stream_roller_bayring, args=(3, label4,label41, label411,label412))

    # Daemon threads allow the program to exit even if threads are running
        thread1.daemon = True
        thread2.daemon = True
        thread3.daemon = True
        #thread4.daemon = True
    

    

        thread1.start()
        thread2.start()
        thread3.start()

        #thread4.start()
        root.mainloop()

    def roller_bearing():
        thread4 = threading.Thread(target=video_stream_roller_bayring, args=(0, label4,label41, label411,label412))
        thread4.daemon = True
        thread4.start()

    

    # Labels with instructions
    label21 = Label(master=f1, text="Camera 2 Feed", font=('Times 14'), width=30, height=2)
    label21.place(x=800, y=650)
    label211 = Label(master=f1, text="Gear detected", font=('Times 14'), width=30, height=2)
    label211.place(x=800, y=750)
    label212 = Label(master=f1, text="Count: ", font=('Times 14'), width=30, height=2)
    label212.place(x=800, y=850)
    label3s = Label(master=f1, text="Station-3", font=('Times 14'), width=15, height=1)
    label3s.place(x=1600, y=65)

    but_needle = Button(master=f1, text="Start: ", font=('Times 14'), width=10, height=1, command=start_needle_bearing)
    but_needle.place(x=850, y=910)
    #frame_needle = Entry(master=f1, text="Put the frame here: ", font=('Times 14'), width=10)
    #frame_needle.place(x=1000, y=930)

    but_roller = Button(master=f2, text="Start: ", font=('Times 14'), width=10, height=1, command=roller_bearing)
    but_roller.place(x=1320, y=500)

    label3t = Label(master=f1, text="Select the correct Arduino COM \n Connect with port 12 of Arduino",  font=('Helvetica', 9, 'bold'), fg='black', bg='lightSkyBlue2',width=30, height=2)
    label3t.place(x=1122, y=900)

    
    

    frame3 = Frame(master=f1, width=250, height=240, bg="black")
    frame3.place(x=1300, y=100)
    label3 = Label(frame3)
    label3.pack()

    # Labels with instructions
    label31 = Label(master=f1, text="Camera 3 Feed", font=('Times 14'), width=30, height=2)
    label31.place(x=1450, y=650)
    label311 = Label(master=f1, text="Gear detected", font=('Times 14'), width=30, height=2)
    label311.place(x=1450, y=750)
    label312 = Label(master=f1, text="Count: ", font=('Times 14'), width=30, height=2)
    label312.place(x=1450, y=850)
    label3s = Label(master=f1, text="Station-3", font=('Times 14'), width=15, height=1)
    label3s.place(x=1600, y=65)






    def calculate_distance_from_center(box_center, image_center):
        """Calculates Euclidean distance from box center to image center."""
        return np.sqrt((box_center[0] - image_center[0]) ** 2 + (box_center[1] - image_center[1]) ** 2)

    def non_max_suppression(boxes, scores, iou_threshold=0.5):
        """Performs Non-Max Suppression on bounding boxes."""
        if len(boxes) == 0:
            return []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        return keep


    def video_stream_roller_bayring(camera_index, label,label1,label11,label112):
        
        cap = cv2.VideoCapture(camera_index,cv2.CAP_DSHOW)
        while True:
            ret, frame = cap.read()
            if ret:
                frame1=frame.copy()
                frame2=frame.copy()
                img_rgb = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)

                # Perform inference on the first model
                results_t = model_roller_roi(img_rgb, conf=0.5, line_width=1)
                bboxes_t = results_t[0].boxes.xyxy.cpu().numpy()
                confidences_t = results_t[0].boxes.conf.cpu().numpy()
                if len(confidences_t) > 0:
                    label1.configure(text="ROI Detected",bg='green' , fg='white')
                    max_conf_index = confidences_t.argmax()
                    best_bbox = bboxes_t[max_conf_index]
                    x1, y1, x2, y2 = map(int, best_bbox)
                    cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cropped_img = img_rgb[y1:y2, x1:x2]
                    cropped_img = cv.resize(cropped_img, (320, 320))

                    # Perform inference on the second model
                    results = model_test_balls(cropped_img)
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    scores = results[0].boxes.conf.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()

                    # Initialize a dictionary to count objects of each class
                    class_counts = {}
                    for cls in classes:
                        class_counts[cls] = class_counts.get(cls, 0)+1
                    label11.configure(text=f"No of Rollers={class_counts.get(0, 'Key not found')}", bg='green', fg='white')
                    # Process class 0
                    class_0_indices = np.where(classes == 0)[0]
                    boxes_class_0 = boxes[class_0_indices]
                    scores_class_0 = scores[class_0_indices]
                    keep_indices = non_max_suppression(boxes_class_0, scores_class_0)
                    filtered_boxes = boxes_class_0[keep_indices]
                    image_center = (cropped_img.shape[1] / 2, cropped_img.shape[0] / 2)

                    final_boxes = []
                    count = 0
                    for box in filtered_boxes:
                        print("lenght of boxes")
                        print(len(box))
                        x1_box, y1_box, x2_box, y2_box = box
                        box_center = ((x1_box + x2_box) / 2, (y1_box + y2_box) / 2)
                        distance = calculate_distance_from_center(box_center, image_center)

                        if 20 <= distance <= 90:
                            final_boxes.append(box)
                            # Draw bounding box on the original frame
                            #cv.rectangle(frame, (int(x1_box), int(y1_box)), (int(x2_box), int(y2_box)), (0, 255, 0), 2)
                            # Coordinates of the cropped area on the original frame
                            #x1, y1, x2, y2 = map(int, best_bbox)  # These are the coordinates of the crop in the original frame

                            # Now, when drawing on the original frame, adjust the box coordinates
                            #for box in filtered_boxes:
                            # x1_box, y1_box, x2_box, y2_box = box

                                # Adjust the box coordinates to the original frame's coordinates
                                
                                # Draw the rectangle on the original frame with adjusted coordinates
                                #cv.rectangle(frame, (adjusted_x1, adjusted_y1), (adjusted_x2, adjusted_y2), (0, 255, 0), 2)
                            count += 1
                        label11.configure(text=f"No of Rollers={count}", bg='green', fg='white')
                    class_counts[0] = class_counts.get(0, 0)
                    print(class_counts.get(1, 0)) 
                    
                    # Update class 1 count to "yes" or "no"
                    #if 1 in class_counts and class_counts[1] >= 1:
                    if 1 in class_counts :
                        class_counts[2] = "yes"
                        label112.configure(text=f"Thrust Bearing=Yes", bg='green', fg='white')    
                    else:
                        class_counts[2] = "no"
                        label112.configure(text=f"Thrust Bearing=No", bg='red', fg='white') 
                    
                    
                else:
                    label11.configure(text="ROI not Detected",bg='red' , fg='white')
                    label1.configure(text="ROI not Detected", bg='red', fg='white')
                    label112.configure(text=f"Thrust Bearing=No", bg='red', fg='white')


                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                label.imgtk = imgtk
                label.configure(image=imgtk)

            else:
                print(f"Camera {camera_index} not accessible")
                break
        cap.release()


    reconnect_attempted = False    


    def video_stream(camera_index, label,label1,label11, label112):
        detection_array_needle = []  
        detection_array_gear=[]
        global start_needle_0, start_needle_1, start_needle_2
        global start_needle_vars
        if 'start_needle_vars' not in globals():
            start_needle_vars = {}  # Initialize dictionary if not present
        var_name = f"start_needle_{camera_index}"
        globals()[var_name] = True
        start_needle_vars[f"start_needle_{camera_index}"] = True
       # global start_needle
       # start_needle=True
        #global detection_array_needle
        turn_on()  
        time.sleep(1)
        turn_off()    
        cou1=0
        counti=0
        frame_interval=2
        cap = cv2.VideoCapture(camera_index,cv2.CAP_DSHOW)
        frame_count = 0
        aa=0
        cc=0
        global reconnect_attempted
        reconnect_attempted = False 
        while  globals()[f"start_needle_{camera_index}"]:
            print("loop running.")
            if not cap.isOpened():
                        print("Camera disconnected. Waiting for reconnection...")
                        #if not reconnect_attempted:  # Ensure reconnection is attempted only once
                        reconnect_attempted = True  # Set the flag

                        cap = cv2.VideoCapture(camera_index,cv2.CAP_DSHOW)
                    # else:
                        print("Frame read failed. Continuing...")                 
           
            ret, frame = cap.read()
            if ret:
                reconnect_attempted = True
                if frame_count % frame_interval == 0:
                    print("Camera connected.")
                    frame1=frame.copy()
                    frame2=frame.copy()
                    img_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                    img_gray_3channel = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
                    results_t = model_t(img_gray_3channel, conf=0.35, line_width=1, imgsz=128)
                    
                    # Retrieve bounding boxes, confidence scores, and class labels
                    bboxes_t = results_t[0].boxes.xyxy.cpu().numpy()
                    confidences_t = results_t[0].boxes.conf.cpu().numpy()
                    class_ids_t = results_t[0].boxes.cls.cpu().numpy().astype(int)  # Get class IDs
                    
                    
                    if len(confidences_t) > 0:
                        # Find the index of the maximum confidence
                        max_conf_index = confidences_t.argmax()
                        
                        # Extract the bounding box, confidence, and class ID of the highest confidence detection
                        best_bbox = bboxes_t[max_conf_index]
                        best_confidence = confidences_t[max_conf_index]
                        best_class_id = class_ids_t[max_conf_index]

                        x1, y1, x2, y2 = map(int, best_bbox)

                        # Set bounding box color based on the class ID
                        if best_class_id == 0:  # Class 0: Needle
                            bb=1
                            color = (0, 0, 255)  # Red
                            labels = "Needle Detected"
                            detection_array_needle.append(1)
                            
                            if sum(detection_array_needle)<=6:
                             #label1.configure(text="Needle Detected",,bg='red' , fg='white')
                             label1.configure(text=f"Needle Detecting: {sum(detection_array_needle)}",bg='red' , fg='white')
                             #aa=0
                            
                            
                            if sum(detection_array_needle)>6 and sum(detection_array_gear[-20:]) < 13:
                                #label1.configure(text="Needle Detected",bg='green' , fg='white')
                                #label1.configure(text=sum(detection_array_needle),bg='green' , fg='white')
                                label1.configure(text="Needle Already Detected",bg='green' , fg='white')
                                aa=1

                               
                        # Combine label with confidence score
                            label_with_confidence = f"{labels} ({best_confidence:.2f})"
                            
                            # Draw the bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Draw the label and confidence score
                            cv2.putText(frame, label_with_confidence, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        elif best_class_id == 1 and aa==0:  # Class 1: Opposite Needle
                            label1.configure(text="Needle not Detected",bg='red' , fg='white')
                            #label1.configure(text="Needle not Detected",bg='red' , fg='white')
                            detection_array_needle.append(0)
                        elif best_class_id == 1 and aa==1:
                            if sum(detection_array_needle)>=-6:
                              label1.configure(text=f"Needle Already Detected",bg='green' , fg='white')
                              detection_array_needle.append(-1)
                            elif sum(detection_array_needle)<-6:
                              aa=0
                              label1.configure(text=f"Needle not Detected",bg='red' , fg='white') 
                              detection_array_needle.append(-1)                             
                            bb=0
                            
                    else: 
                        if aa==1:
                           label1.configure(text=f"Needle Already Detected",bg='green' , fg='white')
                        else:
                            if cc==0:
                                label1.configure(text="Needle not Detected",bg='red' , fg='white')
                        
                        detection_array_needle.append(0)
                    if len(detection_array_needle) > 20:  # Ensure the array size stays at 20
                        detection_array_needle.pop(0)                        

                    
                    img_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                    img_gray_3channel = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
                    results_t = model_ass(frame1, conf=0.5, line_width=1, imgsz=128)
                    
                    # Retrieve bounding boxes, confidence scores, and class labels
                    bboxes_t = results_t[0].boxes.xyxy.cpu().numpy()
                    confidences_t = results_t[0].boxes.conf.cpu().numpy()
                    class_ids_t = results_t[0].boxes.cls.cpu().numpy().astype(int)  # Get class IDs
                    
                    if len(confidences_t) > 0:
                        # Find the index of the maximum confidence
                        max_conf_index = confidences_t.argmax()
                        
                        # Extract the bounding box, confidence, and class ID of the highest confidence detection
                        best_bbox = bboxes_t[max_conf_index]
                        best_confidence = confidences_t[max_conf_index]
                        best_class_id = class_ids_t[max_conf_index]

                        x1a, y1a, x2a, y2a = map(int, best_bbox)

                        # Set bounding box color based on the class ID
                        if best_class_id == 0:  # Class 0: Needle
                            color = (0, 0, 255)  # Red
                            labels = "Gear"
                            label11.configure(text="Gear Detected",bg='green' , fg='white')
                            detection_array_gear.append(1)
                           
                            

                        # Combine label with confidence score
                            label_with_confidence = f"{labels} ({best_confidence:.2f})"
                            
                            # Draw the bounding box
                            cv2.rectangle(frame, (x1a, y1a), (x2a, y2a), color, 2)
                            
                            # Draw the label and confidence score
                            cv2.putText(frame, label_with_confidence, (x1a, y1a - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            if (sum(detection_array_gear)<15 and aa==1) and cc==0:
                               label11.configure(text=f"Gear Detecting: {sum(detection_array_gear)}" ,bg='red' , fg='white')
                               #label1.configure(text=f"Needle Detecting: {sum(detection_array_needle)}",bg='red' , fg='white')

                            elif (sum(detection_array_gear)>=15 and aa==1) or cc==1:
                               label112.configure(text="Both Detected (YES SIGNAL)" ,bg='green' , fg='white')
                               turn_on()
                              # counti=counti+1
                               cc=1

                    else:
                        detection_array_gear.append(0)
                        if cc==0: 
                            label11.configure(text="Gear not Detected",bg='red' , fg='white')
                        if cc==1:
                           label11.configure(text="Gear Already Detected" ,bg='green' , fg='white')

                    if cc==1:###initialise
                        cou1=cou1+1
                        if cou1>=7:
                            detection_array_needle.clear() 
                            detection_array_gear.clear()                             
                            turn_off()
                            cou1=0
                            aa=0
                            cc=0

                            label112.configure(text=f"Waiting for next assembly",bg='yellow' , fg='Black')    
                            #label1.configure(text=f"Needle Detecting: {sum(detection_array_needle)}",bg='red' , fg='white') 
                    if len(detection_array_gear) > 40:  # Ensure the array size stays at 20
                        detection_array_gear.pop(0)  

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame)
                    imgtk = ImageTk.PhotoImage(image=img)
                    label.imgtk = imgtk
                    label.configure(image=imgtk)

                frame_count += 1
            else:
                if  not reconnect_attempted:
                    print(f"Camera {camera_index} not accessible due to initial connection error")
                    break
                else:
                    print(f"Camera {camera_index} not accessible due to reconnection error")
                    cap.release()
                
        cap.release()       
    # Start threads for each camera feed
    root.mainloop()
from tkinter import font
big_font = font.Font(family="Helvetica", size=30, weight="bold")  # You can customize family, size, and style

# Add a label with the big font
label = Label(root, text="Trial period is over!", font=big_font)
label.pack(pady=20)
root.mainloop()
