import cv2
import os
import pandas as pd

path = "E:/Bibliotheken/Desktop/Gianna_macht_Sachen/Uni/Bachelorarbeit"
file = "StressTracking.mp4"
data = os.path.join(path, file)
cap = cv2.VideoCapture(data)

p_coord_list = []
o_coord_list = []

# Load Templates + define width & height
purple_pick = cv2.imread('purple_pick.png', cv2.IMREAD_UNCHANGED)
p_templ_w = purple_pick.shape[0]
p_templ_h = purple_pick.shape[1]
orange_pick = cv2.imread('orange_pick.png', cv2.IMREAD_UNCHANGED)
o_templ_w = orange_pick.shape[0]
o_templ_h = orange_pick.shape[1]

while True:
    # Capture frame
    found_image, frame = cap.read()
    if found_image == False:
        continue
    # Wait 1ms (0ms) between each frame until Esc is pressed
    key = cv2.waitKey(1)
    if key == 27:
        break   

    # Compare Dot_Img's to Frame
    p_result = cv2.matchTemplate(frame, purple_pick, cv2.TM_SQDIFF_NORMED)
    o_result = cv2.matchTemplate(frame, orange_pick, cv2.TM_SQDIFF_NORMED)

    # Get Max Results
    p_min_val, p_max_val, p_min_loc, p_max_loc = cv2.minMaxLoc(p_result)
    o_min_val, o_max_val, o_min_loc, o_max_loc = cv2.minMaxLoc(o_result)

    # Find Center of Dots as Coordinates
    p_coord = ((p_min_loc[0] + p_templ_w // 2), (p_min_loc[1] + p_templ_h // 2))
    o_coord = ((o_min_loc[0] + o_templ_w // 2), (o_min_loc[1] + o_templ_h // 2))

    # Save Coordinates in Lists
    p_coord_list.append(p_coord)
    o_coord_list.append(o_coord)

    # Show Result + Frames
    #cv2.imshow('Purple Result', p_result)
    #cv2.imshow('Orange Result', o_result)
    cv2.imshow('Frame', frame)
    
cap.release()

# Convert Lists to CSV (Index = False macht Nummerierung)
Dot_coord = pd.DataFrame({"Purple Coordinates": p_coord_list,
                          "Orange Coordinates": o_coord_list})
Dot_coord.to_csv('Dot Coordinates.csv', index = False, sep = ";")
