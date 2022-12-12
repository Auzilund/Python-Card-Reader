import cv2
import numpy as np
import os
import imutils

# Load Rank and suit Trian images
path = os.path.dirname(os.path.abspath(__file__))
train_ranks = []
train_suits = []
rank_names = ['Ace','Two','Three','Four','Five','Six','Seven',
                'Eight','Nine','Ten','Jack','Queen','King']
suit_names = ['Spades','Clubs','Diamonds','Hearts']

for i in range(13):
    filename = rank_names[i] + '.jpg'
    image = cv2.imread(path + '/Card_Imgs/' + filename, cv2.IMREAD_GRAYSCALE)
    train_ranks.append(image)

for i in range(4):
    filename = suit_names[i] + '.jpg'
    image = cv2.imread(path + '/Card_Imgs/' + filename, cv2.IMREAD_GRAYSCALE)
    train_suits.append(image)

cap = cv2.VideoCapture(0)

while(True):

    ret, frame = cap.read()

    #Modify the frames.
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # blur = cv2.bilateralFilter(gray, 5, 10, 10)
    # blur = cv2.GaussianBlur(gray,(5,5),0)

    img_w, img_h = np.shape(image)[:2]
    bkg_level = gray[int(img_h/100)][int(img_w/2)]
    thresh_level = bkg_level + 70
    retval, thresh = cv2.threshold(gray, thresh_level, 255, cv2.THRESH_BINARY)

    #Find the cards contours
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(contours) > 2:
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)    
        
        if w > h > 0:
            ar = w / float(h) 
        elif h > w > 0:
            ar = h / float(w)
        else:
            ar = 0
        # This value is used to filter the contours to a specific aspect ratio
        # that fits the card contour aspect ration give or take a few decimals.
        if 1.20 <= ar <= 1.50:
            cv2.drawContours(frame, [c], -1, (219,231,239), 2)
            temp_rect = np.zeros((4,2), dtype = "float32")
            peri = cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c,0.01*peri,True)
            pts = np.float32(approx)
        
            s = np.sum(pts, axis = 2)

            tl = pts[np.argmin(s)]
            br = pts[np.argmax(s)]

            diff = np.diff(pts, axis = -1)
            tr = pts[np.argmin(diff)]
            bl = pts[np.argmax(diff)]

            if w <= 0.8*h:
                temp_rect[0] = tl
                temp_rect[1] = tr
                temp_rect[2] = br
                temp_rect[3] = bl

            if w >= 1.2*h:
                temp_rect[0] = bl
                temp_rect[1] = tl
                temp_rect[2] = tr
                temp_rect[3] = br

            if w > 0.8*h and w < 1.2*h:
                if pts[1][0][1] <= pts[3][0][1]:
                    temp_rect[0] = pts[1][0]
                    temp_rect[1] = pts[0][0]
                    temp_rect[2] = pts[3][0]
                    temp_rect[3] = pts[2][0]
                if pts[1][0][1] > pts[3][0][1]:
                    temp_rect[0] = pts[0][0]
                    temp_rect[1] = pts[3][0]
                    temp_rect[2] = pts[2][0]
                    temp_rect[3] = pts[1][0]
            maxWidth = 200
            maxHeight = 300
            dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
            M = cv2.getPerspectiveTransform(temp_rect,dst)
            warp = cv2.warpPerspective(thresh, M, (maxWidth, maxHeight))
            cv2.imshow('Card', warp)

            # Isolate suit and rank
            corner = warp[0:84, 0:32]
            suit_rank = cv2.resize(corner, (0,0), fx=4, fy=4)
            cv2.imshow('Suit and Rank', suit_rank)

            # Process suit and rank thresh
            white_level = suit_rank[15,int((32*4)/2)]
            thresh_level = white_level - 30
            if (thresh_level <= 0):
                thresh_level = 1
            retval, suit_rank = cv2.threshold(suit_rank, thresh_level, 255, cv2. THRESH_BINARY_INV)

            # Isolate suit/rank into seperate image values
            rank = suit_rank[20:185, 0:128]
            suit = suit_rank[180:336, 0:128]
            rank = cv2.GaussianBlur(rank,(5,5),0)
            cv2.imshow('CardCorner Rank', rank)
            cv2.imshow('CardCorner Suit', suit)
            
            # Contour the rank to match train image
            rank_cnts, hier = cv2.findContours(rank, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            rank_cnts = sorted(rank_cnts, key=cv2.contourArea,reverse=True)
            rank_sized = 0
            if len(rank_cnts) != 0:
                x1,y1,w1,h1 = cv2.boundingRect(rank_cnts[0])
                rank_roi = rank[y1:y1+h1, x1:x1+w1]
                rank_sized = cv2.resize(rank_roi, (70,125), 0, 0)
                cv2.imshow('Rank', rank_sized)

            suit_cnts, hier = cv2.findContours(suit, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            suit_cnts = sorted(suit_cnts, key=cv2.contourArea,reverse=True)
            suit_sized = 0
            if len(suit_cnts) != 0:
                x1,y1,w1,h1 = cv2.boundingRect(suit_cnts[0])
                suit_roi = suit[y1:y1+h1, x1:x1+w1]
                suit_sized = cv2.resize(suit_roi, (70,100), 0, 0)
                cv2.imshow('Suit', suit_sized)


            best_rank_match_diff = 10000
            best_suit_match_diff = 10000
            best_rank_match_name = "Unknown"
            best_suit_match_name = "Unknown"

            i = 0
            for Crank in train_ranks:
                diff_img = cv2.absdiff(rank_sized, Crank)
                rank_diff = int(np.sum(diff_img)/255)
                if rank_diff < best_rank_match_diff:
                    best_rank_diff_img = diff_img
                    best_rank_match_diff = rank_diff
                    best_rank_name = rank_names[i]
                    cv2.imshow('rank diff', diff_img)
                    if i < len(rank_names) -1:
                        i = i+1
                else:
                    if i < len(rank_names) -1:
                        i = i+1
            i = 0
            for Csuit in train_suits:
                diff_img = cv2.absdiff(suit_sized, Csuit)
                suit_diff = int(np.sum(diff_img)/255)
                if suit_diff < best_suit_match_diff:
                    best_suit_diff_img = diff_img
                    best_suit_match_diff = suit_diff
                    best_suit_name = suit_names[i]
                    cv2.imshow('suit diff', diff_img)
                    if i < len(suit_names)-1:
                        i = i+1
                else:
                    if i < len(suit_names)-1:
                        i = i+1
            if (best_rank_match_diff < 2000):
                best_rank_match_name = best_rank_name

            if (best_suit_match_diff < 1000):
                best_suit_match_name = best_suit_name

            # Calc center of card abd draw a dot and name of card
            average = np.sum(pts, axis=0)/len(pts)
            cent_x = int(average[0][0])
            cent_y = int(average[0][1])
            center = [cent_x, cent_y]
            font = cv2.FONT_HERSHEY_SIMPLEX
            x = center[0]
            y = center[1]
            cv2.circle(frame,(x,y),5,(219,231,239),-1)
            cv2.putText(frame,(best_rank_match_name+' of'),(x-60,y-10),font,1,(190,231,239),3,cv2.LINE_AA)
            cv2.putText(frame,best_suit_match_name,(x-60,y+25),font,1,(190,231,239),3,cv2.LINE_AA)

    cv2.imshow('Camera View', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()