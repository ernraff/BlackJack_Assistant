

# Import necessary packages
import numpy as np
import cv2
import time
from enum import Enum

class Thresholds(Enum):
    BKG_THRESH = 60
    CARD_THRESH = 30

class CardDimensions(Enum):
    CORNER_WIDTH = 45
    CORNER_HEIGHT = 130
    RANK_WIDTH = 70
    RANK_HEIGHT = 125

class MaxDifferences(Enum):
    RANK_DIFF_MAX = 2000
    CARD_MAX_AREA = 120000
    CARD_MIN_AREA = 2500


font = cv2.FONT_HERSHEY_SIMPLEX

### Structures to hold query card and train card information ###

class Query_card:
    """Structure to store information about query cards in the camera image."""

    def __init__(self):
        self.contour = [] # Contour of card
        self.width, self.height = 0, 0 # Width and height of card
        self.corner_pts = [] # Corner points of card
        self.center = [] # Center point of card
        self.warp = [] # 200x300, flattened, grayed, blurred image
        self.rank_img = [] # Thresholded, sized image of card's rank
        self.best_rank_match = 0 # Best matched rank
        self.rank_diff = 0 # Difference between rank image and best matched train rank image
class Train_ranks:
    """Structure to store information about train rank images."""

    def __init__(self):
        self.img = [] # Thresholded, sized rank image loaded from hard drive
        self.name = "Placeholder"


### Functions ###
def load_ranks(filepath):
    """This function loads images representing ranks from a directory specified by a file path.
    It then organizes these images into a list of objects belonging to the Train_ranks class"""

    train_ranks = []
    i = 0
    
    for Rank in ['Ace','Two','Three','Four','Five','Six','Seven',
                 'Eight','Nine','Ten','Jack','Queen','King']:
        train_ranks.append(Train_ranks())
        train_ranks[i].name = Rank
        filename = Rank + '.jpg'
        train_ranks[i].img = cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE)
        i = i + 1

    return train_ranks

def preprocess_image(image):
    """Returns a grayed, blurred, and adaptively thresholded camera image."""

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)

    # The optimal threshold level varies based on the surrounding lighting conditions.
    # In bright environments, a higher threshold is necessary to distinguish cards from the background,
    # while dim lighting requires a lower threshold.
    # To ensure the card detector functions consistently regardless of lighting,
    # an adaptive thresholding approach is employed.
    # This method involves sampling the intensity of a background pixel at the center top of the image
    # and setting the adaptive threshold slightly higher (50 units) than that intensity value.
    # This adjustment enables the threshold to adapt to changes in lighting conditions.

    img_w, img_h = np.shape(image)[:2]
    bkg_level = gray[int(img_h/100)][int(img_w/2)]
    thresh_level = bkg_level + Thresholds.BKG_THRESH.value

    retval, thresh = cv2.threshold(blur,thresh_level,255,cv2.THRESH_BINARY)
    
    return thresh

def find_cards(thresh_image):
    """Find all contours in a thresholded camera image that are approximately the size of cards.
   Return the count of cards and a list of card contours sorted from largest to smallest."""


    # Find contours and sort their indices by contour size
    cnts,hier = cv2.findContours(thresh_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(cnts)), key=lambda i : cv2.contourArea(cnts[i]),reverse=True)

    # If there are no contours, do nothing
    if len(cnts) == 0:
        return [], []
    
    # Otherwise, initialize empty sorted contour and hierarchy lists
    cnts_sort = []
    hier_sort = []
    cnt_is_card = np.zeros(len(cnts),dtype=int)

    # Populate empty lists with sorted contour and sorted hierarchy. Now,
    # the indices of the contour list still correspond with those of
    # the hierarchy list. The hierarchy array can be used to check if
    # the contours have parents or not.

    for i in index_sort:
        cnts_sort.append(cnts[i])
        hier_sort.append(hier[0][i])

    # Identify contours as potential cards based on the following conditions:
    # 1) Their area is smaller than the maximum card size,
    # 2) Their area is larger than the minimum card size,
    # 3) They have no parent contours, and 4) They possess four corners.

    for i in range(len(cnts_sort)):
        size = cv2.contourArea(cnts_sort[i])
        peri = cv2.arcLength(cnts_sort[i],True)
        approx = cv2.approxPolyDP(cnts_sort[i],0.01*peri,True)
        
        if ((size < MaxDifferences.CARD_MAX_AREA.value) and (size > MaxDifferences.CARD_MIN_AREA).value
            and (hier_sort[i][3] == -1) and (len(approx) == 4)):
            cnt_is_card[i] = 1

    return cnts_sort, cnt_is_card

def preprocess_card(contour, image):
    """Uses contour to find information about the query card. Isolates rank images from the card."""

    # Initialize new Query_card object
    qCard = Query_card()

    qCard.contour = contour

    # Find perimeter of card and use it to approximate corner points
    peri = cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,0.01*peri,True)
    pts = np.float32(approx)
    qCard.corner_pts = pts

    # Find width and height of card's bounding rectangle
    x,y,w,h = cv2.boundingRect(contour)
    qCard.width, qCard.height = w, h

    # Find center point of card by taking x and y average of the four corners.
    average = np.sum(pts, axis=0)/len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])
    qCard.center = [cent_x, cent_y]

    # Warp card into 200x300 flattened image using perspective transform
    qCard.warp = flattener(image, pts, w, h)

    # Grab corner of warped card image and do a 4x zoom
    Qcorner = qCard.warp[0:CardDimensions.CORNER_HEIGHT.value, 0:CardDimensions.CORNER_WIDTH.value]
    Qcorner_zoom = cv2.resize(Qcorner, (0,0), fx=4, fy=4)

    # Sample known white pixel intensity to determine good threshold level
    white_level = Qcorner_zoom[15,int((CardDimensions.CORNER_WIDTH.value*4)/2)]
    thresh_level = white_level - Thresholds.CARD_THRESH.value
    if (thresh_level <= 0):
        thresh_level = 1
    retval, query_thresh = cv2.threshold(Qcorner_zoom, thresh_level, 255, cv2.THRESH_BINARY_INV)
    
    # Split in to top and bottom half (top shows rank, bottom shows suit)
    Qrank = query_thresh[20:250, 0:250]

    cv2.imshow('Rank', Qrank)
    # Find rank contour and bounding rectangle, isolate and find largest contour
    Qrank_cnts, hier = cv2.findContours(Qrank, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    Qrank_cnts = sorted(Qrank_cnts, key=cv2.contourArea,reverse=True)

    # Find bounding rectangle for largest contour, use it to resize query rank
    # image to match dimensions of the train rank image
    if len(Qrank_cnts) != 0:
        x1,y1,w1,h1 = cv2.boundingRect(Qrank_cnts[0])
        Qrank_roi = Qrank[y1:y1+h1, x1:x1+w1]
        Qrank_sized = cv2.resize(Qrank_roi, (CardDimensions.RANK_WIDTH.value,CardDimensions.RANK_HEIGHT.value), 0, 0)
        qCard.rank_img = Qrank_sized

    return qCard

def match_card(qCard, train_ranks):
    """Identifies the optimal rank  matches for the query card by comparing the rank  images of the query card with those of the train rank images. T
    he most suitable match is determined by selecting the rank or suit image with the lowest difference."""
    rank_value = {
        "Ace" : 11,
        "Two" : 2,
        "Three" : 3,
        "Four" : 4,
        "Five" : 5,
        "Six" : 6,
        "Seven" : 7,
        "Eight" : 8,
        "Nine" : 9,
        "Ten" : 10,
        'Jack' : 10,
        'Queen' : 10,
        'King' : 10,
        'Unknown' : 0
    }

    best_rank_match_diff = 10000
    best_rank_match_name = 0
    i = 0

    # If no contours were found in query card in preprocess_card function,
    # the img size is zero, so skip the differencing process
    # (card will be left as Unknown)
    if len(qCard.rank_img) != 0:

        # Calculate the difference between the rank image of the query card and each of the train rank images,
        # and retain the result with the minimum difference.

        for Trank in train_ranks:

                diff_img = cv2.absdiff(qCard.rank_img, Trank.img)
                rank_diff = int(np.sum(diff_img)/255)
                
                if rank_diff < best_rank_match_diff:
                    best_rank_diff_img = diff_img
                    best_rank_match_diff = rank_diff
                    best_rank_name = rank_value[Trank.name]


    # Combine best rank match and best suit match to get query card's identity.
    # If the best matches have too high of a difference value, card identity
    # is still Unknown
    if (best_rank_match_diff < MaxDifferences.RANK_DIFF_MAX.value):
        best_rank_match_name = best_rank_name

    # Return the identiy of the card and the quality of the suit and rank match
    return best_rank_match_name,best_rank_match_diff
    
    
def draw_results(image, qCard):
    """Draw the card name, center point, and contour on the camera image."""
    x = qCard.center[0]
    y = qCard.center[1]

    cv2.circle(image,(x,y),5,(255,0,0),-1)

    rank_name = qCard.best_rank_match

    # Draw card name twice, so letters have black outline
    cv2.putText(image,(str(rank_name)),(x-60,y-10),font,1,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,(str(rank_name)),(x-60,y-10),font,1,(50,200,200),2,cv2.LINE_AA)

    return image

def draw_hand(image, cards):
    hand_total = 0
    for card in cards:
        hand_total += card.best_rank_match

    if hand_total > 21:
        while hand_total > 21:
            aceChanged = False
            for card in cards:
                if card.best_rank_match == 11:
                    card.best_rank_match = 1
                    hand_total -= 10
                    aceChanged = True
                    break
            if not aceChanged:
                cv2.putText(image,(f"You Lose!"),(800, 100),font,1,(0,0,0),3,cv2.LINE_AA)
                cv2.putText(image,(f"You Lose!"),(800, 100),font,1,(50,200,200),2,cv2.LINE_AA)
                break
    if len(cards) >= 5 and hand_total <= 21:
        cv2.putText(image,(f"Suggestion: Stay"),(800, 100),font,1,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(image,(f"Suggestion: Stay"),(800, 100),font,1,(50,200,200),2,cv2.LINE_AA)
    elif hand_total < 17:
        cv2.putText(image,(f"Suggestion: Hit"),(800, 100),font,1,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(image,(f"Suggestion: Hit"),(800, 100),font,1,(50,200,200),2,cv2.LINE_AA)
    elif hand_total >= 17 and hand_total <= 21:
        cv2.putText(image,(f"Suggestion: Stay"),(800, 100),font,1,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(image,(f"Suggestion: Stay"),(800, 100),font,1,(50,200,200),2,cv2.LINE_AA)

    cv2.putText(image,(f"Current hand: {hand_total}"),(350, 100),font,1,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,(f"Current hand: {hand_total}"),(350, 100),font,1,(50,200,200),2,cv2.LINE_AA)
    return image



def flattener(image, pts, w, h):
    """Transforms an image of a card into a top-down perspective with dimensions of 200x300.
Provides the transformed, resized, and grayscale image.
Refer to www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/ for more details."""

    temp_rect = np.zeros((4,2), dtype = "float32")
    
    s = np.sum(pts, axis = 2)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis = -1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    # Need to create an array listing points in order of
    # [top left, top right, bottom right, bottom left]
    # before doing the perspective transform

    if w <= 0.8*h: # If card is vertically oriented
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2*h: # If card is horizontally oriented
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    # If the card is 'diamond' oriented, a different algorithm
    # has to be used to identify which point is top left, top right
    # bottom left, and bottom right.
    
    if w > 0.8*h and w < 1.2*h: #If card is diamond oriented
        # If furthest left point is higher than furthest right point,
        # card is tilted to the left.
        if pts[1][0][1] <= pts[3][0][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1][0] # Top left
            temp_rect[1] = pts[0][0] # Top right
            temp_rect[2] = pts[3][0] # Bottom right
            temp_rect[3] = pts[2][0] # Bottom left

        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        if pts[1][0][1] > pts[3][0][1]:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0][0] # Top left
            temp_rect[1] = pts[3][0] # Top right
            temp_rect[2] = pts[2][0] # Bottom right
            temp_rect[3] = pts[1][0] # Bottom left
            
        
    maxWidth = 200
    maxHeight = 300

    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)

    return warp
