# OpenCV-Blackjack-Helper

This program is based on the following card detection program from EdjeElectronics: https://github.com/EdjeElectronics.

The program was meaningfully modified to fit our requirements, but the underlying logic is attributed to EdjeElectronics.
This is a Python program that uses OpenCV to detect and identify playing cards from a Camera video feed.

## Usage

Download this repository to a directory and run CardDetector.py from that directory. Cards need to be placed on a dark background for the detector to work. Press 'q' to end the program.

The program allows you to use a USB camera.

The card detector will work best if you use isolated rank images are generated from your own cards. To do this, run Rank_Suit_Isolator.py to take pictures of your cards. It will ask you to take a picture of an Ace, then a Two, and so on. Then, it will ask you to take a picture of one card from each of the suits (Spades, Diamonds, Clubs, Hearts). As you take pictures of the cards, the script will automatically isolate the rank or suit and save them in the Card_Imgs directory (overwriting the existing images).

## Files

CardDetector.py contains the main script

Cards.py has classes and functions that are used by CardDetector.py

PiVideoStream.py creates a video stream from USBCamera, and is used by CardDetector.py

Rank_Suit_Isolator.py is a standalone script that can be used to isolate the rank and suit from a set of cards to create train images

Card_Imgs contains all the train images of the card ranks and suits

## Dependencies

Python 3.6

OpenCV-Python 3.2.0 and numpy 1.8.2:
