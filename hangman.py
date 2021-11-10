##  In this project I choose to add up the score based on the amount of
##  guesses made by the user not just the guesses which the user got correctly
##  which is more in tune with how the game is played.
##
##  Just to note:   guessed_letter and guessed_letters are two different
##                  variables

import random           # For random number generationn
import re               # For regular expression functionality
import string           # For ensuring all the imputs are lowercase letters
import numpy as np

words = [] # list with all the words in the file
with open("words.txt") as read_file:
    for line in read_file:
        words.append(line.rstrip("\n"))

words = random.choice(words) # Picks a random word from the list

# Encode the word choosen with dashes
encoded_word = re.sub('[a-zA-Z]', '_', words)

# A function for handling guesses
def guess(letter, words, encoded):
    # Does the letter exist within the word?
    found = False
    letter = letter.lower() # Ensures the ubput is a lowercase letter

    lc_alphabet = set(string.ascii_lowercase)   # array that contains all the
                                                # letters in the alphabet

    # Check to see if there input letter is valid
    if letter in lc_alphabet:
        letter = letter
        #print("Success, input is a single letter from the alphabet")
    else:
        print("Input is not in the alphabet, therefore is invalid")

    if letter in words:
        found = True
        # Replace the dashes with the letter
        for i in range(0, len(words)):
            if words[i] == letter:
                encoded = encoded[0:i] + letter + encoded[i+1:len(encoded)]
    return (found, encoded)


print("The word is: ", encoded_word)

# Required variables
guessed_letters = []      # List of letters guessed by the user
score = 0                 # Number of guesses
found = False             # Keep track if the word is found

#########################################################################
##
##      if '_' found in encoded word
##
##          if guessed letter in choosen word
##              if guessed letter not in guessed letters
##                  score = score + 1
##                  guessed letter is added to guessed guessed_letters
##                  print("The word is: ", encoded_word)
##              else
##                  if guessed letter in guessed letters
##                      score is unchanged
##                      print("You have already guessed that letter!")
##          else
##              if guessed letter not in choosen words
##                  score = score + 1
##                  print("Wrong guess: ? is not a letter in the word.")
##      else
##          if '_' found not in encoded word
##              print("The word is found")
##              print the score and the uncovered word
##              Tell the program that the word was found so the program will end
##
#############################################################################

while not found:
    found = False           # States that the word is not found
    guessed_letter = input("Guess one letter in the word: ")[:1]
    ## [:1] ensures the input is a single letter, the program will
    ## choose the first letter given

    letter_found, encoded_word = guess(guessed_letter, words, encoded_word)

    if "_" in encoded_word:
        if guessed_letter in encoded_word:
            if guessed_letter not in guessed_letters:
                score = score + 1
                letter_found = guessed_letter
                guessed_letters.append(letter_found)
                #print("Guessed letters: ", guessed_letters) # Checking to see what letters have been guessed
                #print("Your score is ", score)    # Accumulatting the overall score
                print("The word is: ", encoded_word)
                print(" ") # Create a space in between the lines
            else:
                if guessed_letter in guessed_letters:
                    print("You have already guessed that letter!")
                    print(" ") # Create a space in between the lines
                    score = score    ## score is unchanged
        else:
            if guessed_letter not in encoded_word:
                score = score + 1
                #print("Your score is ", score)
                print("Wrong guess: ", guessed_letter, " is not a letter in the word.")
                print(" ") # Create a space in between the lines
    else:
        if "_" not in encoded_word:
            print(" ") # Create a space in between the lines
            print("The Word is found")
            print("Your score is ", score)
            print("The word is: ", encoded_word)
            found = True        # States that the word was found
