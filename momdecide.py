""" Attempt to implement the "mom" language consensus without using a neural
network. Implementation by Kenny Talarico, May 2019. """

import mom
from random import choice
from secrets import randbelow

alphabet = ['a', 'b', 'c', 'd', 'e', 'f']
signs = ['kenny', 'ian', 'dave', 'decker']

def talk(mom1, mom2):
    speaker = mom1
    listener = mom2
    while mom1.dictionary != mom2.dictionary:
        mystery = speaker.speak()
        guess = listener.guess(mystery)
        if listener.dictionary[guess] == speaker.dictionary[mystery]:
            for ch in range(len(mystery)):
                while mystery[ch] != guess[ch]:
                    mystery = mystery[:ch] + choice(alphabet) + mystery[ch:]
                    guess = guess[:ch] + choice(alphabet) + mystery[ch:]
        else:
            ran = randbelow(len(guess))
            newguess = guess[:ran] + choice(alphabet) + guess[ran:]
            listener.dictionary[newguess] = listener.dictionary.pop(guess)
        oldspeaker = speaker
        speaker = listener
        listener = oldspeaker

def main():
    mom1 = mom.Mom(alphabet, signs)
    mom2 = mom.Mom(alphabet, signs)
    print(mom1.dictionary)
    print(mom2.dictionary)
    talk(mom1, mom2)
    print(mom1.dictionary)
    print(mom2.dictionary)


if __name__ == '__main__':
    main()
