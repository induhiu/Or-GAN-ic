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
        # print(mom1.dictionary)
        # print(mom2.dictionary)

        mystery = speaker.speak()
        guess = listener.guess(mystery)

        # print(listener.dictionary[guess])

        # print(listener.dictionary[guess], speaker.dictionary[mystery])

        if listener.dictionary[guess] == speaker.dictionary[mystery]:
            newmystery = mystery
            newguess = guess
            for ch in range(len(mystery)):
                if newmystery[ch] != newguess[ch]:
                    if choice([speaker, listener]) == speaker:
                        newmystery = newmystery[:ch] + guess[ch] + newmystery[ch+1:]
                    else:
                        newguess = newguess[:ch] + mystery[ch] + newguess[ch+1:]
            print(mom1.dictionary)
            print(mom2.dictionary)
            speaker.dictionary[newmystery] = speaker.dictionary.pop(mystery)
            listener.dictionary[newguess] = listener.dictionary.pop(guess)
        else:
            ran = randbelow(len(guess))
            #
            while guess[ran] == mystery[ran]:
                ran = randbelow(len(guess))
            newguess = guess[:ran] + mystery[ran] + guess[ran+1:]
            #
            # print(guess, 'changing to', newguess)

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
    #
    # word1 = 'abcdef'
    # word2 = 'fedcba'
    # for ch in range(len(word1)):
    #     while word1[ch] != word2[ch]:
    #         word1 = word1[:ch] + choice(alphabet) + word1[ch+1:]
    #         word2 = word2[:ch] + choice(alphabet) + word2[ch+1:]
    #
    # print(word1, word2)



if __name__ == '__main__':
    main()
