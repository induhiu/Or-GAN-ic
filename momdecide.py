""" Attempt to implement the "mom" language consensus without using a neural
network. Implementation by Kenny Talarico, May 2019. """

import mom
from random import choice
from secrets import randbelow

alphabet = ['0', '1']
signs = ['kenny', 'ian', 'dave', 'decker']
def talk(mom1, mom2):
    # these will switch each iteration
    speaker = mom1
    listener = mom2

    # main loop
    while mom1.dictionary != mom2.dictionary:

        # to ensure that there are no duplicate words
        # for k in mom1.dictionary:
        #     for l in mom2.dictionary:
        #         while k == l and mom1.dictionary[k] != mom1.dictionary[l]:
        #             change = choice([(mom1, k), (mom2, l)])
        #             change[0].dictionary[change[1][:randbelow(len(change[1]))] +
        #                                  choice(alphabet) +
        #                                  change[1][randbelow(len(change[1]))+1:]] = \
        #                                  change[0].dictionary.pop(change[1])


        mystery = speaker.speak()
        guess = listener.guess(mystery)

        # if the listener guesses correctly
        if listener.dictionary[guess] == speaker.dictionary[mystery]:
            # do some work to merge their words for the same thing
            newmystery = mystery
            newguess = guess
            for ch in range(len(mystery)):
                if newmystery[ch] != newguess[ch]:
                    if choice([speaker, listener]) == speaker:
                        newmystery = newmystery[:ch] + guess[ch] + newmystery[ch+1:]
                    else:
                        newguess = newguess[:ch] + mystery[ch] + newguess[ch+1:]
            speaker.dictionary[newmystery] = speaker.dictionary.pop(mystery)
            listener.dictionary[newguess] = listener.dictionary.pop(guess)
        # otherwise, randomly change a letter in the guess. This adds
        # unpredictability
        else:
            ran = randbelow(len(guess))
            newguess = guess[:ran] + choice(alphabet) + guess[ran+1:]
            listener.dictionary[newguess] = listener.dictionary.pop(guess)

        # swap speaker/listener
        oldspeaker = speaker
        speaker = listener
        listener = oldspeaker

def main():
    mom1 = mom.Mom(alphabet, signs)
    mom2 = mom.Mom(alphabet, signs)

    displayDict = {mom1.dictionary[key]: (key, mom2.getValue(mom1.dictionary[key])) for key in mom1.dictionary}

    print(' ' * 15, "MOM 1", ' ' * 20, "MOM 2")

    for k in list(displayDict.keys()):
        print(k + ':' + (13 - len(k)) * ' ' + displayDict[k][0] +
              (27 - len(displayDict[k][0])) * ' ' + displayDict[k][1])

    print()
    print("Generating...")
    talk(mom1, mom2)
    print()
    print()
    print("---RESULTS---")

    for k in list(mom1.dictionary.keys()):
        print(mom1.dictionary[k] + ': ' + (15 - len(mom1.dictionary[k])) * ' ' + k)



if __name__ == '__main__':
    main()
