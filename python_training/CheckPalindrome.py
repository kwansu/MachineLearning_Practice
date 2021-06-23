with open('words.txt', 'r') as file:
    for word in file:
        word = word.strip('\n')
        if word == word[::-1]:
            print(word)