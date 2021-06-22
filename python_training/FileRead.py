with open('words.txt', 'r') as file:
    count = 0
    for word in file:
        if len(word.strip('\n')) <= 10:
            count+=1

    print(count)