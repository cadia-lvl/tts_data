if __name__ == '__main__':
    with open('small_1hour.txt') as f:
        for line in f:
            sentence, source, score, *transcription = line.split('\t')
            print(transcription, len(transcription))