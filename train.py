from Network import train

if __name__ == '__main__':
    train.run("DIR", epochs=60, config=0)
    #train.run("SIAM")
    #train.run("SIAM_RATING")
    #train.run("DIR_RATING")
    #train.run("TRIPLET")