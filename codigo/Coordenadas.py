class Coord:
    def __init__(self, x, y):
        self.x  = x
        self.y  = y
        

    @staticmethod
    def ReadFromFile(filename):
        file = open(filename,"r+")
        temp = file.read().splitlines()
        resultCoord = []
        for line in temp:
            resultCoord.append(Coord(line.split()[0], line.split()[1]))
        return resultCoord

