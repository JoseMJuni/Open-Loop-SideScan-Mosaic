class Coord:
    def __init__(self, time, x, y, heading):
        self.time    = time
        self.x       = x
        self.y       = y
        self.heading = heading
        

    @staticmethod
    def ReadFromFile(filename):
        file = open(filename,"r+")
        temp = file.read().splitlines()
        resultCoord = []
        for line in temp:
            resultCoord.append(Coord(line.split()[0], line.split()[1], line.split()[2], line.split()[3]))
        return resultCoord