class Velocity:
    def __init__(self, velx, vely, velz, velrot):
        self.velx    = velx
        self.vely    = vely
        self.velz    = velz
        self.velrot  = velrot
        

    @staticmethod
    def ReadFromFile(filename):
        file = open(filename,"r+")
        temp = file.read().splitlines()
        resultVelocity = []
        for line in temp:
            resultVelocity.append(Velocity(line.split()[0], line.split()[1], line.split()[2], line.split()[3]))
        return resultVelocity