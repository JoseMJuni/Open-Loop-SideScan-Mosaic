class Pose:
    def __init__(self, pitch, roll, yaw):
        self.pitch  = pitch
        self.roll  = roll
        self.yaw  = yaw
        

    @staticmethod
    def ReadFromFile(filename):
        file = open(filename,"r+")
        temp = file.read().splitlines()
        resultPose = []
        for line in temp:
            resultPose.append(Pose(line.split()[0], line.split()[1], line.split()[2]))
        return resultPose