import json as JSON

SEPARATOR = "<--- SEP --->"

class Inertial:
    def __init__(self, euler, acc, gyr, mag, heading):
        self.euler  = euler
        self.acc    = acc
        self.gyr    = gyr
        self.mag    = mag
        self.heading= heading
        
    @staticmethod
    def __JsonParse(jsonstr):
        data = JSON.loads(jsonstr)
        sensorBuffer = []
        heading = data['heading']
        for sensor in data['inertial']:
            for axis in data['inertial'][sensor]:
                sensorBuffer.append(data['inertial'][sensor][axis])
        result = Inertial(sensorBuffer[0:3], sensorBuffer[3:6],sensorBuffer[6:9],sensorBuffer[9:12],heading)
        return result
        
    @staticmethod
    def ReadFromFile(filename):
        file = open(filename,"r+")
        temp = file.read().splitlines()
        resultInertial = []
        jsonBlock = ""
        for line in temp:
            if line == SEPARATOR:
                if jsonBlock != "":
                    resultInertial.append(Inertial.__JsonParse(jsonBlock))
                    jsonBlock = ""
                continue
            else:
                jsonBlock = jsonBlock+str(line)
                jsonBlock = str(jsonBlock).replace("'",'"')
        return resultInertial



            
       
    
        


#bufferInertial = Inertial.ReadFromFile("../recursos/datos/sibiu-pro-carboneras-anforas-2.jdb.salida")
#print(bufferInertial[0].euler[0])