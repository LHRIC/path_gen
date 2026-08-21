def tirParse(path,debug=False):
    '''Accepts .tir path, returns dict w/ param names and values'''
    tireDict = {}
    file = open(path, 'r')
    for line in file:
        if line[0] != "$" and line[0] != "[" and line[0] != "!" and line[0] != "{" and line[0] != " ": 
        
            key = line[0:line.find(" ")]
            noSpace = line.replace(" ", "")
            value = noSpace[noSpace.find("=")+1:noSpace.find("$")]

            try:
                tireDict[key] = float(value)
            except:
                tireDict[key] = value

            if debug == True:
                print("Key: " + key + "     Value: " + value + "     Type: " + str(type(tireDict[key])))
            
            #print(key + " = " + value)
    file.close()
    return tireDict

def tirWrite(dict, templatePath, writePath):
    '''Accepts params as dict, modifies .tir file from templatePath and writes to writePath'''
    with open(templatePath, 'r') as file:
        lines = file.readlines()

        keyList = list(dict.keys())
        for i, key in enumerate(keyList):
            for j, line in enumerate(lines):
                if key == line[0:line.find(" ")]:
                    comment = line[line.find("$"):len(line)-1]
                    lines[j] = str(key) + "                     =  " + str(dict[key]) + "          " + comment + "\n"

    with open(writePath, 'w+') as writeFile:
        for line in lines:
            writeFile.write(line)             

#tirParse('FSAE_Defaults.tir')

#replace = {"PCY1": 1, "PDY1": -2}
#pathIn = r"C:\Users\LegoE\OneDrive\Documents\04-LHR_Code\tire_modeling\tests\testIn.tir"
#pathOut = r"C:\Users\LegoE\OneDrive\Documents\04-LHR_Code\tire_modeling\tests\testOut.tir"
#tirWrite(replace,pathIn,pathOut)
