import sys

if len(sys.argv) != 4 :
    print('no son 3 numeros')
    exit()

def getFloat(str):
    return float(str)

print(max(map(getFloat, sys.argv[1:4])))
