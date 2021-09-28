import sys
import math

gravedad = float(sys.argv[1])
radius = float(sys.argv[2])*1000

def escape(gravity, rad):
    return math.sqrt(2*gravity*radius)

print("La velocidad de escape es " + str(escape(gravedad, radius)) + " m/s")