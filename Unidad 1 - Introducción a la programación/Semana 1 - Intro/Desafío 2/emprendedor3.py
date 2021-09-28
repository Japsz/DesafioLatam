import sys

precioVenta = int(sys.argv[1])
users = int(sys.argv[2])
gastos = int(sys.argv[3])
if(len(sys.argv) >= 5):
    prevUtilidades = int(sys.argv[4])
else:
    prevUtilidades = 1000

print(prevUtilidades)

utilidades = (precioVenta*users) - gastos

print("Las utilidades actuales son de: " + str(utilidades))
print("La razon de las utilidades respecto al año anterior es de: " + str(float(utilidades/prevUtilidades)))