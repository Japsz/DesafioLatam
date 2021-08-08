import sys

precioVenta = int(sys.argv[1])
usuarios = int(sys.argv[2])
gastos = int(sys.argv[3])

utilidades = precioVenta*usuarios - gastos

print("Las utilidades son de " + str(utilidades))