import sys

usersNormales = int(sys.argv[1])
usersPremium = int(sys.argv[2])
usersFree = int(sys.argv[3])
precioVenta = int(sys.argv[4])
gastos = int(sys.argv[5])

utilidades = (precioVenta*usersNormales) + usersFree*0 + (precioVenta*2*usersPremium) - gastos

print("Las utilidades son de " + str(utilidades))