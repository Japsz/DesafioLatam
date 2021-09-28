paresHasta = int(input("Ingrese un número para sumar pares\n"))
i = 0
suma = 0
while i <= paresHasta:
    if i != 0: suma += i
    i += 2
print(suma)