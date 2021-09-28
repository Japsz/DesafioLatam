from string import ascii_lowercase

password = str(input("Ingresa contraseña\n"))

totalIntentos = 0

for char in password:
    for letra in ascii_lowercase:
        totalIntentos += 1
        if(letra == char.lower()):
            break

print(str(totalIntentos) + ' intentos')