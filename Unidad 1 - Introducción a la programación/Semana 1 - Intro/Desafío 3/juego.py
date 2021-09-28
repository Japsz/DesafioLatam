import sys
import random

# se define un objeto con las debilidades
debilidades = {
    'tijera': 'piedra',
    'papel': 'tijera',
    'piedra': 'papel',
} 
# se define la jugada del pc
tiradaAi = random.choice(list(debilidades.keys()))
print('Computador juega ' + tiradaAi)

# se consigue y revisa si se la jugada es valida
eleccionJugador = sys.argv[1]
if eleccionJugador not in debilidades:
    print('Argumento invalido: Debe ser piedra, papel o tijera.')
    exit()

if eleccionJugador == tiradaAi : print('Empate') # Si tienen lo mismo, empatan
elif eleccionJugador == debilidades[tiradaAi] : print('Ganaste') # si el jugador tiro la debilidad del Computador, Gana
else : print('Perdiste')