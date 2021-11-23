import pandas as pd

insert_query = ""
# Se importan los csv
albums = pd.read_csv('Album.csv', encoding='latin-1')
artists = pd.read_csv('Artista.csv', encoding='latin-1')
songs = pd.read_csv('Cancion.csv', encoding='latin-1')

# Generar los queries para insertar los artistas
insert_query += "-- Artistas\n"
for index, row in artists.iterrows():
    artistQuery = "INSERT INTO artista (nombre_artista, fecha_de_nacimiento, nacionalidad) VALUES ('"
    artistQuery += str(row['nombre_artista']) + "', '" + row['fecha_de_nacimiento'] + "', '" + row['nacionalidad'] + "');\n"
    print(artistQuery)
    insert_query += artistQuery
# Generar los queries para insertar los albums
insert_query += "-- Albums\n"
for index, row in albums.iterrows():
    albumQuery = "INSERT INTO album (titulo_album, anio, artista) VALUES ('"
    albumQuery += str(row['titulo_album']) + "', " + str(row['anio']) + ", '" + str(row['artista']) + "');\n"
    print(albumQuery)
    insert_query += albumQuery
# Generar los queries para insertar las canciones
insert_query += "-- Canciones\n"
for index, row in songs.iterrows():
    songQuery = "INSERT INTO cancion (titulo_cancion, artista, numero_del_track, album) VALUES ('"
    songQuery += str(row['titulo_cancion']) + "', '" + row['artista'] + "', " + str(row['numero_del_track']) + ", '" + str(row['album']) + "');\n"
    print(songQuery)
    insert_query += songQuery
# Guardar la query en un archivo
with open('inserts.sql', 'w') as file:
    file.write(insert_query)
    file.close()
