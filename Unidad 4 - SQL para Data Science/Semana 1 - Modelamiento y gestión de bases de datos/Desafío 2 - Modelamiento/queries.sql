-- Seleccionar canciones del 2018
SELECT cancion.* from cancion INNER JOIN album ON cancion.album = album.titulo_album WHERE album.anio = 2018;
-- Seleccionar albumes y nacionalidad
SELECT album.*, artista.nacionalidad FROM album LEFT JOIN artista ON album.artista = artista.nombre_artista;
-- Seleccionar canciones ordenados segun año, album y artista
SELECT cancion.*, album.anio FROM cancion INNER JOIN album ON cancion.album = album.titulo_album INNER JOIN artista ON album.artista = artista.nombre_artista ORDER BY album.anio, album.titulo_album, artista.nombre_artista;