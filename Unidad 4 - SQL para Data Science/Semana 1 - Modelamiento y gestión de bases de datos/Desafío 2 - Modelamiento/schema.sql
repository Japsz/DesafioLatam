-- Crear una BD
CREATE DATABASE spotify;
\c spotify
-- Crear las tablas
CREATE TABLE artista (
    nombre_artista varchar(255) PRIMARY KEY,
    fecha_de_nacimiento date NOT NULL,
    nacionalidad varchar(255) NOT NULL
);
CREATE TABLE album (
    titulo_album varchar(255) PRIMARY KEY,
    anio int NOT NULL,
    artista varchar(255) NOT NULL,
    FOREIGN KEY (artista) REFERENCES artista (nombre_artista)
);
CREATE TABLE cancion (
    titulo_cancion varchar(255) PRIMARY KEY,
    album varchar(255) NOT NULL,
    artista varchar(255) NOT NULL,
    numero_del_track int NOT NULL,
    FOREIGN KEY (album) REFERENCES album (titulo_album),
    FOREIGN KEY (artista) REFERENCES artista (nombre_artista)
);