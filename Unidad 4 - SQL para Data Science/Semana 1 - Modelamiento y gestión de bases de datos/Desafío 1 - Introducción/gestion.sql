-- 1 Crear Base de datos
CREATE DATABASE biblioteca;
-- 1.1 Conectar a Base de datos recién creada
\c biblioteca;
-- 2 Crear tabla Libro
CREATE TABLE IF NOT EXISTS libro (
    id_libro SERIAL CONSTRAINT libro_key PRIMARY KEY,
    nombre_libro varchar(255) NOT NULL,
    autor varchar(255) NOT NULL,
    genero varchar(255) NOT NULL
);
-- 3 Insertar dato en la tabla
INSERT INTO
    libro (nombre_libro, autor, genero)
VALUES
    ('Sapo y Sepo', 'Juan', 'Lovecraftiano');
-- 4 Insertar dato en la tabla
INSERT INTO
    libro (nombre_libro, autor, genero)
VALUES
    (
        'La Metamorfosis',
        'Franz Kafka',
        'Realismo magico'
    );
-- 5 Crear tabla Prestamo
CREATE TABLE IF NOT EXISTS prestamo (
    id_prestamo SERIAL CONSTRAINT prestamo_key PRIMARY KEY,
    id_libro SERIAL NOT NULL REFERENCES libro(id_libro),
    nombre_persona varchar(255) NOT NULL,
    fecha_inicio date NOT NULL,
    fecha_fin date NOT NULL
);
-- 6 Modificar tabla Libro
ALTER TABLE libro ADD COLUMN prestado boolean DEFAULT FALSE;
-- 7 Modificar datos en la tabla Libro
UPDATE libro SET prestado = TRUE WHERE id_libro = 1;
-- 8 Modificar datos en la tabla Libro
UPDATE libro SET prestado = TRUE WHERE id_libro = 2;
-- 9 Poblar tabla Prestamo
INSERT INTO prestamo (id_libro, nombre_persona, fecha_inicio, fecha_fin) VALUES (1, 'Juan Cena', '2020-01-01', '2020-01-10');
INSERT INTO prestamo (id_libro, nombre_persona, fecha_inicio, fecha_fin) VALUES (1, 'Elsa DeFrousen', '2020-01-15', '2020-01-24');
INSERT INTO prestamo (id_libro, nombre_persona, fecha_inicio, fecha_fin) VALUES (1, 'Delfin hastaelfin', '2020-02-10', '2020-03-01');
INSERT INTO prestamo (id_libro, nombre_persona, fecha_inicio, fecha_fin) VALUES (1, 'Puma del Occidente', '2020-09-01', '2020-10-10');
INSERT INTO prestamo (id_libro, nombre_persona, fecha_inicio, fecha_fin) VALUES (1, 'Seba Chizel', '2021-11-20', '2021-11-30');
-- 10 Poblar tabla Prestamo
INSERT INTO prestamo (id_libro, nombre_persona, fecha_inicio, fecha_fin) VALUES (2, 'Juan Cena', '2021-01-01', '2021-01-10');
INSERT INTO prestamo (id_libro, nombre_persona, fecha_inicio, fecha_fin) VALUES (2, 'Elsa DeFrousen', '2021-01-15', '2021-01-24');
INSERT INTO prestamo (id_libro, nombre_persona, fecha_inicio, fecha_fin) VALUES (2, 'Delfin hastaelfin', '2021-02-10', '2021-03-01');
INSERT INTO prestamo (id_libro, nombre_persona, fecha_inicio, fecha_fin) VALUES (2, 'Seba Chizel', '2021-09-01', '2021-10-10');
INSERT INTO prestamo (id_libro, nombre_persona, fecha_inicio, fecha_fin) VALUES (2, 'Puma del Occidente', '2021-10-11', '2021-10-20');
INSERT INTO prestamo (id_libro, nombre_persona, fecha_inicio, fecha_fin) VALUES (2, 'Arturo Vidal', '2021-11-20', '2021-11-30');
-- 11 Crear nuevo Libro
INSERT INTO libro (nombre_libro, autor, genero) VALUES ('La granja de los animales', 'George Orwell', 'Actualidad');
-- 12 Seleccionar datos de Libros
SELECT nombre_libro, autor, genero, nombre_persona FROM libro INNER JOIN prestamo ON libro.id_libro = prestamo.id_libro;
-- 13 Seleccionar prestamos de sapo y sepo
SELECT * FROM prestamo WHERE id_libro=1 ORDER BY fecha_inicio DESC;
