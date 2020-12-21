
f'''
INSERT INTO Tabla_Placa(id_TipoAnalisis, id_TipoPatogeno) VALUES (@id_TipoAnalisis, @id_TipoPatogeno)
IF NOT EXISTS(SELECT PathImagen from Tabla_ImagenPlaca where PathImagen = @PathImagen)
BEGIN
	SET @id_Placa = (SELECT TOP 1 id_Placa from Tabla_Placa order by id_placa DESC)
	SET @id_EstadoPrediccion = 0
	INSERT INTO Tabla_ImagenPlaca(id_Placa,PathImagen) VALUES (@id_Placa,@PathImagen)
	INSERT INTO Tabla_Prediccion(id_Placa,id_EstadoPrediccion) VALUES (@id_Placa,@id_EstadoPrediccion)
	print 'La placa se ha agregado correctamente'
END
ELSE
BEGIN
	PRINT 'La placa ya existe'
END
'''