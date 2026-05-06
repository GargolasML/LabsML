import pandas as pd
import glob


def check_requirements():

    #Verificar archivos csv
    csv_files = glob.glob('*.csv')

    if not csv_files:
        raise FileNotFoundError('No hay archivos CSV en la carpeta')
    
    print('Archivos encontrados:', csv_files)

    # Cargamos el archivo
    try:
        df_predictions = pd.read_csv("predictions.csv")
    except FileNotFoundError:
        raise FileNotFoundError('No se ha encontrado el archivo predictions.csv, revise si el nombre es correcto o si el archivo se exportó correctamente')
    except Exception as e:
        raise ImportError(f"Error al cargar el archivo {e}")
    
    print('Archivo cargado correctamente')

    # Verificar que no este vacío
    if df_predictions.empty:
        raise ValueError('El archivo predictions.csv está vacío')
    print('El archivo predictions.csv no está vacío')

    # Verificar número de columnas
    if df_predictions.shape[1] < 2:
        raise ValueError(f'El archivo predictions.csv tiene no tiene las columnas necesarias')
    
    # Verificar columnas requeridas
    required_columns = ['prediction']
    missing_columns = [column for column in required_columns if column not in df_predictions.columns]

    if missing_columns:
        raise ValueError(f'El archivo predictions.csv no tiene las siguientes columnas: {missing_columns}')
    print("Todas las columnas requeridas están presentes, con los nombres adecuados.")

    # Verificar cantidad de predicciones (excluyendo NaN)
    predictions_count = df_predictions['prediction'].count() # Ignora NaN por defecto
    expected_predictions = 2676

    if predictions_count != expected_predictions:
        raise ValueError(f'El archivo predictions.csv tiene {predictions_count} predicciones válidas, se esperaban {expected_predictions}')

    print("El archivo contiene las 2676 filas requeridas.") 
    print('Requisitos completados')
    
