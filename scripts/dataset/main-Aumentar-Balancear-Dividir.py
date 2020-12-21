
import os


def menu():
    os.system('cls')  # NOTA para windows tienes que cambiar clear por cls
    print("\nMENÚ:")
    print("\t1 - Aumentar Dataset")
    print("\t2 - Balancear Dataset")
    print("\t3 - Dividir Dataset en 'Test' y 'Train'")
    print("\t4 - Ejecutar todas las opciones")
    print("\t5 - Salir")


while True:
    # Mostramos el menu
    menu()

    # solicitamos una opción al usuario
    opcionMenu = input("\nELIGE UNA OPCIÓN >> ")

    if opcionMenu == "1":
        print("")
        print("AUMENTANDO DATASET DE IMÁGENES...")
        os.system('python Aumentar_dataset_imagenes.py')
        print("FIN \n_______________________________________________________________")

    elif opcionMenu == "2":
        print("")
        print("BALANCEANDO DATASET DE IMÁGENES...")
        os.system('python Balancear_dataset_imagenes.py')
        print("FIN \n_______________________________________________________________")

    elif opcionMenu == "3":
        print("")
        print("DIVIDIENDO DATASET DE IMÁGENES...")
        os.system('python Dividir_Carpetas_train_test.py')
        print("FIN \n_______________________________________________________________")

    elif opcionMenu == "4":
        print("")
        print("AUMENTANDO DATASET DE IMÁGENES...")
        os.system('python Aumentar_dataset_imagenes.py')
        print("FIN \n_______________________________________________________________")
        print("")
        print("python BALANCEANDO DATASET DE IMÁGENES...")
        os.system('Balancear_dataset_imagenes.py')
        print("FIN \n_______________________________________________________________")
        print("DIVIDIENDO DATASET DE IMÁGENES...")
        os.system('python Dividir_Carpetas_train_test.py')
        print("FIN \n_______________________________________________________________")

    elif opcionMenu == "5":
        break

    else:
        print("")
        input("No has pulsado ninguna opción correcta...\npulsa intro para continuar")
