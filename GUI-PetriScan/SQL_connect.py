from PyQt5.QtSql import QSqlDatabase, QSqlQueryModel, QSqlQuery
from PyQt5.QtWidgets import QTableView, QApplication
import sys


SERVER_NAME = 'SRVBDI'
DATABASE_NAME = 'PLACASPETRI'
USERNAME = 'userplacaspetri'
PASSWORD = 'userplacaspetri2020'


def createConnection():
    connString = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER_NAME};DATABASE={DATABASE_NAME};UID={USERNAME}; PWD={PASSWORD}'

    global db
    db = QSqlDatabase.addDatabase('QODBC')
    # db.setHostName("SRVBDI")
    db.setDatabaseName(connString)
    # db.setUserName("userplacaspetri")
    # db.setPassword("userplacaspetri2020")
    # # db.setDatabaseName(connString)

    if db.open():
        print('conectarse a SQL Server correctamente')
        return True
    else:
        print('conexión fallida')
        return False


def displayData(sqlStatement):
    print('procesando consulta ...')
    qry = QSqlQuery(db)
    qry.prepare(sqlStatement)
    qry.exec()

    model = QSqlQueryModel()
    model.setQuery(qry)

    view = QTableView()
    view.setModel(model)
    return view


if __name__ == '__main__':
    app = QApplication(sys.argv)

    if createConnection():
        SQL_STATEMENT = 'select *from dbo.Tabla_Placa'
        dataView = displayData(SQL_STATEMENT)
        dataView.show()

    app.exit()
    sys.exit(app.exec_())
