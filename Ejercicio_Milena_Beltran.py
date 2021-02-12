#!/usr/bin/env python
# coding: utf-8

# # Python para análisis de datos
# 
# 
# Para iniciar con python se recomienda instalar [ANACONDA](https://www.continuum.io/downloads) , en la version  3.x
# 
# ---
# 
# 
# **Nota** Despues de instalado se puede cambiar de version siguiendo los  [pasos](http://conda.pydata.org/docs/py2or3.html) , [ayuda](http://stackoverflow.com/questions/30492623/using-both-python-2-x-and-python-3-x-in-ipython-notebook)
# ---
# 

# ## Verifiquemos que tenemos lo que necesitamos

# In[68]:


#@title
# Python version
import sys
print('Python: {}'.format(sys.version))

import sklearn
print('sklearn: {}'.format(sklearn.__version__))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))


# 
# ## Buscar ayuda
# Tenemos la opción de usar ? para ver la ayuda de una función, o usar tab para ver las funciones disponibles de un objeto
# 
# Ejercicio: importar la biblioteca random y buscar ayuda sobre la función `random`
# 
# 

# In[69]:


import random 




# # Python
# Python es un lenguaje de programación de alto nivel, interpretado, interactivo, orientado a objetos y de propósito general. Fue creado por Guido van Rossum durante 1985-1990. Al igual que Perl, el código fuente de Python también está disponible bajo la Licencia Pública General de GNU (GPL).
# 
# basado en [Complete Python Bootcamp: Go from zero to hero in Python 3](https://www.udemy.com/complete-python-bootcamp/) y [Python - Tutorial](https://www.tutorialspoint.com/python/index.htm) (*warning: is python2*)
# 
# - se considera la mejor práctica ([PEP8](https://www.python.org/dev/peps/pep-0008/#function-and-variable-names)) que los nombres de las variables están en minúsculas con guiones bajos
# - Python utiliza *escritura dinámica*, lo que significa que puede reasignar variables a diferentes tipos de datos.
# - Python soporta cuatro tipos numéricos diferentes
# 
#   - int (enteros con signo) `5`
#   - long (enteros largos, también se pueden representar en octal y hexadecimal) `5L`,` 0x80`
#   - float (valores reales de punto flotante) `0.0`,` 0.5e52`
#   - complejo (números complejos) `14.5j`
# - Se puede pensar en las listas de la versión más general de una secuencia en Python. A diferencia de las cadenas, son mutables, lo que significa que los elementos dentro de una lista se pueden cambiar.
# - Lo importante de una lista es que los elementos de una lista no tienen que ser del mismo tipo.
# - Una gran característica de las estructuras de datos de Python es que admiten el anidamiento. Esto significa que podemos tener estructuras de datos dentro de estructuras de datos
# -En las estructuras de datos Diccionario([dict](https://docs.python.org/3/tutorial/datastructures.html)) Las claves son únicas dentro de un diccionario, mientras que los valores pueden no serlo.
# - Los valores de un diccionario pueden ser de cualquier tipo, pero las claves deben ser de un tipo de datos inmutables como cadenas, números o tuplas.
# - En Python, la TUPLA es otro tipo de arreglo que dispone el Python. Esta variable está más cerca al concepto matemático de vector, puesto que en Python la tupla se maneja como una lista más cerrada porque no puede cambiar elementos individuales, en otras palabras, es inmutable según lo definen [tuples](https://docs.python.org/3/tutorial/datastructures.html#tuples-and-sequences)
# 
# 
# Un buen punto de partida es descargar un [datasheet] del tema de busqueda esta pagina presenta excelentes resumenes de python [Python Crash Course, Second Edition](https://ehmatthes.github.io/pcc_2e/cheat_sheets/cheat_sheets/), el cual es resumen de un muy buen [libro](https://nostarch.com/pythoncrashcourse2e)

# 
# ## Python basico
# 
# 
# Sin ejecutar el codigo, que crees que retorna cada item? pruebalo
# 
# 1. **numbers** 
# 
# ```
# 3/2
# 3/float(2)
# 3%2
# 3.00//2.00
# 2**3
# 4**0.5
# import math
# math.sqrt(4)
# 2 + 10 * 10 + 3
# (2+10) * (10+3)
# ```
# 2. **Variables**
# 
# ```
# a = 10
# a =a + 10
# a += 10
# b = c = a
# a,b,c = 1,2,"john"
# my_dogs = 2
# my_dogs = ['Sammy', 'Frankie']
# del my_dogs
# 
# ```
# 3. **string**
# 
# ```
# str = 'Hello World!'
# 
# print (str)          
# print (str[0])       
# print (str[2:5])     
# print (str[2:])      
# print (str * 2)      
# print (str + "TEST")
# str.upper()
# str.split()
# 'Carlos: {}'.format(str)
# print(" %s ,  %s" %(str,"I'm Carlos"))
# 
# print('I wrote %s programs today.' %3.75)
# print('I wrote %d programs today.' %3.75)
# print('I wrote %5.2f programs today.' %3.75)
# print('First: %s, Second: %5.2f, Third: %r' %('hi!',3.1415,'bye!'))
# print('The {2} {1} {0}'.format('fox','brown','quick'))
# print('First Object: {a}, Second Object: {b}, Third Object: {c}'.format(a=1,b='Two',c=12.3))
# 
# print('{0:8} | {1:9}'.format('Fruit', 'Quantity'))
# print('{0:8} | {1:9}'.format('Apples', 3.))
# print('{0:8} | {1:9}'.format('Oranges', 10))
# 
# ```
# 
# 4. **Array list**
# 
# ```
# my_list = [1,2,3]
# my_list = ['A string',23,100.232,0]
# my_list + ['new item']
# my_list = my_list +['new item']
# my_list.pop(0)
# my_list.pop(-1)
# my_list.reverse()
# my_list.sort()
# my_list * 2
# my_list=[]
# my_list=array()
# 
# num_list = range(1, 101)
# print(num_list[49])
# type(num_list)
# len(num_list)
# 
# new_list = [0, 1, 2, 3, 4, 5, 6]
# new_list[2] = 2001;
# new_list[2:5]
# new_list[:3]
# new_list[-2:]
# 
# list_1 = [1,2]
# list_2 = [3, 4]
# list_1=list_1+list_2
# list_1.append(list_2)
# [num*2 for num in list_1]
# 
# lst_1=[1,2,3]
# lst_2=[4,5,6]
# lst_3=[7,8,9]
# matrix = [lst_1,lst_2,lst_3]
# matrix[0][0]=8
# print(matrix)
# 
# ```
# 4.  **Dictionary**
# 
# ```
# d = {'a':10, 'b':20, 'c':30}
# d['a']
# d['b']=40
# d.clear()
# 
# print(d)
# 
# dict = {'Name': 'Zara', 'Age': 7, 'Name': 'Manni'}
# dict.values()
# dict.keys()
# dict.items()
# print(dict['Name'])
# dict2['Name']='Carlos'
# dict.update(dict2)
# 
# 
# locations = {'Sinan': ['Baltimore', 'MD'], 'Brandon': ['Arlington', 'VA']}
# print(locations['Brandon'][1])
# 
# from numpy import random
# data = {'a'+str(i) : random.randn() for i in range(7)}
# print(data['a1'])
# data
# ```
# 
# 5. ** functions**
# 
# ```
# def calc(a, b):
#     return a + b
# print(calc(2,3))
# ```
# 
# Para probar cualquiera de los códigos anteriores, simplemente copie el bloque de código en la celda de python a continuación.

# In[70]:


3/2
3/float(2)
3%2
3.00//2.00
2**3
4**0.5
import math
math.sqrt(4)
2 + 10 * 10 + 3
(2+10) * (10+3)

#Se realizan los calculos comococcllomo


# ![numpy](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/NumPy_logo.svg/640px-NumPy_logo.svg.png)
# 
# 
# ## Numpy
# 
# 
# NumPy es el paquete fundamental para la computación científica con Python. Contiene entre otras cosas:
# 
# - un poderoso objeto de matriz N-dimensional
# 
# -funciones sofisticadas (de transmisió
# - Álgebra lineal útil, transformada de Fourier y capacidades de números aleatorios
# 
# 
# Todo Arreglo/array en Numpy es tratado como un objeto que entre sus propiedades tiene:
# 
# - ndarray.ndim –> Proporciona el número de dimensiones de nuestro array. El array identidad es un array cuadrado con una diagonal principal unitaria.
# - ndarray.shape –> Devuelve la dimensión del array, es decir, una tupla de enteros indicando el tamaño del array en cada dimensión. Para una matriz de n filas y m columnas obtendremos (n,m).
# - ndarray.size –> Es el número total de elementos del array.
# - ndarray.dtype –> Es un objeto que describe el tipo de elementos del array.
# - ndarray.itemsize –> devuelve el tamaño del array en bytes.
# - ndarray.data –> El buffer contiene los elementos actuales del array.
# 
# ## Creación de Arrays
# 
# 
# - identity(n,dtype). Devuelve la matriz identidad, es decir, uma matriz cuadrada nula excepto en su diagonal principal que es unitaria. n es el número de filas (y columnas) que tendrá la matriz y dtype es el tipo de dato. Este argumento es opcional. Si no se establece, se toma por defecto como flotante.
# - ones(shape,dtype). Crea un array de unos compuesto de shape elementos.
# - zeros(shape, dtype). Crea un array de ceros compuesto de “shape” elementos”.
# - empty(shape, dtype). Crea un array de ceros compuesto de “shape” elementos” sin entradas.
# - eye(N, M, k, dtype). Crea un array bidimensional con unos en la diagonal k y ceros en el resto. Es similar a identity. Todos los argumentos son opcionales. N es el número de filas, M el de columnas y k es el índice de la diagonal. Cuando k=0 nos referimos a la diagonal principal y por tanto eye es similar a identity.
# - arange([start,]stop[,step,],dtype=None). Crea un array con valores distanciados step entre el valor inicial star y el valor final stop. Si no se establece step python establecerá uno por defecto.
# - linspace(start,stop,num,endpoint=True,retstep=False). Crea un array con valor inicial start, valor final stop y num elementos.
# - meshgrid(x,y). Genera una matriz de coordenadas a partir de dos los arrays x, y.
# 
# ## Funciones
# 
# El módulo numpy contiene todas las operaciones usuales entre arrays como las matemáticas (suma, resta, multiplicación, etc), las lógicas (and, or, xor, etc), binarias, ... 
# [Operaciones Numpy](https://drive.google.com/file/d/17wZqpCvckCb8Ous3HAaRqPu33rJKtTeB/view?usp=sharing)
# 
# 
# [buen tutorial](http://pendientedemigracion.ucm.es/info/aocg/python/modulos_cientificos/numpy/index.html)
# 
# Lo más fácil y sencillo es en caso de duda consultar la documentación oficial. [documentacion](https://docs.scipy.org/doc/numpy/reference/index.html#reference)
# 
# 
# 

# In[36]:


import numpy as np # Importamos numpy como el alias np


# In[71]:



# tipo de datos
arr1 = np.array([1, 2, 3], dtype=np.float64)
arr2 = np.array([1, 2, 3], dtype=np.int32)
#funciones basicas
miArray = np.arange(10) # Creamos un array de 0 a 9 separados de uno en uno
print (miArray) # Presentamos en pantalla el array creado
print (type(miArray))   #Comprobamos que es un ndarray
print (miArray.ndim )   # Consultamos el número de dimensiones
print (miArray.shape )  # Consultamos la dimensión
print (miArray.size )  # Consultamos la dimensión
print (miArray.dtype )  # Consultamos el tipo de elementos del array
print (miArray.itemsize )  # tamaño en bytes
print (miArray.data )  # Consultamos el buffer de memoria.


# In[13]:


# array bidimensionales
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(arr2d[2])

print(arr2d[0][2])
#arr2d[0,2]

#grupo de elementos
print((arr2d[:,1]))

# acesso con operaciones
print(arr2d[arr2d>4])

#fancy indexing (una matrix es en verdad un arreglo)
arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i
print(arr)

#seleccionar columnas en desorden

print (arr[[4, 3, 0, 6]])
#Using negative indices select rows from the end
print(arr[[-3, -5, -7]])

#reorganizar vectores
arr = np.arange(32).reshape((8, 4))
print(arr)


# 
# 
# ### **Ejercicio**
# 
# Crear dos vectores de diez numeros aleatorios enteros [np.random.randint](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.randint.html) entre 1 y 10 y realizar las siguientes operaciones:
# - sumarlos
# - concatenar los dos arreglos
# - encontrar la sumatoria de los valores
# - encontrar la media de los valores
# - imprimir los numeros mayores que 5
# - determinar las posiciones de los numeros que son iguales en los arreglos
# 
# 

# In[65]:


import itertools
print('INICIO_EJERCICIO')
print('Creacion Dos Vectores:')
Matriz_1 = (np.random.randint(1, 10, (1, 10)))
print('Matriz_1 =', Matriz_1)
Matriz_2 = np.random.randint(1, 10, (1, 10))
print('Matriz_2 =', Matriz_2)
print('--------------------------------')
print('Sumarlos:')
SumarArray=Matriz_1+Matriz_2
print('Suma:', SumarArray)
print('--------------------------------')
print('Concatenar los dos arreglos:')
Concatenacion= np.append(Matriz_1, Matriz_2)
print('Concatenacion =' , Concatenacion)
print('--------------------------------')
print('Encontrar la sumatoria de los valores')
Sum_Matriz_1 = sum(itertools.chain.from_iterable(Matriz_1))
print('Sum_Matriz_1 =',Sum_Matriz_1)
Sum_Matriz_2 = sum(itertools.chain.from_iterable(Matriz_2))
print('Sum_Matriz_2 =',Sum_Matriz_2)
print('--------------------------------')
print('Encontrar la media de los valores')
print('Media Matriz_1 =',np.mean(Matriz_1))
print('Media Matriz_2 =',np.mean(Matriz_2))
print('--------------------------------')
print('Imprimir los numeros mayores que 5:')
print('Mayores' + str(Matriz_1[Matriz_1>5]))
print('--------------------------------')
print('Determinar las posiciones de los numeros que son iguales en los arreglos')
print(array1[array1==array2])
print('--------------------------------')
print('FIN_EJERCICIO')


# ## SQLite
# 
# Cuando abre un excel o CSV en Python y lo asigna a un nombre de variable, está usando la memoria de su computadora para guardar esa variable. Acceder a los datos desde una base de datos como SQL no solo es más eficiente, sino que también le permite importar solo las partes de los datos que necesita.
# 
# 
# Cada una de las bases de datos tienen sus librerias para python, a manera de ejemplo veamos SQLite.
# 
# - Revisar la base de datos en linea [sqliteonline](https://sqliteonline.com/)
# 
# 
# - **Ejercicio** escribamos tres query (enunciado en [Markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet) y luego el codigo sql respectivo.
# 
# los datos son   la Guía de valor para vehículos en Colombia suministrada por fasecolda, que se puede descargar de [Kaggle](https://www.kaggle.com/panchicore/vehicles-colombia-fasecolda/version/1)
# 
# 
# [Link de descarga](https://drive.google.com/file/d/1wVSVbq1cJ68dU-0D8c4tPh83Hjy8azPM/view?usp=sharing)
# 
# 
# tutorial [sqllite](http://www.datacarpentry.org/python-ecology-lesson/08-working-with-sql) 
# 

# In[79]:


# Cargamos el archivo (solo en collaborative) en caso contrario colocar el archivo en la raiz
#from google.colab import files
#files.upload()


# In[138]:


import sqlite3

# Create a SQL connection to our SQLite database
con = sqlite3.connect("guia_fasecolda.sqlite")
cur = con.cursor()

# the result of a "cursor.execute" can be iterated over by row

#for row in cur.execute('SELECT Marca,Referencia1,Cilindraje,Peso,"2015","2016","2017" FROM carros where Cilindraje<1500 AND Cilindraje>1200 AND "2015">0;'):
#   print(row)

#for row in cur.execute('SELECT * FROM carros '):
#   print(row)
#AQUI Realize sus consultas sql de los datos
print('Validando campos de tabla carros')
for row in cur.execute('SELECT SQL FROM sqlite_master WHERE name = "carros"'):
   print(row)
print('--------------------------------')
print('Validando # 2 los campos de tabla carros')
for row in cur.execute('PRAGMA table_info(carros)'):
   print(row)
print('--------------------------------')
print('--------------------------------')
print('--------------------------------')
print('--------------------------------')
print('INICIO_EJERCICIO')
print('Ejercicio # 1 SQL-LITE')
for row in cur.execute('SELECT marca, 2018, Nacionalidad FROM carros WHERE 2018 <> 0  AND Nacionalidad = "COL" group by marca ;'):
    print(row)
print('--------------------------------')
print('Ejercicio # 2 SQL-LITE')
for row in cur.execute('SELECT Transmision,count(0) FROM carros GROUP BY Transmision ;'):
    print(row)
print('--------------------------------')
print('Ejercicio # 3 SQL-LITE')
for row in cur.execute('SELECT clase, Marca,Referencia1,count(0) FROM carros WHERE Clase = "MOTOCICLETA" group by Referencia1;'):
    print(row)
print('--------------------------------')
print('SQL-LITE')
for row in cur.execute('SELECT Clase,Servicio,Marca, count(0) FROM carros Where Servicio = "Publico" Group by Clase;'):
    print(row)
print('--------------------------------')
#Be sure to close the connection.
con.close()


# 
# ## Pandas
# 
# ![Panda Logo](https://pandas.pydata.org/_static/pandas_logo.png)
# 
# Documentación de [Pandas](http://pandas.pydata.org/pandas-docs/stable/10min.html)
# 
# Pandas es una librería para el análisis de datos que cuenta con las estructuras de datos que necesitamos para limpiar los datos en bruto y que sean aptos para el análisis (por ejemplo, tablas). Es importante señalar aquí que, dado que pandas lleva a cabo tareas importantes, como alinear datos para su comparación, fusionar conjuntos de datos, gestión de datos perdidos, etc., se ha convertido en una librería muy importante para procesar datos a alto nivel en Python (es decir, estadísticas ). Pandas fue diseñada originalmente para gestionar datos financieros, y como alternativa al uso de hojas de cálculo (es decir, Microsoft Excel). Esta libreria tambien es equivalente en gran parte a las estructuras de datos de R.
# 
# La estructura de datos básica de pandas son:
# 
# - **Series**: Son arrays unidimensionales con indexación (arrays con índice o etiquetados), similar a los diccionarios. Pueden generarse a partir de diccionarios o de listas.
# - **DataFrame**: Es una colección ordenada de columnas con nombres y tipos. Son estructuras de datos similares a las tablas de bases de datos relacionales como SQL, donde una sola fila representa un único caso (ejemplo) y las columnas representan atributos particulares. 
# - **Panel, Panel4D y PanelND**: Estas estructuras de datos permiten trabajar con más de dos dimensiones. 
# 
# Sobre este tipo de estructura de datos se aplican funciones, ejemplo:
# 

# In[175]:


import pandas as pd

dates = pd.date_range('20200101', periods=6)
print(dates)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
print (np.random.randn(6,4))


# In[128]:


#valores iniciales de la lista
df.head()


# In[129]:


#valores finales de la lista
df.tail(3)


# In[130]:


#aceder a los datos como un arreglo numpy
print(df.index)
print(df.columns)
df.values


# In[131]:



# acediendo ciertos datos
df['A']
df[1:3]
df[df.A > 0]

df.sort_values(by='B')


# In[132]:


# toda la estadistica descriptiva

df.describe()


# Existen herramientas que permiten generar una descripción grafica y numerica automaticamente tales como:
# - [sweetviz](https://github.com/fbdesignpro/sweetviz)
# - [pandas profiling](https://github.com/pandas-profiling/pandas-profiling)
# 
# Estas librerias no estan por defecto instaladas en google colab o en su ambiente de trabajo, por lo cual se pueden instalar con 
# 
#   *!pip install package*

# In[133]:


get_ipython().system('pip install sweetviz')


# In[134]:


import sweetviz as sv

my_report = sv.analyze(df)
my_report.show_notebook()


# In[135]:


get_ipython().system('pip install pandas-profiling==2.9.0')


# In[136]:


from pandas_profiling import ProfileReport
profile = ProfileReport(df)
profile.to_widgets()


# ## Comprender los datos con estadísticas descriptivas
# Una vez que haya cargado sus datos en Python, debe poder entenderlos. Cuanto mejor pueda comprender sus datos, mejores y más precisos serán los modelos que puede construir.
# El primer paso para comprender sus datos es utilizar estadísticas descriptivas. 
# 
# ### **Ejercicio**
# 
# - Realize un query sobre los datos que desea analizar con datos de precios de varios años
# - Revise la distribución de sus datos con la función [describe()](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html#pandas.DataFrame.describe).
# - Calcule la correlación por pares entre sus variables usando la función [corr ()](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.corr.html#pandas.DataFrame.corr) .
# 
# [documentación](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html)
# 
# 
# 
# 

# In[209]:


import pandas as pd
import sqlite3
from IPython.display import display, HTML



# Read sqlite query results into a pandas DataFrame
con = sqlite3.connect("guia_fasecolda.sqlite")

#ESCRIBE LA CONSULTA DE LO QUE DESEAS ANALIZAR
#df = pd.read_sql_query('SELECT Marca,Referencia1,Cilindraje,Peso,"2015","2016","2017" FROM carros where Cilindraje<1500 AND Cilindraje>1200 AND "2015">0;', con)
#df = pd.read_sql_query('SELECT * FROM carros', con)
print("Total de memoria ram requerida ",df.memory_usage().sum())

#estadistica descriptiva
display(HTML(df.describe().to_html()))
#correlación entre las columnas
display(HTML(df.corr().to_html()))
print('INICIO_EJERCICIO')
print('Estadisticas Descriptivas')
print("Realize un query sobre los datos que desea analizar con datos de precios de varios años")
print('SQL-LITE')
dataset = pd.read_sql('SELECT marca,clase,Servicio,Nacionalidad FROM carros where Nacionalidad <> ""', con)
print('--------------------------------')
print("Revise la distribución de sus datos con la función describe()")
print(dataset)
dataset.dtypes
#df2 = pd.DataFrame()
#df2.describe()

print("Calcule la correlación por pares entre sus variables usando la función corr ()")
con.close()


# Como notaste en el ejemplo anterior Pandas permite conectarse directamente a la base de datos, tambien permite leer datos en multiples formatos [documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html) , por ejemplo si yo quiero cargar los datos originales de FASECOLDA

# In[160]:


# Cargamos el archivo (solo en collaborative)
from google.colab import files
files.upload()


# In[ ]:



data = pd.read_csv('guia_fasecolda.csv')
print("Total de memoria ram requerida ",data.memory_usage().sum())


print(data.columns)


# 
# 
# ## Comprender los datos con visualización
# 
# Debes dedicar tiempo a comprender mejor tus datos. Una segunda forma de mejorar su comprensión de sus datos es mediante el uso de técnicas de visualización de datos . Usar la representación en Python para comprender los atributos por sí solos y sus interacciones. Ejemplos:
# - Use la función hist () para crear un histograma de cada atributo.
# - Use la función plot (kind = 'box') para crear diagramas de caja y cajas de cada atributo.
# - Utilice la función  pandas.scatter_matrix() para crear gráficos de dispersión por pares de todos los atributos.
# 
# Es importante agregar```% inline matplotlib```  en la primera línea de su código si desea que la imagen esté incrustada en el *notebook* de Júpiter.
# 
# 
# [pyplot](http://matplotlib.org/api/pyplot_api.html) es la mejor opción, claro que si trabajamos con Pandas, el tiene ya programada las graficas utlizando toda la información del dataset [ejemplo](https://pandas.pydata.org/pandas-docs/stable/10min.html#plotting)
# 
# 
# 
# 

# ### **Ejercicio** 
# 
# - Cargar el dataset de Fasecolda suministrado en csv
# 
# -  Comprenda sus datos usando la función `columns` para ver las filas
# -  Utilize diferentes criterios para filtrar los datos.
# -  utilize por lo menos tres funciones de pandas para describir los datos filtrados
# -  Guarde los datos filtrados
# -  Realize por lo menos tres graficas de los datos
# 
# [Link de descarga](https://drive.google.com/file/d/1mPnVLno8uLF1bD1EQbIRVUmxk1c1BVc4/view?usp=sharing)

# In[229]:


# Load CSV using Pandas from URL
import pandas as pd
from IPython.display import display, HTML

print('Cargar el dataset de Fasecolda suministrado en csv') 
dataset = pd.read_csv('guia_fasecolda.csv')

print('Comprenda sus datos usando la función columns para ver las filas') 
print(dataset.tail(50))
print(dataset.columns)
print(dataset.describe())
print(dataset.head())

print('Utilize diferentes criterios para filtrar los datos.') 
print(dataset[dataset.Clase!='MOTOCICLETA'])

# guarda los datos filtrados


display(HTML((data.head().to_html())))
#print(df[data['2015']>50000])


# In[224]:



get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt

plt.figure()
data[['Cilindraje','Peso']].plot(kind = 'box',ylim=(-1000,10000))
plt.title('mi primera grafica')
plt.show()


# ## Conclusiones
# 
# 
# **De acuerdo a este análisis previo de los datos sobre la base de datos de FASECOLDA, que preguntas harias de análisis de datos que puedan generar valor.**
# * ¿Que Clase de Vehiculo tiene mayor participacion en el area de seguros? 
# * ¿Segun la clase de vehiculo cual reporta mayor accidentalidad?          
# 
# **Con que otra base de datos trabajarias para obtener valor de estos datos.**
# 
# *   Costos de los Vehiculos y precio del seguro    
# 
# **Utilize por lo menos dos análisis del curso de estadistica y/o de visualización de datos para este problema.**
# 
# * Analisis Diagnostico:   
# * Analizar por clasificacion de vehiculo cual es el que genera mayores indices de accidentalidad y que costos genera para el sector asegurador 
# 
# * Analitica Predictiva:   
# * Basado en el analisis diagnostico realizar una evaluacion de costos y accidentalidad para realizar reajustes a los valores anuales de los seguros.
#  
# 
# **Plantea una pregunta aprendizaje supervisado y otra de supervisado que pensarias genere valor sobre este dataset**
# 
# * ¿De acuerdo a la clasificacion de los vehiculos puede estimarse el valor del seguro? 
# * ¿Se puede lograr con apoyo de redes sociales estimar la mayor accidentalidad por clase de vehiculo? 
# 

# In[ ]:




