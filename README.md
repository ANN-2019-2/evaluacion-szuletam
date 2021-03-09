# Laboratorios de programación

Este repo contiene las tareas asignadas para el curso Redes Neuronales Artificiales y Algoritmos Bio-inspirados, ofertado en la Facultad de Minas, Universidad Nacional de Colombia, Sede Medellín.

A continuación se describe el procedimiento para realizar los laboratorios de programación.

## Creación del repo individual para cada estudiante y clonación al computador personal


* Recibirá un vínculo al repositorio de Google Classroom para la evaluación del curso


* Ubique su nombre en la lista del curso y haga click para identificarse en GitHub Classroom. Como resultado se creará un repo único para usted. Almacene esta dirección.


* Clone su repositorio en su computador usando GitHub Desktop (Windows y Mac OS). Si usa Linux use los comandos equivalentes de la línea de comandos


## Ejecución del evaluador automático

* Para ejecutar el evaluador sobre todas las tareas digite `python3 grade`.  Para los puntos con una solución errónea, el sistema le reportará el resultado esperado y el resultado computado por su solución. Para los puntos solucionados correctamente, el sistema solo indicará que ejecuto el punto. Finalmente, el sistema le entregará por pantalla un informe detallado de cada punto y la correspondiente nota por punto, laboratorio y para el curso.

* Si esta realizando la solución de un punto particular, resulta más apropiado que entre a la carpeta de dicho punto y ejecute el comando `python3 grader`.  Este evaluará únicamente el punto actual.

## Uso de una máquina virtual de docker con Linux (por completar)

La carpeta ya contiene el archivo VAGRANTFILE preconfigurado para instalar y ejecutar una máquina virtual con Ubuntu 18.04

* Inicie Vagrant de la forma usual (las instrucciones están en https://github.com/jdvelasq/vagrant4docker). 


* Entre a la máquina virtual con `vagrant ssh`.


* Vaya a la carpeta compartida con el disco duro (`cd /vagrant`).


* Entre al directorio raíz de los laboratorios de programación.





