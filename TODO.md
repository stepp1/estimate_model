# TODOs

## Tareas de baja dificultad
1. [Dif. baja] Utilizar top "X %" de píxeles más informativos de MNIST (con respecto a las marginales) para calcular su IM conjunta.

2. [Dif. baja] Ajustar nuevos datasets: MNIST perturbado (rotaciones o traslaciones) y CIFAR10; este último en tres etapas (1) en escala de grises, (2) a color considerando las estadísticas marginales de cada canal y (3) a color considerado la estadística conjunta de los tres canales que conforman el pixel.

3. [Dif. media] Manipular el ajuste del modelo a MNIST para que considere estadísticas no marginales. En particular, grupos de píxeles contiguos; se puede comenzar con algo sencillo (e.g., estadísticas de pares de píxeles contiguos).


## Tareas de media dificultad

Se requiere comprender el modelo generativo con respecto a su lógica formal y algorítmica, así como su implementación en código.

4. [Dif. media] Desarrollar la vía de calcular parámetros para generar un modelo correspondiente a una combinación convexa de parámetro "alpha" de un modelo ajustado (e.g., en MNIST) y un modelo de ruido completamente uniforme; de esta forma, "alpha" resulta en un parámetro de control continuo de la dificultad del problema. Se requiere (1) encontrar la vía de calcular las probabilidades correspondientes, (2) estudiar la expresión de IM analítica y entender cómo "alpha" la afecta, (3) implementar esta modificación en el código de la clase.
5. [Dif. media] Un problema de la implementación actual, es el alto uso de memoria en el cálculo de la IM. Actualmente todos los números que se operan están representados en 64 bits; sería útil utilizar representaciones de 32 bits, e incluso representaciones aún más comprimidas (esto es viable ya que no necesitamos una precisión tan elevada en la cota de cross-entropy).
6. [Dif. media-alta] Debido a la complejidad computacional del cálculo de IM, es deseable paralelizar el mismo. Los términos que conforman la MI se calculan recursivamente pixel a pixel (o grupo a grupo de coordenadas en el caso general); esta construcción sigue una operación que es asociativa, por lo cual se puede utilizar la estrategia de "divide and conquer" para calcular el resultado final. Una forma de resolver esto es establecer una cola (queue) de términos a procesar y operarlos de forma asíncrona, encolando los resultados, hasta que la cola tenga un solo elemento.
7. [Dif. alta] Actualmente diversos términos que componen la suma de IM suponen un aporte marginal en el cálculo de la misma. Esto es posible verificarlo a posteriori en los cálculos de IM; de aquí surge la necesidad de ir eliminando términos en la construcción recursiva de los mismos de forma inteligente. El primer desafío es cuantificar la merma de MI que supone la eliminación de un término durante la construcción recursiva en el resultado final y de esta forma proponer un criterio de eliminación que logre reducir la cantidad de términos y en último lugar reducir el crecimiento asintótico de los mismos.
8. [Dif. alta] Proponer, analizar, desarrollar e implementar cualquier otro método que permita reducir la complejidad algorítmica y de memoria del cálculo de la IM sin que suponga un detrimento al cálculo de la IM analítica. Alternativamente, es aceptable una merma, siempre que se conozca una cota o un intervalo de confianza asociado al resultado.