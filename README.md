# Tracking-Urban-Crime-Trajectories-Through-Dynamic-Neighborhood-Embeddings
Tracking Urban Crime Trajectories Through Dynamic Neighborhood Embeddings
1. Descripción del proyecto
Se propone una investigación orientada al aprendizaje de representaciones dinámicas de zonas urbanas a partir del conjunto público de carpetas de investigación de la Fiscalía General de Justicia de la Ciudad de México. El proyecto parte del supuesto de que el crimen reportado no solo presenta heterogeneidad espacial, sino también una evolución temporal que puede transformar gradualmente el perfil funcional de distintos territorios urbanos.

La propuesta no se centra en la predicción directa de delitos, sino en una pregunta metodológicamente robusta y científicamente publicable: cómo representar la evolución temporal de los perfiles delictivos de los barrios o zonas urbanas, y cómo identificar trayectorias de estabilidad, transición o cambio abrupto en dichas zonas. En lugar de tratar cada zona como una entidad estática, el proyecto la modela como una secuencia temporal de estados representados en un espacio latente.

El interés científico radica en que muchas zonas de la ciudad pueden conservar una identidad funcional relativamente estable a lo largo del tiempo, mientras que otras pueden experimentar transformaciones graduales o cambios más abruptos en su patrón de crimen reportado (antes tenían mucha criminalidad y hoy, tal vez no, o viceversa). Aprender estas trayectorias permitiría avanzar hacia una comprensión más fina de la dinámica urbana, aportando evidencia sobre persistencia territorial, reconfiguración funcional y heterogeneidad temporal del dato administrativo.

En este contexto, la propuesta integra tres componentes principales: armonización semántica de categorías delictivas ruidosas, construcción de firmas espacio-temporales por zona y periodo, y aprendizaje de embeddings dinámicos para modelar trayectorias urbanas. Esta formulación permite producir conocimiento útil para análisis urbano y representación temporal del territorio, sin afirmar causalmente por qué una zona cambia, sino identificando patrones consistentes de transformación observada.

2. Objetivo general
Construir representaciones urbanas robustas a registros policiales ruidosos y utilizarlas para dos fines complementarios:

O1: aprender embeddings temporales que permitan representar cada zona urbana como una trayectoria en el tiempo.
O2: identificar y caracterizar zonas con patrones de persistencia, transición gradual o cambio abrupto en su perfil delictivo reportado.
3. Justificación e impacto
Aunque los datasets policiales abiertos de la FGJ de CDMX han sido utilizados en diversos estudios, la mayoría de trabajos se concentran en análisis descriptivos, detección de hotspots o predicción del crimen, tratando con frecuencia las zonas urbanas como unidades estáticas. Sin embargo, la ciudad es un sistema dinámico, y los patrones delictivos reportados pueden cambiar en intensidad, composición y temporalidad, generando trayectorias urbanas diferenciadas que no son capturadas adecuadamente por enfoques estáticos. Es importante destacar que la palabra trayectoria no se refiere a camino o ruta, sino a la evolución criminológica de zonas urbanas en el tiempo.

Esta propuesta busca situarse en la intersección entre tres líneas relevantes: label noise en datos administrativos, urban region representation learning y análisis temporal de dinámicas territoriales. En lugar de preguntar únicamente qué zonas son similares, el proyecto busca responder cómo evolucionan esas similitudes a lo largo del tiempo y qué tipos de transformación urbana pueden detectarse en el espacio latente aprendido.

Desde una perspectiva aplicada, el proyecto aportaría en:

construir representaciones temporales más robustas del espacio urbano;
identificar zonas persistentes (el crimen no ha cambiado en el tiempo) y zonas en transformación (el crimen aumentó o disminuyó sistemáticamente en el tiempo);
detectar momentos de cambio en el perfil reportado de distintos territorios;
generar tipologías temporales de evolución barrial;
y aportar una metodología replicable para estudiar trayectorias urbanas con datos administrativos ruidosos.
El impacto práctico radica en que los resultados podrían ser útiles para investigadores urbanos, analistas de seguridad y entidades públicas interesadas en comprender no solo cómo se distribuye el crimen reportado, sino también cómo cambian los barrios en el tiempo. Al mismo tiempo, el enfoque sería transferible a otras ciudades latinoamericanas con datos administrativos de estructura similar.

4. Pregunta de investigación
La pregunta central de la propuesta sería la siguiente: ¿Es posible aprender representaciones dinámicas de zonas urbanas robustas a categorías delictivas ruidosas y utilizarlas para modelar trayectorias temporales que permitan identificar patrones de persistencia, transición y cambio abrupto en el crimen reportado?

De manera complementaria, el proyecto podría explorar dos subpreguntas:

¿Qué tipos de trayectorias temporales presentan las distintas zonas urbanas de la ciudad?
¿Qué zonas muestran cambios de perfil más intensos, más graduales o más persistentes a lo largo del tiempo?
