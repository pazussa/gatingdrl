# Medición de Métricas de Complejidad Computacional

## Valores que se Miden

### 1. **I_avg** - Número promedio de iteraciones de resolución de conflictos
- **Descripción**: Promedio de iteraciones necesarias para resolver conflictos temporales
- **Valor teórico LaTeX**: ~2 iteraciones
- **Dónde se captura**: En el bucle iterativo de `environment_impl.py:step()`
- **Significado**: Indica la dificultad promedio para encontrar una solución sin conflictos

### 2. **H_max/T_min** - Relación hiperperíodo máximo / período mínimo
- **Descripción**: Factor de amplificación debido a diferencias entre períodos de flujos
- **Valor teórico LaTeX**: ~10
- **Dónde se captura**: En `check_operation_isolation()` mediante `math.lcm(period1, period2)`
- **Significado**: Cuanto mayor sea esta relación, más iteraciones necesarias en verificación de conflictos

### 3. **F** - Número de flujos procesados
- **Descripción**: Cantidad total de flujos programados exitosamente
- **Valor teórico LaTeX**: ≤200
- **Dónde se captura**: Al completar cada flujo en `environment_impl.py`

### 4. **H** - Número promedio de saltos por flujo
- **Descripción**: Longitud promedio de las rutas de los flujos
- **Valor teórico LaTeX**: ≤5
- **Dónde se captura**: Al registrar cada flujo completado

### 5. **Complejidad teórica estimada**
- **Fórmula**: O((F·H)² × I_avg × H_max/T_min)
- **Valor teórico LaTeX**: ~2 × 10⁷ operaciones elementales para configuración típica

## Cómo Usar las Herramientas de Medición

### Opción 1: Script Rápido de Medición

```bash
# Medición rápida con parámetros por defecto
python measure_complexity.py

# Medición personalizada
python measure_complexity.py --time_steps 20000 --stream_count 50 --topo UNIDIR
```

### Opción 2: Entrenamiento Normal con Métricas

```bash
# Las métricas se capturan automáticamente durante cualquier entrenamiento
python ui/train.py --time_steps 50000 --stream_count 100 --show-log
```

### Opción 3: Análisis de Métricas Existentes

Si ya tienes un archivo `out/complexity_metrics.txt`, puedes analizarlo directamente:

```bash
cat out/complexity_metrics.txt
```

## Interpretación de Resultados

### Valores Esperados vs Observados

| Métrica | Valor LaTeX | Rango Esperado | Interpretación |
|---------|-------------|----------------|----------------|
| I_avg | ~2.0 | 1.0-4.0 | >4.0 indica problemas de convergencia |
| H_max/T_min | ~10.0 | 5.0-20.0 | >20.0 indica períodos muy heterogéneos |
| F | ≤200 | 10-500 | Depende de la configuración |
| H | ≤5.0 | 2.0-8.0 | Depende de la topología |

### Alertas de Rendimiento

- **I_avg > 5.0**: El algoritmo tiene dificultades para resolver conflictos
- **H_max/T_min > 50**: Los períodos son demasiado heterogéneos
- **Tiempo promedio > 10ms por operación**: Posible cuello de botella computacional

## Archivos Generados

1. **`out/complexity_metrics.txt`**: Reporte completo de métricas
2. **`out/train.log`**: Log detallado con métricas resumidas
3. **`out/performance_score.png`**: Gráfica de convergencia del entrenamiento

## Validación de la Fórmula de Complejidad

El sistema calcula automáticamente:

```
Complejidad = (F × H)² × I_avg × (H_max / T_min)
```

Y compara con los valores teóricos del LaTeX para validar que la implementación coincide con la descripción teórica.

## Casos de Uso

### Configuración de Desarrollo
```bash
python measure_complexity.py --time_steps 5000 --stream_count 10
```

### Configuración de Validación
```bash
python measure_complexity.py --time_steps 20000 --stream_count 50 --topo UNIDIR
```

### Configuración de Producción
```bash
python ui/train.py --time_steps 100000 --stream_count 200 --topo SIMPLE --show-log
```

## Comparación con Estado del Arte

### Referencias Clave Identificadas

#### 1. **TSNsched Framework (Santos et al., 2025)**
- **Problema**: TSN scheduling es NP-complete
- **Enfoque**: SMT solvers para reducir complejidad
- **Nuestra ventaja**: Análisis más granular con DRL

#### 2. **No-wait Packet Scheduling (Craciunas et al., 2016)**
- **Problema**: NW-PSP para modelar TSN scheduling
- **Enfoque**: Mapeo a teorías lógicas
- **Nuestra ventaja**: Solución adaptativa con aprendizaje

#### 3. **DRL Complexity Analysis Papers**
- **Complejidad típica**: O(N²) polynomial time
- **Nuestro enfoque**: O((F·H)² × I_avg × H_max/T_min)
- **Ventaja**: Parámetros específicos de TSN más detallados

### Metodología de Comparación

```bash
# Benchmarking experimental sistemático
python test_variability.py --compare-with-baseline
```

### Métricas de Evaluación Estándar
- **Complejidad temporal**: Tiempo de ejecución vs tamaño del problema
- **Complejidad espacial**: Uso de memoria durante entrenamiento
- **Escalabilidad**: Comportamiento con diferentes topologías
- **Calidad de solución**: Ratio de flujos programados exitosamente

## Troubleshooting

### Error: "Métricas no disponibles"
- Verificar que `tools/complexity_metrics.py` existe
- Verificar que no hay errores de importación

### Valores anómalos
- I_avg = 0: No se detectaron conflictos (escenario muy simple)
- H_max/T_min = 0: Todos los flujos tienen el mismo período
- F muy bajo: Posible problema de convergencia del algoritmo

### Rendimiento lento
- Reducir `--stream_count` para configuraciones iniciales
- Usar topologías más simples (SIMPLE vs UNIDIR)
- Verificar que no hay memory leaks en métricas
