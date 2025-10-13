## Математические определения

### Строгая математическая свёртка:
$$
(I * K)_{i, j} = \sum_{m} \sum_{n} I_{i - m, j - n} \cdot K_{m, n}
$$

### Кросс-корреляция:
$$
(I \star K)_{i, j} = \sum_{m} \sum_{n} I_{i + m, j + n} \cdot K_{m, n}
$$

## Операция перехода

### Определяем перевёрнутое ядро:
$$
K'_{m, n} = K_{-m, -n}
$$

### Показываем эквивалентность:
$$
\begin{aligned}
(I * K')_{i, j} &= \sum_{m} \sum_{n} I{i - m, j - n} \cdot K'_{m, n} \\
&= \sum_{m} \sum_{n} I_{i - m, j - n} \cdot K_{-m, -n}
\end{aligned}
$$

### Замена переменных:
$$
\text{Пусть } p = -m,\ q = -n \Rightarrow m = -p,\ n = -q
$$

$$
\begin{aligned}
(I * K')_{i, j} &= \sum_{p} \sum_{q} I_{i + p, j + q} \cdot K_{p, q} \\
&= (I \star K)_{i, j}
\end{aligned}
$$

## Визуализация для 1D случая

### Математическая свёртка:
$$
(I * K)_{i} = \sum_{m} I_{i - m} \cdot K_{m}
$$

### Кросс-корреляция:
$$
(I \star K)_{i} = \sum_{m} I_{i + m} \cdot K_{m}
$$

### Пример вычисления для сигнала `[a,b,c,d]` и ядра `[x,y,z]`:

**Математическая свёртка:**
$$
\begin{aligned}
&I * K = [a,b,c,d] * [x,y,z] \\
&[a\cdot z + b\cdot y + c\cdot x,\ b\cdot z + c\cdot y + d\cdot x]
\end{aligned}
$$

**Кросс-корреляция:**
$$
\begin{aligned}
&I \star K = [a,b,c,d] \star [x,y,z] \\
&[a\cdot x + b\cdot y + c\cdot z,\ b\cdot x + c\cdot y + d\cdot z]
\end{aligned}
$$

## Ключевой вывод

$$
\text{Если } K'_{m,n} = K_{-m,-n},\text{ то } (I * K')_{i,j} = (I \star K)_{i,j}
$$

Это показывает, что **кросс-корреляция эквивалентна свёртке с перевёрнутым ядром**, что объясняет, почему в CNN мы можем использовать кросс-корреляцию, но называть её "свёрткой" - обучаемые фильтры автоматически адаптируются к этой разнице.
