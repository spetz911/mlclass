# машинное обучение

## обновления:

+ 22.02 выложил слайды на тему первой лекции в lecture1_slides


## полезные материалы:

#### http://wiki.scipy.org/NumPy_for_Matlab_Users

описано как пользоваться numpy

большинство функций Matlab соответствуют Python, но находятся в модуле numpy,
т.е. вместо mean(X) пишем np.mean(X)

#### http://scikit-learn.org/stable/install.html

установка нужных либ

#### http://www.youtube.com/playlist?list=PLJOzdkh8T5kp99tGTEFjH_b9zqEQiiBtC

курс лекций ШАД Яндекса, очень крутой.

К первой лабе подходят лекции:
+ 006. Методы восстановления регрессии
+ 001. Вводная лекция

PS рекомендуется смотреть на скорости 1.25


#### https://www.coursera.org/course/ml

Курс очень поверхностный, из него взяты лабораторные работы.

Вместо Octave используется scipy, так что все *.m файлы превращаются в один *.py


## как запускать тесты:

python -m unittest -v test_sanity

места, которые надо сделать помечены 'YOUR CODE HERE'

## часть 1

пропускаем 1.1 Submitting Solutions

пропускаем 2.4 Visualizing J(θ)

зато делаем Extra Credit Exercises (optional)

под пунктом 3.3 Normal Equations
имеется ввиду использование псевдообратной матрицы вместо градиентного спуска


## команды numpy

np.ravel приводит матрицу или вектор к одномерному горизонтальному массиву

X[:, i] берет столбец с индексом i

