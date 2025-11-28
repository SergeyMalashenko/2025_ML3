import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from sklearn.linear_model import LinearRegression
plt.style.use('fivethirtyeight')


def fit_model(x_train, y_train):
    # Подбирает линейную регрессию, чтобы найти фактические значения b и w, которые минимизируют потери
    regression = LinearRegression()
    regression.fit(x_train, y_train)
    b_minimum, w_minimum = regression.intercept_[0], regression.coef_[0][0]
    return b_minimum, w_minimum


def find_index(b, w, bs, ws):
    # Ищет ближайшие индексы для обновлённых b
    # и w внутри их соответствующих диапазонов
    b_idx = np.argmin(np.abs(bs[0, :] - b))
    w_idx = np.argmin(np.abs(ws[:, 0] - w))

    # Ближайшие значения для b и w
    fixedb, fixedw = bs[0, b_idx], ws[w_idx, 0]
    
    return b_idx, w_idx, fixedb, fixedw


def figure1(x_train, y_train, x_val, y_val):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    ax[0].scatter(x_train, y_train)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_ylim([0, 3.1])
    ax[0].set_title('Генерируемые данные — Обучения')

    ax[1].scatter(x_val, y_val, c='r')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].set_ylim([0, 3.1])
    ax[1].set_title('Генерируемые данные - Валидации')
    fig.tight_layout()
    
    return fig, ax


def figure2(x_train, y_train, b, w, color='k'):
    # Генерирует равномерно распределённый признак x
    x_range = np.linspace(0, 1, 101)
    # Вычисляет yhat
    yhat_range = b + w * x_range

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim([0, 3])

    # Набор данных
    ax.scatter(x_train, y_train)
    # Предсказания
    ax.plot(x_range, yhat_range, label='Предсказания модели', c=color, linestyle='--')

    # Аннотации
    ax.annotate('b = {:.4f} w = {:.4f}'.format(b[0], w[0]), xy=(.2, .55), c=color)
    ax.legend(loc=0)
    fig.tight_layout()
    return fig, ax


def figure3(x_train, y_train, b, w):
    fig, ax = figure2(x_train, y_train, b, w)
    
    # Первая точка данных
    x0, y0 = x_train[0][0], y_train[0][0]
    # Первая точка данных
    ax.scatter([x0], [y0], c='r')
    # Вертикальная линия, показывающая ошибку между точкой данных и предсказанием
    ax.plot([x0, x0], [b[0] + w[0] * x0, y0 - .03], c='r', linewidth=2, linestyle='--')
    ax.arrow(x0, y0 - .03, 0, .03, color='r', shape='full', lw=0, length_includes_head=True, head_width=.03)
    ax.arrow(x0, b[0] + w[0] * x0 + .05, 0, -.03, color='r', shape='full', lw=0, length_includes_head=True, head_width=.03)
    # Аннотации
    ax.annotate(r'$error_0$', xy=(.8, 1.5))

    fig.tight_layout()
    return fig, ax


def figure4(x_train, y_train, b, w, bs, ws, all_losses):
    b_minimum, w_minimum = fit_model(x_train, y_train)
    
    
    figure = plt.figure(figsize=(12, 6))

    # 1-й график
    ax1 = figure.add_subplot(1, 2, 1, projection='3d')
    ax1.set_xlabel('b')
    ax1.set_ylabel('w')
    ax1.set_title('Поверхность потерь')

    surf = ax1.plot_surface(bs, ws, all_losses, rstride=1, cstride=1, alpha=.5, cmap=plt.cm.jet, linewidth=0, antialiased=True)
    ax1.contour(bs[0, :], ws[:, 0], all_losses, 10, offset=-1, cmap=plt.cm.jet)


    bidx, widx, _, _ = find_index(b_minimum, w_minimum, bs, ws)
    ax1.scatter(b_minimum, w_minimum, all_losses[bidx, widx], c='k')
    ax1.text(-.3, 2.5, all_losses[bidx, widx], 'Минимум', zdir=(1, 0, 0))
    # Случайное начало
    bidx, widx, _, _ = find_index(b, w, bs, ws)
    ax1.scatter(b, w, all_losses[bidx, widx], c='k')
    # Аннотации
    ax1.text(-.2, -1.5, all_losses[bidx, widx], 'Случайное\n Начало', zdir=(1, 0, 0))

    ax1.view_init(40, 260)
    
    # 2-й график
    ax2 = figure.add_subplot(1, 2, 2)
    ax2.set_xlabel('b')
    ax2.set_ylabel('w')
    ax2.set_title('Поверхность потерь')

    # Поверхность потерь
    CS = ax2.contour(bs[0, :], ws[:, 0], all_losses, cmap=plt.cm.jet)
    ax2.clabel(CS, inline=1, fontsize=10)
    # Минимум
    ax2.scatter(b_minimum, w_minimum, c='k')
    # Случайное начало
    ax2.scatter(b, w, c='k')
    # Аннотации
    ax2.annotate('Случайное начало', xy=(-.2, 0.05), c='k')
    ax2.annotate('Минимум', xy=(.5, 2.2), c='k')
    
    figure.tight_layout()
    return figure, (ax1, ax2)


def figure5(x_train, y_train, b, w, bs, ws, all_losses):
    b_minimum, w_minimum = fit_model(x_train, y_train)

    b_idx, w_idx, fixedb, fixedw = find_index(b, w, bs, ws)

    b_range = bs[0, :]
    w_range = ws[:, 0]
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].set_title('Поверхность потерь')
    axs[0].set_xlabel('b')
    axs[0].set_ylabel('w')
    # Поверхность потерь
    CS = axs[0].contour(bs[0, :], ws[:, 0], all_losses, cmap=plt.cm.jet)
    axs[0].clabel(CS, inline=1, fontsize=10)
    # Минимум
    axs[0].scatter(b_minimum, w_minimum, c='k')
    # Начальная точка
    axs[0].scatter(fixedb, fixedw, c='k')
    # Вертикальное сечение
    axs[0].plot([fixedb, fixedb], w_range[[0, -1]], linestyle='--', c='r', linewidth=2)
    # Аннотации
    axs[0].annotate('Минимум', xy=(.5, 2.2), c='k')
    axs[0].annotate('Случайное начало', xy=(fixedb + .1, fixedw + .1), c='k')

    axs[1].set_ylim([-.1, 15.1])
    axs[1].set_xlabel('w')
    axs[1].set_ylabel('Потери')
    axs[1].set_title('Фикс.: b = {:.2f}'.format(fixedb))
    # Потери
    axs[1].plot(w_range, all_losses[:, b_idx], c='r', linestyle='--', linewidth=2)
    # Начальная точка
    axs[1].plot([fixedw], [all_losses[w_idx, b_idx]], 'or')

    fig.tight_layout()
    return fig, axs


def figure6(x_train, y_train, b, w, bs, ws, all_losses):
    b_minimum, w_minimum = fit_model(x_train, y_train)
    
    b_idx, w_idx, fixedb, fixedw = find_index(b, w, bs, ws)
    
    b_range = bs[0, :]
    w_range = ws[:, 0]
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].set_title('Поверхность потерь')
    axs[0].set_xlabel('b')
    axs[0].set_ylabel('w')
    # Поверхность потерь
    CS = axs[0].contour(bs[0, :], ws[:, 0], all_losses, cmap=plt.cm.jet)
    axs[0].clabel(CS, inline=1, fontsize=10)
    # Минимум
    axs[0].scatter(b_minimum, w_minimum, c='k')
    # Начальная точка
    axs[0].scatter(fixedb, fixedw, c='k')
    # Горизонтальное сечение
    axs[0].plot(b_range[[0, -1]], [fixedw, fixedw], linestyle='--', c='k', linewidth=2)
    # Аннотации
    axs[0].annotate('Минимум', xy=(.5, 2.2), c='k')
    axs[0].annotate('Случайное начало', xy=(fixedb + .1, fixedw + .1), c='k')

    axs[1].set_ylim([-.1, 15.1])
    axs[1].set_xlabel('b')
    axs[1].set_ylabel('Потери')
    axs[1].set_title('Фикс.: w = {:.2f}'.format(fixedw))
    # Потери
    axs[1].plot(b_range, all_losses[w_idx, :], c='k', linestyle='--', linewidth=2)
    # Начальная точка
    axs[1].plot([fixedb], [all_losses[w_idx, b_idx]], 'ok')

    fig.tight_layout()
    return fig, axs


def figure7(b, w, bs, ws, all_losses):
    b_range = bs[0, :]
    w_range = ws[:, 0]

    b_idx, w_idx, fixedb, fixedw = find_index(b, w, bs, ws)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].set_ylim([-.1, 6.1])
    axs[0].set_xlabel('w')
    axs[0].set_ylabel('MSE (потери)')
    axs[0].set_title('Фикс.: b = {:.2f}'.format(fixedb))
    # Красный прямоугольник
    rect = Rectangle((-.3, 2.3),.5, .5)
    pc = PatchCollection([rect], facecolor='r', alpha=.3, edgecolor='r')
    axs[0].add_collection(pc)
    # Потери - фиксированное b
    axs[0].plot(w_range, all_losses[:, b_idx], c='r', linestyle='--', linewidth=2)
    # Начальная точка
    axs[0].plot([fixedw], [all_losses[w_idx, b_idx]], 'or')

    axs[1].set_ylim([-.1, 6.1])
    axs[1].set_xlabel('b')
    axs[1].set_ylabel('MSE (потери)')
    axs[1].set_title('Фикс.: w = {:.2f}'.format(fixedw))
    axs[1].label_outer()
    # Чёрный прямоугольник
    rect = Rectangle((.3, 2.3), .5, .5)
    pc = PatchCollection([rect], facecolor='k', alpha=.3, edgecolor='k')
    axs[1].add_collection(pc)
    # Потери - фиксированное w
    axs[1].plot(b_range, all_losses[w_idx, :], c='k', linestyle='--', linewidth=2)
    # Начальная точка
    axs[1].plot([fixedb], [all_losses[w_idx, b_idx]], 'ok')

    fig.tight_layout()
    return fig, axs


def loss_curves(b_idx, w_idx, b_idx_after, w_idx_after, all_losses):
    # ДО
    # Кривая потерь для b при фиксированном значении w
    loss_fixedw = all_losses[w_idx, :]
    # Кривая потерь для w при фиксированном значении b
    loss_fixedb = all_losses[:, b_idx]
    # Потери «до»
    loss_before = all_losses[w_idx, b_idx]

    # ПОСЛЕ
    # Потери после обновления w
    loss_after_w = all_losses[w_idx_after, b_idx]
    # Потери после обновления b
    loss_after_b = all_losses[w_idx, b_idx_after]
    return loss_fixedb, loss_fixedw, loss_before, loss_after_b, loss_after_w


def calc_gradient(parm_before, parm_after, loss_before, loss_after):
    # Вычисляет изменения в параметре и потерях
    delta_parm = parm_after - parm_before
    delta_loss = loss_after - loss_before
    # Вычисляет градиент для параметра
    manual_grad = delta_loss / delta_parm
    return manual_grad, delta_parm, delta_loss


def figure8(b, w, bs, ws, all_losses):
    b_range = bs[0, :]
    w_range = ws[:, 0]

    # ДО
    b_idx, w_idx, bs_before, ws_before = find_index(b, w, bs, ws)
    # ПОСЛЕ
    b_idx_after, w_idx_after, bs_after, ws_after = find_index(b + .12, w + .12, bs, ws)

    loss_fixedb, loss_fixedw, loss_before, loss_after_b, loss_after_w = loss_curves(b_idx, w_idx, b_idx_after, w_idx_after, all_losses)

    # Вычисляет градиент для b
    manual_grad_b, delta_b, delta_mse_b = calc_gradient(bs_before, bs_after, loss_before, loss_after_b)
    # Вычисляет градиент для w
    manual_grad_w, delta_w, delta_mse_w = calc_gradient(ws_before, ws_after, loss_before, loss_after_w)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].set_ylim([2.3, 2.8])
    axs[0].set_xlim([-.3, .2])
    axs[0].set_xlabel('w')
    axs[0].set_ylabel('MSE (потери)')
    axs[0].set_title('Фикс.: b = {:.2f}'.format(bs_before))
    # Кривая потерь
    axs[0].plot(w_range, loss_fixedb, c='r', linestyle='--', linewidth=2)
    # Точка - до
    axs[0].plot([ws_before], [loss_before], 'or')
    # Точка - после
    axs[0].plot([ws_after], [loss_after_w], 'or')

    # Стрелки
    axs[0].arrow(ws_after, loss_before, .01, 0, color='r', shape='full', lw=0, length_includes_head=True, head_width=.01)
    axs[0].arrow(ws_before, loss_after_w, 0, -0.01, color='r', shape='full', lw=0, length_includes_head=True, head_width=.01)
    axs[0].plot([ws_before, ws_after], [loss_before, loss_before], 'r-', linewidth=1.5)
    axs[0].plot([ws_before, ws_before], [loss_after_w, loss_before], 'r-', linewidth=1.5)

    # Аннотации
    axs[0].annotate(r'$\delta w = {:.2f}$'.format(delta_w), xy=(.0, 2.7), c='k', fontsize=15)
    axs[0].annotate(r'$\delta MSE = {:.2f}$'.format(delta_mse_w), xy=(-.23, 2.45), c='k', fontsize=15)
    axs[0].annotate(r'$\frac{\delta MSE}{\delta w} \approx' + '{:.2f}$'.format(manual_grad_w), xy=(-.05, 2.6), c='k', fontsize=17)

    axs[1].set_ylim([2.3, 2.8])
    axs[1].set_xlim([.3, .8])
    axs[1].set_xlabel('b')
    axs[1].set_ylabel('MSE (потери)')
    axs[1].set_title('Фикс.: w = {:.2f}'.format(ws_before))
    # Кривая потерь
    axs[1].plot(b_range, loss_fixedw, c='k', linestyle='--', linewidth=2)
    # Точка - до
    axs[1].plot([bs_before], [loss_before], 'ok')
    # Точка - после
    axs[1].plot([bs_after], [loss_after_b], 'ok')

    # Стрелки
    axs[1].arrow(bs_after, loss_before, .01, 0, color='k', shape='full', lw=0, length_includes_head=True, head_width=.01)
    axs[1].arrow(bs_before, loss_after_b, 0, -0.01, color='k', shape='full', lw=0, length_includes_head=True, head_width=.01)
    axs[1].plot([bs_before, bs_after], [loss_before, loss_before], 'k-', linewidth=1.5)
    axs[1].plot([bs_before, bs_before], [loss_after_b, loss_before], 'k-', linewidth=1.5)

    # Аннотации
    axs[1].annotate(r'$\delta b = {:.2f}$'.format(delta_b), xy=(.67, 2.7), c='k', fontsize=15)
    axs[1].annotate(r'$\delta MSE = {:.2f}$'.format(delta_mse_b), xy=(.45, 2.32), c='k', fontsize=15)
    axs[1].annotate(r'$\frac{\delta MSE}{\delta b} \approx' + '{:.2f}$'.format(manual_grad_b), xy=(.62, 2.6), c='k', fontsize=17)

    axs[1].label_outer()

    fig.tight_layout()
    return fig, axs


def figure9(x_train, y_train, b, w):
    # Поскольку мы обновили b и w, давайте восстановим начальные значения
    # Вот именно в этом случае полезно использовать случайный seed, например
    np.random.seed(42)
    b_initial = np.random.randn(1)
    w_initial = np.random.randn(1)

    fig, ax = figure2(x_train, y_train, b_initial, w_initial)

    # Генерирует равномерно распределённый признак x
    x_range = np.linspace(0, 1, 101)
    # Предсказания модели для обновлённых параметров
    yhat_range = b + w * x_range
    # Обновление предсказаний
    ax.plot(x_range, yhat_range, label='Использование параметров\nпосле одного обновления', c='g', linestyle='--')
    # Аннотации
    ax.annotate('b = {:.4f} w = {:.4f}'.format(b[0], w[0]), xy=(.2, .95), c='g')

    fig.tight_layout()
    return fig, ax


def figure10(b, w, bs, ws, all_losses, manual_grad_b, manual_grad_w, lr):
    b_range = bs[0, :]
    w_range = ws[:, 0]

    # ДО
    b_idx, w_idx, bs_before, ws_before = find_index(b, w, bs, ws)
    # ПОСЛЕ
    new_b_idx, new_w_idx, bs_after, ws_after = find_index(bs_before - lr * manual_grad_b,
                                                          ws_before - lr * manual_grad_w,
                                                          bs,
                                                          ws)
    # Потери до
    loss_before = all_losses[w_idx, b_idx]
    loss_fixedb = all_losses[:, b_idx]
    loss_fixedw = all_losses[w_idx, :]
    loss_after_b = all_losses[w_idx, new_b_idx]
    loss_after_w = all_losses[new_w_idx, b_idx]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].set_ylim([-.1, 6.1])
    axs[0].set_xlabel('w')
    axs[0].set_ylabel('MSE (потери)')
    axs[0].set_title('Фикс.: b = {:.2f}'.format(bs_before))

    # Кривая потерь для w при фиксированном b
    axs[0].plot(w_range, loss_fixedb, c='r', linestyle='--', linewidth=2)
    # w до обновления
    axs[0].plot([ws_before], [loss_before], 'or')

    # Стрелки
    axs[0].arrow(ws_after, loss_before, .1, 0, color='r', shape='full', lw=0, length_includes_head=True, head_width=.1)
    axs[0].plot([ws_before, ws_after], [loss_before, loss_before], 'r-', linewidth=1.5)
    axs[0].plot([ws_after], [loss_after_w], 'or')

    # Аннотации
    axs[0].annotate(r'$\eta = {:.2f}$'.format(lr), xy=(1.6, 5.5), c='k', fontsize=17)
    axs[0].annotate(r'$-\eta \frac{\delta MSE}{\delta b} \approx' + '{:.2f}$'.format(-lr * manual_grad_w), xy=(1, 2), c='k', fontsize=17)

    axs[1].set_ylim([-.1, 6.1])
    axs[1].set_xlabel('b')
    axs[1].set_ylabel('MSE (потери)')
    axs[1].set_title('Фикс.: w = {:.2f}'.format(ws_before))
    axs[1].label_outer()

    # Кривая потерь для b при фиксированном значении w
    axs[1].plot(b_range, loss_fixedw, c='k', linestyle='--', linewidth=2)
    # b перед обновлением
    axs[1].plot([bs_before], [loss_before], 'ok')

    # Стрелки
    axs[1].arrow(bs_after, loss_before, .1, 0, color='k', shape='full', lw=0, length_includes_head=True, head_width=.1)
    axs[1].plot([bs_before, bs_after], [loss_before, loss_before], 'k-', linewidth=1.5)
    axs[1].plot([bs_after], [loss_after_b], 'ok')

    # Аннотации
    axs[1].annotate(r'$\eta = {:.2f}$'.format(lr), xy=(0.6, 5.5), c='k', fontsize=17)
    axs[1].annotate(r'$-\eta \frac{\delta MSE}{\delta w} \approx' + '{:.2f}$'.format(-lr * manual_grad_b), xy=(1, 2), c='k', fontsize=17)

    fig.tight_layout()
    return fig, axs


def figure14(x_train, y_train, b, w, bad_bs, bad_ws, bad_x_train):
    bad_b_range = bad_bs[0, :]
    bad_w_range = bad_ws[:, 0]

    # Таким образом мы пересчитываем поверхность X_TRAIN, используя новые диапазоны
    all_predictions = np.apply_along_axis(func1d=lambda x: bad_bs + bad_ws * x, axis=1, arr=x_train)
    all_errors = (all_predictions - y_train.reshape(-1, 1, 1))
    all_losses = (all_errors ** 2).mean(axis=0)

    # Затем мы вычисляем поверхность для BAD_X_TRAIN, используя новые диапазоны
    bad_all_predictions = np.apply_along_axis(func1d=lambda x: bad_bs + bad_ws * x, axis=1, arr=bad_x_train)
    bad_all_errors = (bad_all_predictions - y_train.reshape(-1, 1, 1))
    bad_all_losses = (bad_all_errors ** 2).mean(axis=0)

    b_idx, w_idx, fixedb, fixedw = find_index(b, w, bad_bs, bad_ws)

    b_minimum, w_minimum = fit_model(x_train, y_train)

    bad_b_minimum, bad_w_minimum = fit_model(bad_x_train, y_train)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].set_xlabel('b')
    axs[0].set_ylabel('w')
    axs[0].set_title('Поверхность потерь - До')

    # Поверхность потерь - ДО
    CS = axs[0].contour(bad_bs[0, :], bad_ws[:, 0], all_losses, cmap=plt.cm.jet)
    axs[0].clabel(CS, inline=1, fontsize=10)
    # Точка минимума - ДО
    axs[0].scatter(b_minimum, w_minimum, c='k')
    # Изначальная случайная точка
    axs[0].scatter(fixedb, fixedw, c='k')

    # Вертикальное сечение
    axs[0].plot([fixedb, fixedb], bad_w_range[[0, -1]], linestyle='--', c='r', linewidth=2)
    # Горизонтальное сечение
    axs[0].plot(bad_b_range[[0, -1]], [fixedw, fixedw], linestyle='--', c='k', linewidth=2)

    # Аннотации
    axs[0].annotate('Минимум', xy=(.5, .35), c='k')
    axs[0].annotate('Случайное начало', xy=(fixedb - .6, fixedw - .3), c='k')

    axs[1].set_xlabel('b')
    axs[1].set_ylabel('w')
    axs[1].set_title('Поверхность потерь - После')

    # Поверхность потерь - ПОСЛЕ
    CS = axs[1].contour(bad_bs[0, :], bad_ws[:, 0], bad_all_losses, cmap=plt.cm.jet)
    axs[1].clabel(CS, inline=1, fontsize=10)
    # Точка минимума - ПОСЛЕ
    axs[1].scatter(bad_b_minimum, bad_w_minimum, c='k')
    # Изначальная случайная точка
    axs[1].scatter(fixedb, fixedw, c='k')

    # Вертикальное сечение
    axs[1].plot([fixedb, fixedb], bad_w_range[[0, -1]], linestyle='--', c='r', linewidth=2)
    # Горизонтальное сечение
    axs[1].plot(bad_b_range[[0, -1]], [fixedw, fixedw], linestyle='--', c='k', linewidth=2)

    # Аннотации
    axs[1].annotate('Минимум', xy=(.5, .35), c='k')
    axs[1].annotate('Случайное Начало', xy=(fixedb - .6, fixedw - .3), c='k')

    fig.tight_layout()
    return fig, axs


def figure15(x_train, y_train, b, w, bad_bs, bad_ws, bad_x_train):
    bad_b_range = bad_bs[0, :]
    bad_w_range = bad_ws[:, 0]

    # Таким образом мы пересчитываем поверхность для X_TRAIN, используя новые диапазоны
    all_predictions = np.apply_along_axis(func1d=lambda x: bad_bs + bad_ws * x, axis=1, arr=x_train)
    all_errors = (all_predictions - y_train.reshape(-1, 1, 1))
    all_losses = (all_errors ** 2).mean(axis=0)

    # Затем мы вычисляем поверхность для BAD_X_TRAIN, используя новые диапазоны
    bad_all_predictions = np.apply_along_axis(func1d=lambda x: bad_bs + bad_ws * x, axis=1, arr=bad_x_train)
    bad_all_errors = (bad_all_predictions - y_train.reshape(-1, 1, 1))
    bad_all_losses = (bad_all_errors ** 2).mean(axis=0)

    bad_b_idx, bad_w_idx, fixedb, fixedw = find_index(b, w, bad_bs, bad_ws)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].set_ylim([-.1, 15.1])
    axs[0].set_xlim([-1, 3.2])
    axs[0].set_xlabel('w')
    axs[0].set_ylabel('Потери')
    axs[0].set_title('Фикс.: b = {:.2f}'.format(fixedb))

    # Кривая потерь для b при фиксированном w - ДО
    axs[0].plot(bad_w_range, all_losses[:, bad_b_idx], c='r', linestyle='--', linewidth=1, label='До')
    axs[0].plot([fixedw], [all_losses[bad_w_idx, bad_b_idx]], 'or')
    # Кривая потерь для b при фиксированном w — ПОСЛЕ
    axs[0].plot(bad_w_range, bad_all_losses[:, bad_b_idx], c='r', linestyle='--', linewidth=2, label='После')
    axs[0].plot([fixedw], [bad_all_losses[bad_w_idx, bad_b_idx]], 'or')

    axs[0].legend()

    axs[1].set_ylim([-.1, 15.1])
    axs[1].set_xlabel('b')
    axs[1].set_ylabel('Потери')
    axs[1].set_title('Фикс.: w = {:.2f}'.format(fixedw))

    # Кривая потерь для w при фиксированном b - ДО
    axs[1].plot(bad_b_range, all_losses[bad_w_idx, :], c='k', linestyle='--', linewidth=1, label='До')
    axs[1].plot([fixedb], [all_losses[bad_w_idx, bad_b_idx]], 'ok')
    # Кривая потерь для w при фиксированном b — ПОСЛЕ
    axs[1].plot(bad_b_range, bad_all_losses[bad_w_idx, :], c='k', linestyle='--', linewidth=2, label='После')
    axs[1].plot([fixedb], [bad_all_losses[bad_w_idx, bad_b_idx]], 'ok')

    axs[1].legend()

    fig.tight_layout()
    return fig, axs


def figure17(x_train, y_train, scaled_bs, scaled_ws, bad_x_train, scaled_x_train):
    # Таким образом мы пересчитываем поверхность для X_TRAIN, используя новые диапазоны
    all_predictions = np.apply_along_axis(func1d=lambda x: scaled_bs + scaled_ws * x, axis=1, arr=x_train)
    all_errors = (all_predictions - y_train.reshape(-1, 1, 1))
    all_losses = (all_errors ** 2).mean(axis=0)

    # Таким образом мы пересчитываем поверхность для BAD_X_TRAIN, используя новые диапазоны
    bad_all_predictions = np.apply_along_axis(func1d=lambda x: scaled_bs + scaled_ws * x, axis=1, arr=bad_x_train)
    bad_all_errors = (bad_all_predictions - y_train.reshape(-1, 1, 1))
    bad_all_losses = (bad_all_errors ** 2).mean(axis=0)

    # Затем мы рассчитываем поверхность для SCALED_X_TRAIN, используя новые диапазоны
    scaled_all_predictions = np.apply_along_axis(func1d=lambda x: scaled_bs + scaled_ws * x, axis=1, arr=scaled_x_train)
    scaled_all_errors = (scaled_all_predictions - y_train.reshape(-1, 1, 1))
    scaled_all_losses = (scaled_all_errors ** 2).mean(axis=0)

    b_minimum, w_minimum = fit_model(x_train, y_train)

    bad_b_minimum, bad_w_minimum = fit_model(bad_x_train, y_train)

    scaled_b_minimum, scaled_w_minimum = fit_model(scaled_x_train, y_train)

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))

    axs[0].set_xlabel('b')
    axs[0].set_ylabel('w')
    axs[0].set_title('П-ть потерь - изначальная')

    # Поверхность потерь - ИЗНАЧАЛЬНАЯ
    CS = axs[0].contour(scaled_bs[0, :], scaled_ws[:, 0], all_losses, cmap=plt.cm.jet)
    axs[0].clabel(CS, inline=1, fontsize=10)
    # Точка минимума - ИЗНАЧАЛЬНАЯ
    axs[0].scatter(b_minimum, w_minimum, c='k')

    # Аннотации
    axs[0].annotate('Минимум', xy=(.3, 1.6), c='k')

    axs[1].set_xlabel('b')
    axs[1].set_ylabel('w')
    axs[1].set_title('П-ть потерь - "Плохая"')

    # Поверхность потерь - ПЛОХАЯ
    CS = axs[1].contour(scaled_bs[0, :], scaled_ws[:, 0], bad_all_losses, cmap=plt.cm.jet)
    axs[1].clabel(CS, inline=1, fontsize=10)
    # Точка минимума - ПЛОХАЯ
    axs[1].scatter(bad_b_minimum, bad_w_minimum, c='k')

    # Аннотации
    axs[1].annotate('Минимум', xy=(.3, -.25), c='k')

    axs[2].set_xlabel('b')
    axs[2].set_ylabel('w')
    axs[2].set_title('П-ть потерь - масштабированная')

    # Поверхность потерь - МАСШТАБИРОВАННАЯ
    CS = axs[2].contour(scaled_bs[0, :], scaled_ws[:, 0], scaled_all_losses, cmap=plt.cm.jet)
    axs[2].clabel(CS, inline=1, fontsize=10)
    # Точка минимума - МАСШТАБИРОВАННАЯ
    axs[2].scatter(scaled_b_minimum, scaled_w_minimum, c='k')

    # Аннотации
    axs[2].annotate('Минимум', xy=(1.3, .15), c='k')

    fig.tight_layout()
    return fig, axs


def figure18(x_train, y_train):
    b_minimum, w_minimum = fit_model(x_train, y_train)
    # Генерирует равномерно распределённый признак x
    x_range = np.linspace(0, 1, 101)
    # Вычисляет yhat
    yhat_range = b_minimum + w_minimum * x_range

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim([0, 3.1])

    # Набор данных
    ax.scatter(x_train, y_train)
    # Предсказания
    ax.plot(x_range, yhat_range, label='Предсказания конечной модели', c='k', linestyle='--')

    # Аннотации
    ax.annotate('b = {:.4f} w = {:.4f}'.format(b_minimum, w_minimum), xy=(.4, 1.5), c='k', rotation=34)
    ax.legend(loc=0)
    fig.tight_layout()
    return fig, ax
