import numpy as np
import matplotlib.pyplot as plt


plt.style.use('seaborn')

def f1(x, h):
    return 1 - h/x

def f2(x, R):
    return R/np.sqrt(1 + x**2)

# x_total = np.arange(-5, 5, 0.01)
# x_mais = np.arange(-5, -0.01, 0.01)
# x_menos = np.arange(0.01, 5, 0.01)



# Rs = [0.5, 1, 2, 3]

# param = [0.5, 1, 2]

# fig = plt.figure(figsize=(15,10))
# a1 = fig.add_subplot(221)
# a2 = fig.add_subplot(222)
# a3 = fig.add_subplot(223)
# a4 = fig.add_subplot(224)
# plots = [a1,a2,a3,a4]

# fig.tight_layout(pad=5.0)   # spacing between subplots


# assintota = np.array([1 for n in range(len(x_total))])

# j = 0
# for ax in plots:

#     ax.plot(x_total, assintota, color = 'r', markersize = 3, linestyle = 'dotted')
#     ax.plot(x_total, f2(x_total, Rs[j]), label = '$y_{2}(u)$', color = 'k')
#     for h in param:
#         ax.plot(x_mais, f1(x_mais, h), label =  'h = ' + str(h))
#         ax.plot(x_menos, f1(x_menos, h), color = ax.get_lines()[-1].get_color())


#     ax.set_ylim([-4,4])
#     ax.legend()
#     ax.set_title('R = '+str(Rs[j]), fontsize = 15)
    
#     j = j + 1

u = np.linspace(-1, 1, 2000)
h = -u**3
r1 = np.power(1 + u**2, 3/2) - 1
r2 = 3/2*(u**2)
plt.plot(r1, h, color = 'r')
plt.plot(r2, h, color = 'b')
plt.xlabel("r", fontsize = 13)
plt.ylabel("h", fontsize = 13)
plt.legend(['Exact', 'Approximation'])


    
