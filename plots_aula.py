import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

""" This is the code for all plots and simulations using during the class 
about linear systems in 2D """


plt.style.use('default')
plt.figure(dpi=600) # To improve quality

########################### GENERAL CODE ######################################

def solve_linear_system(X0, t, a, b, c, d):
    """ Returns the solutions x(t) and y(t) of a 2D linear system 
        X0 = (x(0), y(0)) --- vector of initial conditions
    """
    
    def f(X, t, a, b, c, d):
        x = X[0]
        y = X[1]
        dxdt = a*x + b*y
        dydt = c*x + d*y
        
        return [dxdt, dydt]
            
        
    solution = odeint(f, X0, t, args=(a, b, c, d))
    
    return solution


def plot_vector_field(solutions, a, b, c, d, aux = 0.1, reverse = False, flow = False):
    """ Plots the vector field in the phase space.
    
    solutions --- a list containing trajectories for different initial conditions
    The model is fixed by the parameters a, b, c, and d.
    
    If flow = True, performs a streamplot.
    
    aux --- just a 'padding' to adjust the size of the plot
    """
    
    if reverse:   # if reverse = True, solves backwards in time
        k = -1
    else:
        k = 1
        
    upper_limits_x = []
    lower_limits_x = []
    upper_limits_y = []
    lower_limits_y = []
    
    for i in range(len(solutions)):
        
        upper_limits_x.append(max(solutions[i][:,0]))
        lower_limits_x.append(min(solutions[i][:,0]))
        
        upper_limits_y.append(max(solutions[i][:,1]))
        lower_limits_y.append(min(solutions[i][:,1]))
        
    global_max_x = max(upper_limits_x)
    global_min_x = min(lower_limits_x)
    global_max_y = max(upper_limits_y)
    global_min_y = min(lower_limits_y)
        
 
    y_grid = np.linspace(global_min_y - aux, global_max_y + aux, 10)
    x_grid = np.linspace(global_min_x - aux, global_max_x + aux, 20)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Velocity field
    dx = k*(a*X + b*Y)  
    dy = k*(c*X + d*Y)
    
    if flow:
        
        plt.streamplot(X, Y, dx, dy, color = 'k')
        
    else:
        
        plt.quiver(X, Y, dx, dy, headwidth = 4, headlength = 5, headaxislength = 5)
    

    
  
##############################################################################
##############################################################################
######################### SPECIFIC EXAMPLES ##################################
##############################################################################
##############################################################################
    
########################## Simple Harmonic oscillator ########################
    
# t = np.linspace(0, 15, 3000)
# a = 0
# b = 1
# c = -2
# d = 0 
 
# X0  = (1, 2)
# sol = solve_linear_system(X0, t, a, b, c, d)
        
# #### Plot 1
# plt.plot(t, sol[:,0])
# plt.plot(t, sol[:,1])
# plt.xlabel("$t$", fontsize = 13)
# plt.ylabel("Solutions", fontsize = 13)
# plt.legend(["Position", "Velocity"], loc = 'best')
    
### Plot 2
# marker_on = [1]
# plt.plot(sol[:,0], sol[:,1], '-gD', color = 'r', markevery = marker_on)
# plt.xlabel("$x$", fontsize = 13)
# plt.ylabel("$v$", fontsize = 13)

# Initial_conditions = [(0, 1), (1,2), (-3, 1)]
# solutions = []
# for X0 in Initial_conditions:
#     sol = solve_linear_system(X0, t, a, b, c, d)
#     plt.plot(sol[:,0], sol[:,1], label = "C = "+str(round(X0[0]**2 + X0[1]**2, 2)))
#     solutions.append(sol)
    
# plt.legend()    
# plot_vector_field(solutions, a, b, c, d, flow=False)
# plt.xlabel("$x$", fontsize = 13)
# plt.ylabel("$v$", fontsize = 13)

###############################################################################
        

################################ Initial discussion #########################

# t = np.linspace(0, 10, 1000)
# a = 0  # -2, -1, -0.5, 0, 1
# b = 0
# c = 0
# d = -1
# Initial = [(-2,3), (-1, -3), (0, 3), (1, -3), (2, 3)]

# solutions = []

# for X0 in Initial:
#     sol = solve_linear_system(X0, t, a, b, c, d)
#     plt.plot(sol[:,0], sol[:,1])
#     solutions.append(sol)
    
   
# plot_vector_field(solutions, a, b, c, d)
# plt.xlabel("$x$", fontsize = 13)
# plt.ylabel("$y$", fontsize = 13)
# plt.title('a = '+str(a), fontsize = 15)

##############################################################################
        
        
############## Attractive but not Liapunov stable ############################
     
# x = np.linspace(-2, 2, 1000)
# y = np.linspace(-2, 2, 1000)
# X, Y = np.meshgrid(x, y)

# v_x = X + X*Y -(X + Y)*np.sqrt(X**2 + Y**2)
# v_y = Y - X**2 + (X - Y)*np.sqrt(X**2 + Y**2)

# plt.streamplot(X, Y, v_x, v_y, density = 1, arrowsize = 2, color = 'r')

##############################################################################
        

#################### Stable node prototype ##################################

# t = np.linspace(0, 10, 1000) 

# a = -3
# b = 2
# c = 1
# d = -2

# Initial = [(2, -7), (5, -7), (-6, -1), (6, 1), (2, 6), (-2, 6), (-5, 0), (6, -1)]



# # Plotting eigendirections

# x = np.linspace(-5, 5, 2000)
# plt.plot(x, x, label = 'Slow eigendirection', color = 'b')
# plt.plot(x, -x/2, label = 'Fast eigendirection', color = 'r')

# solutions = []

# for X0 in Initial:
#     sol = solve_linear_system(X0, t, a, b, c, d)
#     #plt.plot(sol[:,0], sol[:,1], color = 'g')
#     solutions.append(sol)
    
   
# #plot_flow(solutions, a, b, c, d)
# plot_vector_field(solutions, a, b, c, d)
# plt.xlabel("$x$", fontsize = 13)
# plt.ylabel("$y$", fontsize = 13)
# plt.legend()

##############################################################################
        

########################### Saddle-node propotype ############################

# t = np.linspace(0, 1, 1000)
# a = 1
# b = 1
# c = 4
# d = -2
# Initial = [(-40, 90), (40, -90), (-8, 95), (8, -95)]

# # Plotting eigendirections

# x = np.linspace(-100, 100, 2000)
# plt.plot(x, x, label = 'Unstable manifold', color = 'r')
# x = np.linspace(-25, 25, 2000)
# plt.plot(x, -4*x, label = 'Stable manifold', color = 'b')

# solutions = []

# for X0 in Initial:
#     sol = solve_linear_system(X0, t, a, b, c, d)
#     plt.plot(sol[:,0], sol[:,1], color = 'g')
#     solutions.append(sol)
    
   
# plot_vector_field(solutions, a, b, c, d)
# plt.xlabel("$x$", fontsize = 13)
# plt.ylabel("$y$", fontsize = 13)
# plt.legend(loc = 'best')

#############################################################################

        
##################### Canonical cases (center and spiral) ##################


# t = np.linspace(0, 200, 2000)
# a = -1
# b = -2
# c = 2
# d = -1
# Initial = [(20, 0)]


# solutions = []

# for X0 in Initial:
#     sol = solve_linear_system(X0, t, a, b, c, d)
#     plt.plot(sol[:,0], sol[:,1])
#     solutions.append(sol)
    
   
# plot_vector_field(solutions, a, b, c, d, aux = 0.5)

# plt.xlabel("$x$", fontsize = 13)
# plt.ylabel("$y$", fontsize = 13)




#############################################################################


################################## Center ##################################

# t = np.linspace(0, 20, 2000)
# a = 5
# b = 2
# c = -17
# d = -5
# Initial = [(1, 3), (-2, 5), (3, 1)]


# solutions = []

# for X0 in Initial:
#     sol = solve_linear_system(X0, t, a, b, c, d)
#     plt.plot(sol[:,0], sol[:,1])
#     solutions.append(sol)
    
   
# plot_vector_field(solutions, a, b, c, d, aux = 0.5)
# plt.xlabel("$x$", fontsize = 13)
# plt.ylabel("$y$", fontsize = 13)

##############################################################################


############################### Spiral #######################################

# t = np.linspace(0, 50, 2000)
# a = 1
# b = -4
# c = 1
# d = -2
# Initial = [(20, 20), (-20, 20), (20, -20), (-20, 20)]


# solutions = []

# for X0 in Initial:
#     sol = solve_linear_system(X0, t, a, b, c, d)
#     plt.plot(sol[:,0], sol[:,1])
#     solutions.append(sol)
    
   
# plot_vector_field(solutions, a, b, c, d, aux = 0.1)
# plt.xlabel("$x$", fontsize = 13)
# plt.ylabel("$y$", fontsize = 13)

##############################################################################
        

############################ Star node ####################################

# t = np.linspace(0, 50, 2000)
# a = -1
# b = 0
# c = 0
# d = -1
# Initial = [(20, 20), (-20, 20), (20, -20), (-20, -20)]


# solutions = []

# for X0 in Initial:
#     sol = solve_linear_system(X0, t, a, b, c, d)
#     plt.plot(sol[:,0], sol[:,1])
#     solutions.append(sol)
    
   
# plot_vector_field(solutions, a, b, c, d, aux = 0.1)
# plt.xlabel("$x$", fontsize = 13)
# plt.ylabel("$y$", fontsize = 13)

##############################################################################
        

########################### Degenerate node ##################################

# t = np.linspace(0, 10, 2000)
# a = -1
# b = 1
# c = 0
# d = -1
# Initial = [(-10, -8), (-10, 8),(-6, -8), (-6, 8), (-2, -8), (-2, 8), (2, -8), 
#            (2, 8), (6, -8), (6, 8), (10, -8), (10, 8)]


# # Plot eigendirection

# x = np.linspace(-10, 10, 5000)
# y = np.zeros(x.shape)

# plt.plot(x, y, label = 'Eigendirection', color = 'r')

# solutions = []

# for X0 in Initial:
#     sol = solve_linear_system(X0, t, a, b, c, d)
#     plt.plot(sol[:,0], sol[:,1], color = 'b')
#     solutions.append(sol)
    
   
# plot_vector_field(solutions, a, b, c, d, aux = 0.1)
# plt.xlabel("$x$", fontsize = 13)
# plt.ylabel("$y$", fontsize = 13)
##############################################################################
        
        
################# Degenerate animation ###################################

# aux = np.arange(-4, -1, 0.05)

# t = np.linspace(0, 10, 2000)
# a = -1
# b = 1
# c = 0
# d = [-1]

# Initial = [(6, 12), (-6, 9), (0, -12), (6, -10)]


# x = np.linspace(-5, 5, 1000)
# y = np.zeros(x.shape)

# for i in range(len(d)):
#     plt.plot(x, y, color = 'b')
#     plt.plot(x, (1+d[i])*x, color = 'r')

#     solutions = []
#     for X0 in Initial:
    
#         sol = solve_linear_system(X0, t, a, b, c, d[i])
#         solutions.append(sol)
#         #plt.plot(sol[:,0], sol[:,1], color = 'g')

#     plot_vector_field(solutions, a, b, c, d[i], aux = 2, flow = True)

#     plt.xlabel("$x$", fontsize = 13)
#     plt.ylabel("$y$", fontsize = 13)
#     plt.savefig('batata-'+str(i))
#     plt.close()
    
############################################################################


################### Damped harmonic oscillator ###############################


## Plots for animation ####
        
# t = np.linspace(0, 10, 2000)
# a = 0
# b = 1
# c = -1
# aux = np.arange(0, 1, 0.01)
# d = [-n for n in aux]
# X0 = (2, 5)


# for i in range(len(d)):
#     sol = solve_linear_system(X0, t, a, b, c, d[i])

#     plt.plot(sol[:,0], sol[:,1], color = 'r')
#     plt.xlabel("$x$", fontsize = 13)
#     plt.ylabel("$v$", fontsize = 13)
#     plt.savefig("frame-"+str(i))
#     plt.close()
        
        
### Comparing solutions with phase portrait ###

# fig = plt.figure(figsize = (8, 14))

# p1 = fig.add_subplot(211)
# p2 = fig.add_subplot(212)


# t = np.linspace(0, 20, 2000)
# a = 0
# b = 1
# c = -1
# d = -1.5

# init = [(2, 1), (3, 5), (-3, 5)]

# for X0 in init:
#     sol = solve_linear_system(X0, t, a, b, c, d)
#     p1.plot(t, sol[:,0])
#     p2.plot(sol[:,0], sol[:,1])
    
# p1.set_xlabel("Time", fontsize = 14)
# p1.set_ylabel("Position", fontsize = 14)
# p1.set_title("Trajectories in real space", fontsize = 16)
# p2.set_xlabel("Position", fontsize = 14)
# p2.set_ylabel("Velocity", fontsize = 14)
# p2.set_title("Trajectories in phase space", fontsize = 16)
    

        



    

   


    
    
    

    





    