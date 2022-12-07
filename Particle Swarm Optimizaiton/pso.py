
#	Ahmed Al Ali
#	Analysis of Particle Swarm Optimization
#   19-Nov-2022
#   Natural Computing , University of Edinburgh
#########################################################################


#   Import Dependiencies
#########################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from fitness import sphere,rastrigin

#   Particle Class
##################

class Particle:

    def __init__(self, d, bound):

        '''
        Parameters
            function: fitness function
            d: dimension
            range: Axis Boundary
        Info
            Setting Up Particles
        '''
        self.d=d
        self.maxx=bound
        self.minx=-1*bound
        self.vmax=0.2*(self.maxx)
        self.vmin=0.2*(self.minx)
        self.position = np.random.uniform(low=self.minx, high=self.maxx, size=self.d)
        self.velocity = np.random.uniform(low=self.vmin,high=self.vmax, size=self.d)
        self.best_particle_pos = self.position

        self.fitness=rastrigin(self.position,self.d)
        self.best_particle_fitness = self.fitness

    
    def update_velocity(self, w, c1, c2, pbest,gbest):

        '''
        Parameters
            w: Inertia Weight
            c1: Cognitive Parameter
            c2: Social Learning Parameter
            gbest: Global Best Position as a Swarm of Shape (d,1)
            pbest: Particle Best Position of Shape (d,1)
        Info
            Update Particle Velcoity
        '''

        cur_vel = self.velocity
        r1 = np.random.uniform(low=0, high=1, size=self.d)
        r2 = np.random.uniform(low=0, high=1, size=self.d)
        c1r1 = np.multiply(c1, r1)
        c2r2 = np.multiply(c2, r2)
        best_self_dif = np.subtract(pbest, self.position)
        best_swarm_dif = np.subtract(gbest, self.position)
        # the next line is the main equation, namely the velocity update,
        # the velocities are added to the positions at swarm level
        new_vel = w * cur_vel + np.multiply(c1r1, best_self_dif) + np.multiply(c2r2, best_swarm_dif)
        self.velocity = new_vel
        return new_vel
    
    def update_position(self,pos):

        '''
        Parameter
            function: the fitness function
            pos: Position of particle array
        Info
            Update Particle Position
        '''
        
        self.position = pos
        self.fitness = rastrigin(self.position,self.d)
        if self.fitness<self.best_particle_fitness:
            self.best_particle_fitness = self.fitness
            self.best_particle_pos = pos

#   Particle Swarm Optimization Algorithm
###############################################

class PSO:

    def __init__(self,N, w, c1, c2, d,Tmax, bound,update_w=False, update_c1=False, update_c2=False, plot=False, verbose=False):

        '''
        The Symbols were adapted as per [1] and [2]

        Parameters
            function: the fitness function
            N: Population of Particles Size
            n: Total Dimensions
            w: Inertia Weight
            update_w: Boolean Value which indicates if w changes with iteration or stays constant
            c1: Individual Cognitive Parameter
            update_c1: Boolean Value which indicates if c1 changes with iteration or stays constant
            c2: Social Learning Parameter
            update_c2: Boolean Value which indicates if c2 changes with iteration or stays constant
            Tmax: Maximum Iteration
            Range: Axis range of each dimension
            plot: Boolean Value which indicates if cst function should be plotted
            verbose: Boolean Value which indicates if global fitness should be prinited for each iteration

        According to M. Clerc AND J. Kennedy [3], to define a standard for PSO. The Optimal
        Parameters are w=0.72984 and c1 + c2 >= 4 specifcally c1 = 2.05 and c2 = 2.05

        '''

        self.N = N
        self.w = w
        self.d=d
        self.c1, self.c2 = c1, c2
        self.Tmax = Tmax
        self.update_w = update_w 
        self.update_c1 = update_c1
        self.update_c2 = update_c2
        self.plot = plot
        self.verbose = verbose

        self.swarm = [Particle(d,bound) for i in range(N)]
        self.best_swarm_pos = np.random.uniform(low=-500, high=500, size=d)
        self.best_swarm_fitness = 1e100


    def update_coeff(self):
        
        '''
        Based on the idea inpired by G. Sermpinis [5] and [6]
        Suggested updating the coefficent C1 and C2 per itertion
        Additionally the lienar decay of the paremter 'w' was intially prposed by
        Yuhui and Russ Y.H. Shi and R.C. Eberhart [4]
    
        '''
        if self.update_w:
            self.w = 0.9 - 0.5*(self.t/self.Tmax)
        if self.update_c1:
            self.c1 = 3.5 - 3*(self.t/self.Tmax)
        if self.update_c2:
            self.c2 = 0.5 + 3*(self.t/self.Tmax)

    def plot(self):
        position_vlaues=self.position_values
        iterations=list(range(1,self.t+1))

        plt.plot(iterations,position_vlaues,marker='.',color="black",markersize=4)
        plt.title("w="+str(self.w)+",c1="+str(self.c1)+",c2="+str(self.c2))
        plt.ylabel('Particle Position (x)')
        plt.xlabel('Iternation Number (t)')
        return(plt)

    def run(self,tuning,plot):
            if tuning:
                fitness_criterion=10
                for t in range(self.Tmax):

                        self.t=t
                        self.update_coeff()

                        for p in range(len(self.swarm)):

                            particle = self.swarm[p]
                            new_position = particle.position + particle.update_velocity(self.w, self.c1, self.c2,particle.best_particle_pos, self.best_swarm_pos)

                            if self.best_swarm_fitness <= fitness_criterion:
                                print('Time:', t, 'w:', self.w, 'c1:', self.c1, 'c2:', self.c2, 'Best Pos:',
                                      self.best_swarm_pos, 'Best Fit:', self.best_swarm_fitness)
                                print('Fitness Criteria Reached')
                                return (self.w, self.c1, self.c2)

                            elif (new_position @ new_position > 1.0e+22 or self.t==self.Tmax-1):
                                print('Time:', t, 'w:', self.w, 'c1:', self.c1, 'c2:', self.c2, 'Best Pos:',
                                      self.best_swarm_pos, 'Best Fit:', self.best_swarm_fitness)
                                print("Diverges")
                                return(None,None,None)



                            self.swarm[p].update_position(new_position)
                            new_fitness = rastrigin(new_position, self.d)

                            if new_fitness < self.best_swarm_fitness:  # to update the global best both
                                # position (for velocity update) and
                                # fitness (the new group norm) are needed
                                self.best_swarm_fitness = new_fitness
                                self.best_swarm_pos = new_position

                        if t % 100 == 0:  # we print only two components even it search space is high-dimensional
                            print("Time: %6d,w:%9.4f,c1:%9.4f,c2:%9.4f,  Best Fitness: %14.6f,  Best Pos: %9.4f,%9.4f" % (
                            t,self.w,self.c1,self.c2, self.best_swarm_fitness, self.best_swarm_pos[0], self.best_swarm_pos[1]), end=" ")
                            if self.d > 2:
                                print('...')
                            else:
                                print('')

            else:
                fitness_criterion = 1e-08
                self.goal = 0
                self.position_values=[]
                for t in range(self.Tmax):
                    self.t = t

                    for p in range(len(self.swarm)):

                        particle = self.swarm[p]
                        new_position = particle.position + particle.update_velocity(self.w, self.c1, self.c2,particle.best_particle_pos,self.best_swarm_pos)

                        if self.best_swarm_fitness <= fitness_criterion:
                            print('Time:', t, 'w:', self.w, 'c1:', self.c1, 'c2:', self.c2, 'Best Pos:',
                                  self.best_swarm_pos, 'Best Fit:', self.best_swarm_fitness)
                            self.goal = 1
                            print('Fitness Criteria Reached')
                            if plot:
                                return(PSO.plot(self))
                            else:
                                return (self.t, self.goal, self.best_swarm_pos, self.best_swarm_fitness)

                        elif (new_position @ new_position > 1.0e+22  or self.t==self.Tmax-1):
                            print('Time:', t, 'w:', self.w, 'c1:', self.c1, 'c2:', self.c2, 'Best Pos:',
                                  self.best_swarm_pos, 'Best Fit:', self.best_swarm_fitness)
                            print('Diverges')
                            if plot:
                                return(PSO.plot(self))
                            else:
                                return (None, self.goal,None,None)

                        self.swarm[p].update_position(new_position)
                        new_fitness = rastrigin(new_position, self.d)

                        if new_fitness < self.best_swarm_fitness:  # to update the global best both
                            # position (for velocity update) and
                            # fitness (the new group norm) are needed
                            self.best_swarm_fitness = new_fitness
                            self.best_swarm_pos = new_position


                        if t % 1 == 0:  # we print only two components even it search space is high-dimensional
                            print(
                                "Time: %6d,w:%9.4f,c1:%9.4f,c2:%9.4f,  Best Fitness: %14.6f,  Best Pos: %9.4f,%9.4f" % (
                                    t, self.w, self.c1, self.c2, self.best_swarm_fitness, self.best_swarm_pos[0],
                                    self.best_swarm_pos[1]), end=" ")
                            if self.d > 2:
                                print('...')
                            else:
                                print('')
                    self.position_values.append(self.best_swarm_pos[0])






#
# REFERENCES:
#################

# [1] ALMEIDA, BRUNO & COPPO LEITE, VICTOR. (2019). PARTICLE SWARM OPTIMIZATION: A POWERFUL TECHNIQUE FOR
#     SOLVING ENGINEERING PROBLEMS. 10.5772/INTECHOPEN.89633.
#
# [2] HE, YAN & MA, WEI & ZHANG, JI. (2016). THE PARAMETERS SELECTION OF PSO ALGORITHM INFLUENCING ON PERFORMANCE OF FAULT DIAGNOSIS.
#     MATEC WEB OF CONFERENCES. 63. 02019. 10.1051/MATECCONF/20166302019.
#
# [3] CLERC, M., AND J. KENNEDY. THE PARTICLE SWARM — EXPLOSION, STABILITY, AND CONVERGENCE IN A MULTIDIMENSIONAL COMPLEX SPACE.
#     IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION 6, NO. 1 (FEBRUARY 2002): 58–73.                                                                                                                                                                                           #
# [4] Y. H. SHI AND R. C. EBERHART, “A MODIFIED PARTICLE SWARM OPTIMIZER,” IN PROCEEDINGS OF THE IEEE INTERNATIONAL
#     CONFERENCES ON EVOLUTIONARY COMPUTATION, PP. 69–73, ANCHORAGE, ALASKA, USA, MAY 1998.
#
# [5] G. SERMPINIS, K. THEOFILATOS, A. KARATHANASOPOULOS, E. F. GEORGOPOULOS, & C. DUNIS, FORECASTING FOREIGN EXCHANGE
#     RATES WITH ADAPTIVE NEURAL NETWORKS USING RADIAL-BASIS FUNCTIONS AND PARTICLE SWARM OPTIMIZATION,
#     EUROPEAN JOURNAL OF OPERATIONAL RESEARCH.
#

if __name__ == '__main__':
    '''#Auto Hyper paramerter tuning
    N=30
    d=3
    Tmax=1000
    bound=5.12
    T=list(range(1,200))

    w_stat=[]
    c1_stat=[]
    c2_stat=[]


    for i in T:
        wstat,c1stat,c2stat=PSO(N=N,d=d,w=0,c1=0,c2=0,Tmax=Tmax,bound=bound,update_w=True, update_c1=True, update_c2=True, plot=False, verbose=False).run(tuning=True)
        w_stat.append(wstat)
        c1_stat.append(c1stat)
        c2_stat.append(c2stat)


    Parameter_tuning=pd.DataFrame({'Evalaution Turn':T,
                  'w':w_stat,
                  'c1':c1_stat,
                  'c2':c2_stat})

    Parameter_tuning.to_csv("/Users/ahmed/Documents/UOE/Courses/Semester 1/Natural COmputing /Project 40%/Particle_Swarm_Optimization-main/Optimization Data/sphere_Tuning.csv")'''

    Parameter_tuning=pd.read_csv("/Users/ahmed/Documents/UOE/Courses/Semester 1/Natural COmputing /Project 40%/Particle_Swarm_Optimization-main/Optimization Data/Tuning/Rastrigin_Tuning.csv")

    #PSO Perfomrance Evaluation
    N = 30
    d = 3
    w = Parameter_tuning[['w']].mean()[0]
    c1 = Parameter_tuning[['c1']].mean()[0]
    c2 = Parameter_tuning[['c2']].mean()[0]
    Tmax = 1000
    bound = 5.12
    T=list(range(1,200))
    success =[]
    fitness=[]
    position=[]
    iterations=[]
    

    for i in T:
        t, goal,pos,fit = PSO(N=N, d=d,c1=c1,c2=c2,w=w ,Tmax=Tmax, bound=bound, update_w=False, update_c1=False,update_c2=False, plot=False, verbose=False).run(tuning=False,plot=False)
   
        iterations.append(t)
        success.append(goal)
        fitness.append(fit)
        position.append(pos)

    Evalaution = pd.DataFrame({'Evalaution Turn': T,
                                'iterations': iterations,
                                'success': success,
                                'fitness':fitness,
                                'position':position
     })

    #Evalaution.to_csv("/Users/ahmed/Documents/UOE/Courses/Semester 1/Natural COmputing /Project 40%/Particle_Swarm_Optimization-main/Optimization Data/rastrigin_Evalaution_Auto.csv")'''


    # PSO Scaling Perfomrance Evaluation
    '''for i in [60,90]:
        d = 3
        N=i
        w=0
        c1=0
        c2=0
        Tmax = 1000
        bound = 5.12
        T = list(range(1, 200))
        for j in ["Tuning","Standard"]:
            success = []
            fitness = []
            position = []
            iterations = []
            if j=="Standard":
                w=0.7
                c1=2.05
                c2=2.05
            else:
                Parameter_tuning = pd.read_csv(
                    "/Users/ahmed/Documents/UOE/Courses/Semester 1/Natural COmputing /Project 40%/Particle_Swarm_Optimization-main/Optimization Data/Tuning/Rastrigin_Tuning.csv")
                w = Parameter_tuning[['w']].mean()[0]
                c1 = Parameter_tuning[['c1']].mean()[0]
                c2 = Parameter_tuning[['c2']].mean()[0]

            for k in T:
                t, goal, pos, fit = PSO(N=N, d=d, c1=c1, c2=c2, w=w, Tmax=Tmax, bound=bound, update_w=False,update_c1=False, update_c2=False, plot=False, verbose=False).run(tuning=False,plot=False)

                iterations.append(t)
                success.append(goal)
                fitness.append(fit)
                position.append(pos)

            Evalaution = pd.DataFrame({'Evalaution Turn': T,
                                       'iterations': iterations,
                                       'success': success,
                                       'fitness': fitness,
                                       'position': position
                                       })

            Evalaution.to_csv(
                "/Users/ahmed/Documents/UOE/Courses/Semester 1/Natural COmputing /Project 40%/Particle_Swarm_Optimization-main/Optimization Data/N="+str(i)+"/Rastrigin_"+str(j)+"_Evalaution_Auto.csv")'''

    '''#Inertia Analysis
    for i in [0.1,0.7,1.0]:
        d = 3
        N=30
        w=i
        c1=2.0
        c2=2.0
        Tmax = 1000
        bound = 5.12
        T = list(range(1, 200))
        plot=PSO(N=N, d=d, c1=c1, c2=c2, w=w, Tmax=Tmax, bound=bound, update_w=False,update_c1=False, update_c2=False, plot=False, verbose=False).run(tuning=False,plot=True)
        plot.savefig("/Users/ahmed/Documents/UOE/Courses/Semester 1/Natural COmputing /Project 40%/Particle_Swarm_Optimization-main/Optimization Data/Plots/w"+str(i)+"_rastrigin.jpg")
    #C1,C2 analysis
    for i,j in zip([4,2,0],[0,2,4]):
        d = 3
        N=30
        w=0.1
        c1=i
        c2=j
        Tmax = 1000
        bound = 5.12
        T = list(range(1, 200))
        plot=PSO(N=N, d=d, c1=c1, c2=c2, w=w, Tmax=Tmax, bound=bound, update_w=False,update_c1=False, update_c2=False, plot=False, verbose=False).run(tuning=False,plot=True)
        plot.savefig("/Users/ahmed/Documents/UOE/Courses/Semester 1/Natural COmputing /Project 40%/Particle_Swarm_Optimization-main/Optimization Data/Plots/c1"+str(i)+"c2"+str(j)+"_rastrigin.jpg")'''

