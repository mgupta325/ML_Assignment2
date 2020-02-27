import mlrose as ml
import numpy as np
import matplotlib.pyplot as plt
import time


fitness = ml.FourPeaks(t_pct=0.15)
problem = ml.DiscreteOpt(length=40, fitness_fn=fitness, maximize=True, max_val=2)
init_state = np.random.randint(2, size=40)

# ##rhc
best=[]
for r in range(1,19,2):

    best_state, best_fitness, fitness_curve = ml.random_hill_climb(problem, max_attempts = 100, max_iters=2000,restarts=r, init_state = init_state, curve=True)
    # plt.plot(fitness_curve)
    best.append(best_fitness)
best=np.array(best)
plt.figure()
plt.plot(np.arange(1,19,2),best)
plt.xlabel('number of random restarts')
plt.ylabel('Best Fitness Value')
plt.title('best fitness value vs no. of restarts (4 Peaks)')
plt.savefig('4_peaks_rhc.jpg')
# ###sa
#

bs1, best_f1, fitness_curve1 = ml.simulated_annealing(problem, schedule=ml.ExpDecay(), max_attempts=100,
                                                                    max_iters=3000, init_state=init_state, curve=True)
bs2, best_f2, fitness_curve2 = ml.simulated_annealing(problem, schedule=ml.GeomDecay(), max_attempts=100,
                                                                    max_iters=3000, init_state=init_state, curve=True)
bs3, best_f3, fitness_curve3 = ml.simulated_annealing(problem, schedule=ml.ArithDecay(), max_attempts=100,
                                                                    max_iters=3000, init_state=init_state, curve=True)
plt.figure()
plt.plot(fitness_curve1,label='exp')
plt.plot(fitness_curve2,label='geom')
plt.plot(fitness_curve3,label='arith')
plt.legend()
plt.xlabel('number of iterations')
plt.ylabel('Fitness Value')
plt.title('fitness curves for sa_diff_types of decay schedules(4 Peaks)')
plt.savefig('4_peaks_sa.jpg')

# ##ga
best=[]
for p in range(150,1000,100):

    best_state, best_fitness, fitness_curve = ml.genetic_alg(problem,pop_size=p, max_attempts=100, max_iters=9000, curve=True)
    # plt.plot(fitness_curve)
    best.append(best_fitness)
best=np.array(best)
plt.figure()
plt.plot(np.arange(150,1000,100),best)
plt.xlabel('population size')
plt.ylabel('Best Fitness Value')
plt.title('best fitness value vs pop_size_ga (4 Peaks)')
plt.savefig('4_peaks_ga.jpg')

m=np.arange(0.05,.55,.05)
best=[]
for i in range(0,10,1):


    best_state, best_fitness, fitness_curve = ml.genetic_alg(problem,pop_size=700,mutation_prob=m[i], max_attempts=100, max_iters=9000, curve=True)
    # plt.plot(fitness_curve)
    best.append(best_fitness)
best=np.array(best)
plt.figure()
plt.plot((np.arange(0.05,.55,.05)),best)
plt.xlabel('mutation prob')
plt.ylabel('Best Fitness Value')
plt.title('best fitness value vs mutation_prob_ga (4 Peaks)')
plt.savefig('4_peaks_ga1.jpg')

# ##mimic
best=[]
for p in range(50,800,100):

    best_state, best_fitness, fitness_curve =ml.mimic(problem, pop_size=p,keep_pct=0.2, max_attempts=10, max_iters=200,curve=True)
    # plt.plot(fitness_curve)
    best.append(best_fitness)
best=np.array(best)
plt.figure()
plt.plot(np.arange(50,800,100),best)
plt.xlabel('population size')
plt.ylabel('Best Fitness Value')
plt.title('best fitness value vs pop_size_MIMIC (4 Peaks)')
plt.savefig('4_peaks_mimic.jpg')

m=np.arange(0.05,.55,.05)
best=[]
for i in range(0,10,1):

    best_state, best_fitness, fitness_curve = ml.mimic(problem, pop_size=500,keep_pct=m[i], max_attempts=15, max_iters=200,curve=True)
    # plt.plot(fitness_curve)
    best.append(best_fitness)
best=np.array(best)
plt.figure()
plt.plot(np.arange(0.05,.55,.05),best)
plt.xlabel('Proportion of samples to keep at each iteration')
plt.ylabel('Best Fitness Value')
plt.title('best fitness value vs Proportion of samples_mimic (4 Peaks)')
plt.savefig('4_peaks_mimic1.jpg')


# ####after tuning

schedule = ml.ExpDecay()
s=time.time()
best_state, best_fitness, fitness_curve = ml.random_hill_climb(problem, max_attempts=100, max_iters=2500,restarts=8,
                                                               init_state=init_state, curve=True)
e=time.time()
t=e-s

s1=time.time()
best_state1, best_fitness1, fitness_curve1 = ml.simulated_annealing(problem, schedule=schedule, max_attempts=100,
                                                                    max_iters=4000, init_state=init_state, curve=True)
e1=time.time()
t1=e1-s1

s2=time.time()
best_state2, best_fitness2, fitness_curve2 = ml.genetic_alg(problem,pop_size=750,mutation_prob=.1, max_attempts=100, max_iters=9000, curve=True)
e2=time.time()
t2=e2-s2

s3=time.time()
best_state3, best_fitness3, fitness_curve3 = ml.mimic(problem, pop_size=400,keep_pct=0.2, max_attempts=10, max_iters=200,
                                                      curve=True)
e3=time.time()
t3=e3-s3
print("time taken by rhc,sa,ga,mimic for optimization: ",t,t1,t2,t3)
plt.figure()
plt.plot(fitness_curve,label='rhc')
plt.plot(fitness_curve1,label='sa')
plt.plot(fitness_curve2,label='ga')
plt.plot(fitness_curve3,label='mimic')
plt.legend()
plt.xlabel('number of iterations')
plt.ylabel(' Fitness Value')
plt.title('Fitness curves for different algorithms on 4peaks')
plt.show()
plt.savefig('4_peaks.jpg')
t_0=[]
t_1=[]
t_2=[]
t_3=[]
e_0=[]
e_1=[]
e_2=[]
e_3=[]
bs_0=[]
bs_1=[]
bs_2=[]
bs_3=[]


for n in range(5,120,20):

    
    problem = ml.DiscreteOpt(length=n, fitness_fn=fitness, maximize=True, max_val=2)

    schedule = ml.ExpDecay()
    s = time.time()
    best_state, best_fitness, fitness_curve = ml.random_hill_climb(problem, max_attempts=100, max_iters=10000,
                                                                   restarts=8,
                                                                   init_state=None, curve=True)
    e = time.time()
    t = e - s
    e_0.append(len(fitness_curve))
    t_0.append(t)
    bs_0.append(best_fitness)

    s1 = time.time()
    best_state1, best_fitness1, fitness_curve1 = ml.simulated_annealing(problem, schedule=schedule, max_attempts=100,
                                                                        max_iters=10000, init_state=None,
                                                                        curve=True)
    e1 = time.time()
    t1 = e1 - s1
    e_1.append(len(fitness_curve1))
    t_1.append(t1)
    bs_1.append(best_fitness1)
   
    s2 = time.time()
    best_state2, best_fitness2, fitness_curve2 = ml.genetic_alg(problem, pop_size=400, mutation_prob=.1,
                                                                max_attempts=100, max_iters=9000, curve=True)
    e2 = time.time()
    t2 = e2 - s2
    e_2.append(len(fitness_curve2))
    t_2.append(t2)
    bs_2.append(best_fitness2)
    
    s3 = time.time()
    best_state3, best_fitness3, fitness_curve3 = ml.mimic(problem, pop_size=300, keep_pct=0.2, max_attempts=10,
                                                          max_iters=200,
                                                          curve=True)
    e3 = time.time()
    

    t3 = e3 - s3
    e_3.append(len(fitness_curve3))
    t_3.append(t3)
    bs_3.append(best_fitness3)
    print("time taken by rhc,sa,ga,mimic for optimization: ", t, t1, t2, t3)

t_0=np.array(t_0)
t_1=np.array(t_1)
t_2=np.array(t_2)
t_3=np.array(t_3)

e_0=np.array(e_0)
e_1=np.array(e_1)
e_2=np.array(e_2)
e_3=np.array(e_3)

bs_0=np.array(bs_0)
bs_1=np.array(bs_1)
bs_2=np.array(bs_2)
bs_3=np.array(bs_3)

plt.figure()
plt.plot(np.arange(5,120,20),bs_0, label='rhc')
plt.plot(np.arange(5,120,20),bs_1, label='sa')
plt.plot(np.arange(5,120,20),bs_2, label='ga')
plt.plot(np.arange(5,120,20),bs_3, label='mimic')
plt.legend()
plt.xlabel('size of problem')
plt.ylabel('Best Fitness Value')
plt.title('best fitness value vs size of problem')
plt.savefig('4_peaks_fitness.jpg')

plt.figure()
plt.plot(np.arange(5,120,20),t_0, label='rhc')
plt.plot(np.arange(5,120,20),t_1, label='sa')
plt.plot(np.arange(5,120,20),t_2, label='ga')
plt.plot(np.arange(5,120,20),t_3, label='mimic')
plt.legend()
plt.xlabel('size of problem')
plt.ylabel('Time taken')
plt.title('time taken vs size of problem')
plt.savefig('4_peaks_time.jpg')

plt.figure()
plt.plot(np.arange(5,120,20),e_0, label='rhc')
plt.plot(np.arange(5,120,20),e_1, label='sa')
plt.plot(np.arange(5,120,20),e_2, label='ga')
plt.plot(np.arange(5,120,20),e_3, label='mimic')
plt.legend()
plt.xlabel('size of problem')
plt.ylabel('# of iterations')
plt.title('# of iterations vs problem size (4Peaks)')
plt.savefig('4_peaks_iter.jpg')


