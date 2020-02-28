import mlrose as ml
import numpy as np
import matplotlib.pyplot as plt
import time

weights = [10, 5, 2, 8, 15,10,5,5, 10,3,10,12,15,8,4,15,7,10,14,2,5,20,4,23,12,5,4,8,9,7,10,5,5,8,5,7,2,4,14,5,6,15,12,11,3,13,15,17,8,10]
values = [1, 2, 3,5,4,6,9,12,1,5,6,14,6,8,1,4,5,7,7,8,3,7,3,7,6,9,12,4,6,3,3,3,3,8,6,9,3,2,3,6,8,3,5,7,5,6,9,1,12,6]
# weights = [10, 5, 2, 8, 15,10,5,5, 10,3,10,12,15,8,4,15,7,10,14,2,5,20,4,23,12,5,13,14,6,9,3,20,2,6,4,15,11,13,12,14,13,15,16,5,14,19,5,9,6,4,5,6,3,9,2,7,4,6,74,1,2,4,6,3,6,4,7,4,10,11,5,3,19,11,5,4,6,3,2,13,2,5,5,8,5,7,2,4,14,5,6,15,12,11,3,13,15,17,8,10]
# values = [1, 2, 3, 5,4,6,9,12,1,5,6,14,6,8,1,4,5,7,7,8,3,7,3,7,6,9,12,4,6,3,3,3,3,8,6,9,3,2,3,6,8,3,5,7,1,12,6,5,6,3,4,10,12,4,5,4,7,6,12,2,4,5,12,5,3,5,6,7,2,19,11,5,3,1,6,9,8,11,10,2,12,14,11,13,3,4,5,8,9,3,4,4,4,1,2,15,16,11,4,4]

fitness = ml.Knapsack(weights, values, 0.5)
problem = ml.DiscreteOpt(length=50, fitness_fn=fitness, maximize=True, max_val=2)

best_state, best_fitness, fitness_curve = ml.random_hill_climb(problem, max_attempts = 100, max_iters=5000,restarts=40, curve=True)
print(best_fitness)
plt.plot(fitness_curve)
plt.show()

# init_state = np.random.randint(2, size=50)
schedule = ml.ExpDecay()
# ##rhc
best=[]
for r in range(0,20,2):
    best_state, best_fitness, fitness_curve = ml.random_hill_climb(problem, max_attempts = 100, max_iters=5000,restarts=r, curve=True)
    # plt.plot(fitness_curve)
    best.append(best_fitness)
best=np.array(best)
plt.figure()
plt.plot(np.arange(0,20,2),best)
plt.xlabel('number of random restarts')
plt.ylabel('Best Fitness Value')
plt.title('best fitness value vs no. of restarts (knapsack)')
plt.savefig('knapsack_rhc.jpg')
###sa


bs1, best_f1, fitness_curve1 = ml.simulated_annealing(problem, schedule=ml.ExpDecay(), max_attempts=100,
                                                                    max_iters=3000,  curve=True)
bs2, best_f2, fitness_curve2 = ml.simulated_annealing(problem, schedule=ml.GeomDecay(), max_attempts=100,
                                                                    max_iters=3000, curve=True)
bs3, best_f3, fitness_curve3 = ml.simulated_annealing(problem, schedule=ml.ArithDecay(), max_attempts=100,
                                                                    max_iters=3000, curve=True)
plt.figure()
plt.plot(fitness_curve1,label='exp')
plt.plot(fitness_curve2,label='geom')
plt.plot(fitness_curve3,label='arith')
plt.legend()
plt.xlabel('number of iterations')
plt.ylabel('Fitness Value')
plt.title('fitness curves for sa_diff_types of decay schedules(knapsack)')
plt.savefig('knapsack_sa.jpg')



##ga
best=[]
for p in range(100,500,50):

    best_state, best_fitness, fitness_curve = ml.genetic_alg(problem,pop_size=p, max_attempts=20, max_iters=2000, curve=True)
    # plt.plot(fitness_curve)
    best.append(best_fitness)
best=np.array(best)
plt.figure()
plt.plot(np.arange(100,500,50),best)
plt.xlabel('population size')
plt.ylabel('Best Fitness Value')
plt.title('best fitness value vs pop_size_ga (knapsack)')
plt.savefig('knapsack_ga.jpg')

m=np.arange(0.05,.55,.05)
best=[]
for i in range(0,10,1):
    best_state, best_fitness, fitness_curve = ml.genetic_alg(problem,pop_size=200,mutation_prob=m[i], max_attempts=20, max_iters=2000, curve=True)
    # plt.plot(fitness_curve)
    best.append(best_fitness)
best=np.array(best)
plt.figure()
plt.plot((np.arange(0.05,.55,.05)),best)
plt.xlabel('mutation prob')
plt.ylabel('Best Fitness Value')
plt.title('best fitness value vs mutation_prob_ga (knapsack)')
plt.savefig('knapsack_ga1.jpg')

##mimic
best=[]
for p in range(50,300,50):

    best_state, best_fitness, fitness_curve =ml.mimic(problem, pop_size=p,keep_pct=0.2, max_attempts=10, max_iters=200,curve=True)
    # plt.plot(fitness_curve)
    best.append(best_fitness)
best=np.array(best)
plt.figure()
plt.plot(np.arange(50,300,50),best)
plt.xlabel('population size')
plt.ylabel('Best Fitness Value')
plt.title('best fitness value vs pop_size_MIMIC (knapsack)')
plt.savefig('knapsack_mimic.jpg')

m=np.arange(0.05,.55,.05)
best=[]
for i in range(0,10,1):

    best_state, best_fitness, fitness_curve = ml.mimic(problem, pop_size=200,keep_pct=m[i], max_attempts=10, max_iters=200,curve=True)
    # plt.plot(fitness_curve)
    best.append(best_fitness)
best=np.array(best)
plt.figure()
plt.plot(np.arange(0.05,.55,.05),best)
plt.xlabel('Proportion of samples to keep at each iteration')
plt.ylabel('Best Fitness Value')
plt.title('best fitness value vs Proportion of samples_mimic (knapsack)')
plt.savefig('knapsack_mimic1.jpg')


###after tuning

problem = ml.DiscreteOpt(length = 50, fitness_fn = fitness, maximize = True, max_val = 2)
schedule = ml.ExpDecay()
s=time.time()
best_state, best_fitness, fitness_curve = ml.random_hill_climb(problem, max_attempts=100, max_iters=2500,restarts=8,
                                                                curve=True)
e=time.time()
t=e-s

s1=time.time()
best_state1, best_fitness1, fitness_curve1 = ml.simulated_annealing(problem, schedule=schedule, max_attempts=100,
                                                                    max_iters=3000,  curve=True)
e1=time.time()
t1=e1-s1

s2=time.time()
best_state2, best_fitness2, fitness_curve2 = ml.genetic_alg(problem,pop_size=250,mutation_prob=.1, max_attempts=20, max_iters=3000, curve=True)
e2=time.time()
t2=e2-s2

s3=time.time()
best_state3, best_fitness3, fitness_curve3 = ml.mimic(problem, pop_size=200,keep_pct=0.2, max_attempts=10, max_iters=200,
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
plt.title('Fitness curves for different algorithms on knapsack')
plt.savefig('knapsack.jpg')

t_0=[]
t_1=[]
t_2=[]
t_3=[]
bs_0=[]
bs_1=[]
bs_2=[]
bs_3=[]
e_0=[]
e_1=[]
e_2=[]
e_3=[]

for n in range(5,100,16):
    fitness = ml.Knapsack(weights[:n], values[:n], 0.6)
    problem = ml.DiscreteOpt(length=n, fitness_fn=fitness, maximize=True, max_val=2)

    schedule = ml.ExpDecay()

    s = time.time()
    best_state, best_fitness, fitness_curve = ml.random_hill_climb(problem, max_attempts=5, max_iters=500,
                                                                   restarts=8,
                                                                    curve=True)
    e = time.time()
    t = e - s
    e_0.append(len(fitness_curve))
    t_0.append(t)
    bs_0.append(best_fitness)


    s1 = time.time()
    best_state1, best_fitness1, fitness_curve1 = ml.simulated_annealing(problem, schedule=schedule, max_attempts=4,
                                                                        max_iters=1000,
                                                                        curve=True)
    e1 = time.time()
    t1 = e1 - s1
    e_1.append(len(fitness_curve1))
    t_1.append(t1)
    bs_1.append(best_fitness1)


    s2 = time.time()
    best_state2, best_fitness2, fitness_curve2 = ml.genetic_alg(problem, pop_size=160, mutation_prob=.4,
                                                                max_attempts=5, max_iters=1000, curve=True)
    e2 = time.time()
    t2 = e2 - s2
    e_2.append(len(fitness_curve2))
    t_2.append(t2)
    bs_2.append(best_fitness2)


    s3 = time.time()
    best_state3, best_fitness3, fitness_curve3 = ml.mimic(problem, pop_size=250, keep_pct=0.2, max_attempts=10,
                                                          max_iters=100,
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
plt.plot(np.arange(5,100,16),bs_0, label='rhc')
plt.plot(np.arange(5,100,16),bs_1, label='sa')
plt.plot(np.arange(5,100,16),bs_2, label='ga')
plt.plot(np.arange(5,100,16),bs_3, label='mimic')
plt.legend()
plt.xlabel('size of problem')
plt.ylabel('Best Fitness Value')
plt.title('best fitness value vs size of problem(knapsack')
plt.savefig('knapsack_fitness.jpg')


plt.figure()
plt.plot(np.arange(5,100,16),t_0, label='rhc')
plt.plot(np.arange(5,100,16),t_1, label='sa')
plt.plot(np.arange(5,100,16),t_2, label='ga')
plt.plot(np.arange(5,100,16),t_3, label='mimic')
plt.legend()
plt.xlabel('size of problem')
plt.ylabel('Time taken')
plt.title('time taken vs size of problem(knapsack)')
plt.savefig('knapsack_time.jpg')

plt.figure()
plt.plot(np.arange(5,100,16),e_0, label='rhc')
plt.plot(np.arange(5,100,16),e_1, label='sa')
plt.plot(np.arange(5,100,16),e_2, label='ga')
plt.plot(np.arange(5,100,16),e_3, label='mimic')
plt.legend()
plt.xlabel('size of problem')
plt.ylabel('# of iterations.')
plt.title('# of iterations (knapsack)')
plt.savefig('knapsack_eval.jpg')




