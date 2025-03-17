import numpy as np
import simpy
import matplotlib.pyplot as plt


def simulation(_lambda, mu, simulation_time):
    class Operator:
        def __init__(self, env, _lambda, mu):
            self.env = env
            self.server = simpy.Resource(env, capacity=1)
            self._lambda = _lambda
            self.mu = mu

            self.total_requests = 0
            self.served_requests = 0
            self.lost_requests = 0
            self.busy_time = 0
            self.last_event_time = 0

            env.process(self.generate_requests())

        def generate_requests(self):
            while True:
                # Здесь мы генерируем запросы таким образом, чтобы интенсивность потока в среднем была 1/lambda
                yield self.env.timeout(np.random.exponential(1 / self._lambda))
                self.total_requests += 1

                if self.server.count == 0:
                    self.env.process(self.process_request())
                else:
                    self.lost_requests += 1

        def process_request(self):
            with self.server.request() as request:
                yield request
                arrival_time = self.env.now

                yield self.env.timeout(np.random.exponential(1 / self.mu))

                self.served_requests += 1
                self.busy_time += self.env.now - arrival_time

    env = simpy.Environment()
    system = Operator(env, lambda_, mu)
    env.run(until=simulation_time)

    return {
        "Всего запросов": system.total_requests,
        "Обслужено запросов": system.served_requests,
        "Потеряно запросов": system.lost_requests,
        "Вероятность отказа": system.lost_requests / system.total_requests,
        "Коэффициент загрузки системы": system.busy_time / simulation_time,
    }


def erlang_b(lambda_, mu):
    # n = 1 (так как канал 1)
    rho = lambda_ / mu
    return rho / (1 + rho)


lambda_values = np.linspace(4, 15, 12)
mu = 6
simulation_time = 1000

simulation_results = []
theoretical_results = []

table_data = []

for lambda_ in lambda_values:
    result = simulation(lambda_, mu, simulation_time)
    
    table_data.append((lambda_, mu, simulation_time, result))
    
    simulation_results.append(result["Вероятность отказа"])
    theoretical_results.append(erlang_b(lambda_, mu))

plt.plot(lambda_values, simulation_results, label='Simulated')
plt.plot(lambda_values, theoretical_results, label='Theoretical', linestyle='dashed')
plt.xlabel('Интенсивность входящего потока (λ)')
plt.ylabel('Вероятность отказа')
plt.legend()
plt.title('Вероятность отказа vs Скорость входящего потока')
plt.grid()
plt.show()


for index, value in enumerate(table_data):
    print(f"Experiment {index}: lambda: {value[0]}, mu: {value[1]}, time: {value[2]}, result: {value[3]}")
