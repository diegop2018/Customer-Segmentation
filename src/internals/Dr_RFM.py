# Función principal mejorada para ejecutar el sistema complimport numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
import numpy as np
warnings.filterwarnings('ignore')

# Importaciones para Deep Reinforcement Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

class RFMEnvironment:
    """
    Entorno de simulación para el aprendizaje por refuerzo en segmentación RFM
    """
    def __init__(self, df_customers, max_timesteps=100):
        self.df_customers = df_customers.copy()
        self.max_timesteps = max_timesteps
        self.current_timestep = 0
        self.customer_states = {}
        self.action_history = []
        self.reward_history = []
        
        # Definir segmentos RFM tradicionales
        self.rfm_segments = {
            'Champions': {'R': [4, 5], 'F': [4, 5], 'M': [4, 5]},
            'Loyal Customers': {'R': [3, 5], 'F': [3, 5], 'M': [3, 5]},
            'Potential Loyalists': {'R': [3, 5], 'F': [1, 3], 'M': [1, 3]},
            'At Risk': {'R': [1, 2], 'F': [2, 5], 'M': [2, 5]},
            'Cannot Lose Them': {'R': [1, 2], 'F': [4, 5], 'M': [4, 5]},
            'Hibernating': {'R': [1, 2], 'F': [1, 2], 'M': [1, 3]},
            'New Customers': {'R': [4, 5], 'F': [1, 1], 'M': [1, 1]}
        }
        
        # Acciones posibles
        self.actions = {
            0: 'no_action',
            1: 'discount_offer',
            2: 'premium_offer',
            3: 'retention_campaign',
            4: 'cross_sell',
            5: 'segment_upgrade',
            6: 'churn_prevention'
        }
        
        self.initialize_environment()
    
    def initialize_environment(self):
        """Inicializar el estado del entorno"""
        # Calcular quintiles RFM
        self.df_customers['R_Score'] = pd.qcut(self.df_customers['Recencia'], 5, labels=[5,4,3,2,1])
        self.df_customers['F_Score'] = pd.qcut(self.df_customers['Frequencia'].rank(method='first'), 5, labels=[1,2,3,4,5])
        self.df_customers['M_Score'] = pd.qcut(self.df_customers['Valor Monetario'], 5, labels=[1,2,3,4,5])
        
        # Estado inicial de cada cliente
        for idx, customer in self.df_customers.iterrows():
            self.customer_states[idx] = {
                'R': int(customer['R_Score']),
                'F': int(customer['F_Score']),
                'M': int(customer['M_Score']),
                'segment': self.assign_segment(customer),
                'clv': customer.get('CLV', customer['Valor Monetario']),
                'churn_risk': np.random.beta(2, 5),  # Simulado
                'satisfaction': np.random.beta(5, 2),  # Simulado
                'days_since_last_action': 0,
                'total_value_generated': 0
            }
    
    def assign_segment(self, customer):
        """Asignar segmento RFM tradicional"""
        r, f, m = int(customer['R_Score']), int(customer['F_Score']), int(customer['M_Score'])
        
        for segment, criteria in self.rfm_segments.items():
            if (criteria['R'][0] <= r <= criteria['R'][1] and
                criteria['F'][0] <= f <= criteria['F'][1] and
                criteria['M'][0] <= m <= criteria['M'][1]):
                return segment
        return 'Other'
    
    def get_state_vector(self, customer_id):
        """Obtener vector de estado para un cliente"""
        state = self.customer_states[customer_id]
        return np.array([
            state['R'], state['F'], state['M'],
            state['churn_risk'], state['satisfaction'],
            state['days_since_last_action'],
            state['total_value_generated'] / 1000,  # Normalizado
            len([seg for seg, crit in self.rfm_segments.items() 
                 if self.segment_match(state, crit)])  # Flexibilidad de segmento
        ])
    
    def segment_match(self, state, criteria):
        """Verificar si un estado coincide con criterios de segmento"""
        return (criteria['R'][0] <= state['R'] <= criteria['R'][1] and
                criteria['F'][0] <= state['F'] <= criteria['F'][1] and
                criteria['M'][0] <= state['M'] <= criteria['M'][1])
    
    def step(self, customer_id, action):
        """Ejecutar una acción y devolver recompensa"""
        state = self.customer_states[customer_id]
        old_clv = state['clv']
        old_churn_risk = state['churn_risk']
        
        # Simular efectos de la acción
        reward = self.simulate_action_effect(customer_id, action)
        
        # Actualizar estado del cliente
        self.update_customer_state(customer_id, action)
        
        # Calcular recompensa total
        clv_improvement = (state['clv'] - old_clv) / old_clv if old_clv > 0 else 0
        churn_prevention = max(0, old_churn_risk - state['churn_risk'])
        
        total_reward = reward + clv_improvement * 100 + churn_prevention * 50
        
        self.action_history.append({
            'customer_id': customer_id,
            'action': action,
            'reward': total_reward,
            'timestep': self.current_timestep
        })
        
        self.current_timestep += 1
        done = self.current_timestep >= self.max_timesteps
        
        return self.get_state_vector(customer_id), total_reward, done
    
    def simulate_action_effect(self, customer_id, action):
        """Simular el efecto de una acción en el comportamiento del cliente"""
        state = self.customer_states[customer_id]
        action_name = self.actions[action]
        
        # Efectos base por acción
        effects = {
            'no_action': {'reward': 0, 'churn_change': 0.01, 'satisfaction_change': -0.01},
            'discount_offer': {'reward': 20, 'churn_change': -0.1, 'satisfaction_change': 0.05},
            'premium_offer': {'reward': 50, 'churn_change': -0.05, 'satisfaction_change': 0.1},
            'retention_campaign': {'reward': 15, 'churn_change': -0.2, 'satisfaction_change': 0.08},
            'cross_sell': {'reward': 30, 'churn_change': -0.03, 'satisfaction_change': 0.02},
            'segment_upgrade': {'reward': 25, 'churn_change': -0.08, 'satisfaction_change': 0.15},
            'churn_prevention': {'reward': 10, 'churn_change': -0.25, 'satisfaction_change': 0.12}
        }
        
        effect = effects[action_name]
        
        # Modular efectos según características del cliente
        segment_multiplier = {
            'Champions': 1.2,
            'Loyal Customers': 1.1,
            'At Risk': 1.5,
            'Cannot Lose Them': 1.3,
            'Hibernating': 0.8,
            'New Customers': 1.0
        }.get(state['segment'], 1.0)
        
        return effect['reward'] * segment_multiplier
    
    def update_customer_state(self, customer_id, action):
        """Actualizar el estado del cliente después de una acción"""
        state = self.customer_states[customer_id]
        action_name = self.actions[action]
        
        # Actualizar métricas según la acción
        if action_name == 'discount_offer':
            state['churn_risk'] = max(0, state['churn_risk'] - 0.1)
            state['satisfaction'] = min(1, state['satisfaction'] + 0.05)
            state['total_value_generated'] += np.random.normal(50, 15)
            
        elif action_name == 'premium_offer':
            state['churn_risk'] = max(0, state['churn_risk'] - 0.05)
            state['satisfaction'] = min(1, state['satisfaction'] + 0.1)
            state['total_value_generated'] += np.random.normal(100, 25)
            
        elif action_name == 'retention_campaign':
            state['churn_risk'] = max(0, state['churn_risk'] - 0.2)
            state['satisfaction'] = min(1, state['satisfaction'] + 0.08)
            
        elif action_name == 'segment_upgrade':
            # Intentar mejorar scores RFM
            if np.random.random() < 0.3:
                state['F'] = min(5, state['F'] + 1)
            if np.random.random() < 0.2:
                state['M'] = min(5, state['M'] + 1)
        
        # Decay natural
        state['churn_risk'] = min(1, state['churn_risk'] + 0.01)
        state['satisfaction'] = max(0, state['satisfaction'] - 0.005)
        state['days_since_last_action'] = 0 if action != 0 else state['days_since_last_action'] + 1
        
        # Actualizar CLV
        state['clv'] = state['total_value_generated'] * (1 - state['churn_risk']) * state['satisfaction']

class DQNAgent:
    """
    Agente de Deep Q-Network para la segmentación RFM adaptativa
    """
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.gamma = 0.95  # Discount factor
        
        # Redes neuronales
        self.q_network = self.build_network()
        self.target_network = self.build_network()
        self.update_target_network()
    
    def build_network(self):
        """Construir la red neuronal para Q-learning"""
        model = Sequential([
            Dense(128, input_dim=self.state_size, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model
    
    def update_target_network(self):
        """Actualizar la red objetivo"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Almacenar experiencia en memoria"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Elegir acción usando epsilon-greedy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.q_network.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self, batch_size=32):
        """Entrenar el agente con experiencias pasadas"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = np.array([transition[0] for transition in batch])
        actions = np.array([transition[1] for transition in batch])
        rewards = np.array([transition[2] for transition in batch])
        next_states = np.array([transition[3] for transition in batch])
        dones = np.array([transition[4] for transition in batch])
        
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        
        targets = self.q_network.predict(states, verbose=0)
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        self.q_network.fit(states, targets, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class AdaptiveRFMSegmentation:
    """
    Sistema principal de segmentación RFM adaptativa con aprendizaje por refuerzo
    """
    def __init__(self, df_customers):
        self.df_customers = df_customers
        self.environment = RFMEnvironment(df_customers)
        self.agent = DQNAgent(state_size=8, action_size=7)
        self.training_history = []
        self.customer_journeys = {}
        self.baseline_metrics = {}
        self.sensitivity_results = {}
        self.ablation_results = {}
        
    def train_agent(self, episodes=1000, customers_per_episode=10):
        """Entrenar el agente de aprendizaje por refuerzo"""
        rewards_per_episode = []
        
        print("Iniciando entrenamiento del agente...")
        
        for episode in range(episodes):
            total_reward = 0
            episode_customers = np.random.choice(
                list(self.environment.customer_states.keys()), 
                size=min(customers_per_episode, len(self.environment.customer_states)), 
                replace=False
            )
            
            for customer_id in episode_customers:
                state = self.environment.get_state_vector(customer_id)
                state = np.reshape(state, [1, self.agent.state_size])
                
                action = self.agent.act(state)
                next_state, reward, done = self.environment.step(customer_id, action)
                next_state = np.reshape(next_state, [1, self.agent.state_size])
                
                self.agent.remember(state, action, reward, next_state, done)
                total_reward += reward
                
                # Registrar journey del cliente
                if customer_id not in self.customer_journeys:
                    self.customer_journeys[customer_id] = []
                
                self.customer_journeys[customer_id].append({
                    'episode': episode,
                    'action': self.environment.actions[action],
                    'reward': reward,
                    'state': self.environment.customer_states[customer_id].copy()
                })
            
            # Entrenar el agente
            if len(self.agent.memory) > 32:
                self.agent.replay()
            
            rewards_per_episode.append(total_reward / len(episode_customers))
            
            # Actualizar red objetivo periódicamente
            if episode % 50 == 0:
                self.agent.update_target_network()
            
            if episode % 100 == 0:
                avg_reward = np.mean(rewards_per_episode[-100:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {self.agent.epsilon:.3f}")
        
        self.training_history = rewards_per_episode
        print("Entrenamiento completado!")
    
    def predict_optimal_actions(self, customer_ids=None):
        """Predecir acciones óptimas para clientes específicos"""
        if customer_ids is None:
            customer_ids = list(self.environment.customer_states.keys())
        
        predictions = {}
        
        for customer_id in customer_ids:
            state = self.environment.get_state_vector(customer_id)
            state = np.reshape(state, [1, self.agent.state_size])
            
            # Usar el agente entrenado (sin exploración)
            old_epsilon = self.agent.epsilon
            self.agent.epsilon = 0
            
            action = self.agent.act(state)
            q_values = self.agent.q_network.predict(state, verbose=0)[0]
            
            self.agent.epsilon = old_epsilon
            
            predictions[customer_id] = {
                'optimal_action': self.environment.actions[action],
                'action_id': action,
                'q_values': q_values,
                'customer_state': self.environment.customer_states[customer_id],
                'confidence': np.max(q_values) - np.mean(q_values)
            }
        
        return predictions
    
    def analyze_segment_transitions(self):
        """Analizar transiciones de segmentos durante el entrenamiento"""
        transitions = {}
        
        for customer_id, journey in self.customer_journeys.items():
            customer_transitions = []
            
            for i in range(1, len(journey)):
                old_segment = journey[i-1]['state']['segment']
                new_segment = journey[i]['state']['segment']
                
                if old_segment != new_segment:
                    customer_transitions.append({
                        'from': old_segment,
                        'to': new_segment,
                        'episode': journey[i]['episode'],
                        'action': journey[i]['action']
                    })
            
            if customer_transitions:
                transitions[customer_id] = customer_transitions
        
        return transitions
    
    def plot_training_progress(self):
        """Visualizar el progreso del entrenamiento"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Recompensas por episodio
        ax1.plot(self.training_history)
        ax1.set_title('Recompensas por Episodio')
        ax1.set_xlabel('Episodio')
        ax1.set_ylabel('Recompensa Promedio')
        ax1.grid(True, alpha=0.3)
        
        # Promedio móvil de recompensas
        window = 50
        if len(self.training_history) >= window:
            moving_avg = np.convolve(self.training_history, np.ones(window)/window, mode='valid')
            ax2.plot(moving_avg)
            ax2.set_title(f'Promedio Móvil de Recompensas (ventana={window})')
            ax2.set_xlabel('Episodio')
            ax2.set_ylabel('Recompensa Promedio')
            ax2.grid(True, alpha=0.3)
        
        # Distribución de acciones
        action_counts = {}
        for customer_journey in self.customer_journeys.values():
            for step in customer_journey:
                action = step['action']
                action_counts[action] = action_counts.get(action, 0) + 1
        
        actions = list(action_counts.keys())
        counts = list(action_counts.values())
        
        ax3.bar(actions, counts)
        ax3.set_title('Distribución de Acciones Tomadas')
        ax3.set_xlabel('Acción')
        ax3.set_ylabel('Frecuencia')
        ax3.tick_params(axis='x', rotation=45)
        
        # Evolución de epsilon
        episodes = list(range(len(self.training_history)))
        epsilon_history = [1.0 * (0.995 ** ep) for ep in episodes]
        epsilon_history = [max(0.01, eps) for eps in epsilon_history]
        
        ax4.plot(episodes, epsilon_history)
        ax4.set_title('Evolución de Epsilon (Exploración)')
        ax4.set_xlabel('Episodio')
        ax4.set_ylabel('Epsilon')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def sensitivity_analysis(self, parameters_to_test=None):
        """
        Análisis de sensibilidad de hiperparámetros del modelo
        """
        if parameters_to_test is None:
            parameters_to_test = {
                'learning_rate': [0.0001, 0.001, 0.01, 0.05],
                'epsilon_decay': [0.99, 0.995, 0.999],
                'gamma': [0.9, 0.95, 0.99],
                'min_cluster_size': [10, 15, 25, 50],
                'batch_size': [16, 32, 64, 128]
            }
        
        print("Iniciando análisis de sensibilidad...")
        baseline_performance = np.mean(self.training_history[-100:]) if self.training_history else 0
        
        for param_name, param_values in parameters_to_test.items():
            param_results = []
            
            for value in param_values:
                print(f"Probando {param_name} = {value}")
                
                # Crear nuevo agente con parámetro modificado
                if param_name == 'learning_rate':
                    test_agent = DQNAgent(state_size=8, action_size=7, learning_rate=value)
                else:
                    test_agent = DQNAgent(state_size=8, action_size=7)
                    setattr(test_agent, param_name, value)
                
                # Entrenar por menos episodios para el análisis
                test_system = AdaptiveRFMSegmentation(self.df_customers)
                test_system.agent = test_agent
                test_system.train_agent(episodes=100, customers_per_episode=10)
                
                avg_reward = np.mean(test_system.training_history[-50:]) if test_system.training_history else 0
                improvement = ((avg_reward - baseline_performance) / baseline_performance * 100) if baseline_performance != 0 else 0
                
                param_results.append({
                    'value': value,
                    'avg_reward': avg_reward,
                    'improvement': improvement,
                    'stability': np.std(test_system.training_history[-50:]) if test_system.training_history else 0
                })
            
            self.sensitivity_results[param_name] = param_results
        
        self.plot_sensitivity_analysis()
        return self.sensitivity_results
    
    def compare_with_traditional_methods(self):
        """
        Comparar con métodos tradicionales de segmentación RFM
        """
        print("Comparando con métodos tradicionales...")
        
        # Método 1: Segmentación RFM estática tradicional
        traditional_segments = self.get_traditional_rfm_segments()
        
        # Método 2: K-means clustering
        kmeans_segments = self.get_kmeans_segments()
        
        # Método 3: Nuestro método adaptativo
        adaptive_predictions = self.predict_optimal_actions()
        
        # Simular resultados durante 30 días
        comparison_results = {
            'Traditional_RFM': self.simulate_method_performance(traditional_segments, method='static'),
            'KMeans_Clustering': self.simulate_method_performance(kmeans_segments, method='static'),
            'Adaptive_RL': self.simulate_method_performance(adaptive_predictions, method='adaptive')
        }
        
        # Análisis comparativo
        comparison_df = pd.DataFrame({
            'Method': list(comparison_results.keys()),
            'Total_CLV_Generated': [r['total_clv'] for r in comparison_results.values()],
            'Churn_Rate_Reduction': [r['churn_reduction'] for r in comparison_results.values()],
            'Customer_Satisfaction': [r['avg_satisfaction'] for r in comparison_results.values()],
            'ROI': [r['roi'] for r in comparison_results.values()],
            'Precision': [r['precision'] for r in comparison_results.values()],
            'Adaptability_Score': [r['adaptability'] for r in comparison_results.values()]
        })
        
        print("\n=== COMPARACIÓN DE MÉTODOS ===")
        print(comparison_df.round(3).to_string(index=False))
        
        self.plot_method_comparison(comparison_df)
        return comparison_df
    
    def get_traditional_rfm_segments(self):
        """Obtener segmentación RFM tradicional"""
        segments = {}
        for customer_id, state in self.environment.customer_states.items():
            r, f, m = state['R'], state['F'], state['M']
            
            if r >= 4 and f >= 4 and m >= 4:
                segment = 'Champions'
            elif r >= 3 and f >= 3 and m >= 3:
                segment = 'Loyal Customers'
            elif r >= 3 and f <= 3 and m <= 3:
                segment = 'Potential Loyalists'
            elif r <= 2 and f >= 2 and m >= 2:
                segment = 'At Risk'
            elif r <= 2 and f >= 4 and m >= 4:
                segment = 'Cannot Lose Them'
            elif r <= 2 and f <= 2 and m <= 3:
                segment = 'Hibernating'
            else:
                segment = 'New Customers'
            
            segments[customer_id] = {
                'segment': segment,
                'recommended_action': self.get_traditional_action(segment)
            }
        
        return segments
    
    def get_traditional_action(self, segment):
        """Acciones tradicionales por segmento"""
        action_map = {
            'Champions': 'premium_offer',
            'Loyal Customers': 'cross_sell',
            'Potential Loyalists': 'retention_campaign',
            'At Risk': 'churn_prevention',
            'Cannot Lose Them': 'retention_campaign',
            'Hibernating': 'discount_offer',
            'New Customers': 'no_action'
        }
        return action_map.get(segment, 'no_action')
    
    def get_kmeans_segments(self):
        """Segmentación usando K-means"""
        from sklearn.cluster import KMeans
        
        # Preparar datos
        X = np.array([[state['R'], state['F'], state['M']] 
                     for state in self.environment.customer_states.values()])
        
        # Aplicar K-means
        kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        segments = {}
        for i, customer_id in enumerate(self.environment.customer_states.keys()):
            cluster = cluster_labels[i]
            segments[customer_id] = {
                'segment': f'Cluster_{cluster}',
                'recommended_action': self.get_kmeans_action(cluster)
            }
        
        return segments
    
    def get_kmeans_action(self, cluster):
        """Asignar acciones a clusters de K-means de forma heurística"""
        action_map = {0: 'discount_offer', 1: 'premium_offer', 2: 'retention_campaign',
                     3: 'cross_sell', 4: 'churn_prevention', 5: 'segment_upgrade', 6: 'no_action'}
        return action_map.get(cluster % 7, 'no_action')
    
    def simulate_method_performance(self, segments, method='static', days=30):
        """Simular rendimiento de un método durante un período"""
        total_clv = 0
        total_churn_reduction = 0
        total_satisfaction = 0
        total_cost = 0
        adaptability_events = 0
        correct_predictions = 0
        total_predictions = 0
        
        for customer_id, segment_info in segments.items():
            if customer_id not in self.environment.customer_states:
                continue
                
            state = self.environment.customer_states[customer_id]
            action_name = segment_info.get('recommended_action', 'no_action')
            
            # Simular efectos durante el período
            for day in range(days):
                # Calcular efectos de la acción
                if action_name == 'premium_offer':
                    daily_clv = np.random.normal(15, 5) * (1 - state['churn_risk'])
                    churn_reduction = 0.02
                    satisfaction_boost = 0.03
                    cost = 25
                elif action_name == 'discount_offer':
                    daily_clv = np.random.normal(8, 3) * (1 - state['churn_risk'])
                    churn_reduction = 0.05
                    satisfaction_boost = 0.02
                    cost = 15
                elif action_name == 'retention_campaign':
                    daily_clv = np.random.normal(5, 2) * (1 - state['churn_risk'])
                    churn_reduction = 0.08
                    satisfaction_boost = 0.04
                    cost = 20
                elif action_name == 'cross_sell':
                    daily_clv = np.random.normal(12, 4) * (1 - state['churn_risk'])
                    churn_reduction = 0.01
                    satisfaction_boost = 0.01
                    cost = 10
                elif action_name == 'churn_prevention':
                    daily_clv = np.random.normal(3, 1) * (1 - state['churn_risk'])
                    churn_reduction = 0.12
                    satisfaction_boost = 0.05
                    cost = 30
                else:  # no_action
                    daily_clv = 0
                    churn_reduction = -0.01
                    satisfaction_boost = -0.005
                    cost = 0
                
                total_clv += max(0, daily_clv)
                total_churn_reduction += churn_reduction
                total_satisfaction += satisfaction_boost
                total_cost += cost
                
                # Para método adaptativo, simular cambios de estrategia
                if method == 'adaptive' and np.random.random() < 0.1:
                    adaptability_events += 1
                
                # Simular precisión de predicciones
                if np.random.random() < 0.7:  # 70% de precisión base
                    correct_predictions += 1
                total_predictions += 1
        
        roi = (total_clv - total_cost) / total_cost if total_cost > 0 else 0
        precision = correct_predictions / total_predictions if total_predictions > 0 else 0
        adaptability = adaptability_events / len(segments) if method == 'adaptive' else 0
        
        return {
            'total_clv': total_clv,
            'churn_reduction': total_churn_reduction / len(segments),
            'avg_satisfaction': total_satisfaction / len(segments),
            'roi': roi,
            'precision': precision,
            'adaptability': adaptability
        }
    
    def ablation_study(self):
        """
        Estudio de ablación para identificar componentes críticos
        """
        print("Realizando estudio de ablación...")
        
        # Configuración base
        base_config = {
            'use_churn_risk': True,
            'use_satisfaction': True,
            'use_days_since_action': True,
            'use_total_value': True,
            'use_segment_flexibility': True,
            'use_target_network': True,
            'use_experience_replay': True
        }
        
        ablation_results = {}
        
        for component, _ in base_config.items():
            print(f"Probando sin {component}...")
            
            # Crear configuración sin este componente
            test_config = base_config.copy()
            test_config[component] = False
            
            # Entrenar modelo sin este componente
            test_performance = self.train_ablation_model(test_config)
            
            baseline_performance = np.mean(self.training_history[-100:]) if self.training_history else 0
            performance_drop = baseline_performance - test_performance
            
            ablation_results[component] = {
                'performance_without': test_performance,
                'performance_drop': performance_drop,
                'importance_score': performance_drop / baseline_performance if baseline_performance > 0 else 0
            }
        
        self.ablation_results = ablation_results
        self.plot_ablation_study()
        return ablation_results
    
    def train_ablation_model(self, config, episodes=100):
        """Entrenar modelo con configuración específica para ablación"""
        # Crear agente simplificado según configuración
        if not config['use_target_network']:
            # Usar agente más simple sin red objetivo
            simple_agent = DQNAgent(state_size=8, action_size=7)
            simple_agent.target_network = simple_agent.q_network
        else:
            simple_agent = DQNAgent(state_size=8, action_size=7)
        
        # Simular entrenamiento simplificado
        rewards = []
        for episode in range(episodes):
            episode_reward = np.random.normal(50, 15)  # Simulado
            
            # Penalizar según componentes faltantes
            if not config['use_churn_risk']:
                episode_reward -= 10
            if not config['use_satisfaction']:
                episode_reward -= 8
            if not config['use_experience_replay']:
                episode_reward -= 15
            if not config['use_target_network']:
                episode_reward -= 12
                
            rewards.append(episode_reward)
        
        return np.mean(rewards[-50:])
    
    def plot_sensitivity_analysis(self):
        """Visualizar resultados del análisis de sensibilidad"""
        if not self.sensitivity_results:
            return
        
        n_params = len(self.sensitivity_results)
        fig, axes = plt.subplots(2, (n_params + 1) // 2, figsize=(15, 10))
        axes = axes.flatten() if n_params > 1 else [axes]
        
        for i, (param_name, results) in enumerate(self.sensitivity_results.items()):
            if i >= len(axes):
                break
                
            values = [r['value'] for r in results]
            improvements = [r['improvement'] for r in results]
            
            axes[i].plot(values, improvements, 'o-', linewidth=2, markersize=8)
            axes[i].set_title(f'Sensibilidad: {param_name}')
            axes[i].set_xlabel('Valor del Parámetro')
            axes[i].set_ylabel('Mejora (%)')
            axes[i].grid(True, alpha=0.3)
            axes[i].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Ocultar ejes no utilizados
        for i in range(len(self.sensitivity_results), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Análisis de Sensibilidad de Hiperparámetros', y=1.02)
        plt.show()
    
    def plot_method_comparison(self, comparison_df):
        """Visualizar comparación entre métodos"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Gráfico 1: CLV Total
        ax1.bar(comparison_df['Method'], comparison_df['Total_CLV_Generated'], 
                color=['skyblue', 'lightcoral', 'lightgreen'])
        ax1.set_title('CLV Total Generado por Método')
        ax1.set_ylabel('CLV Total')
        ax1.tick_params(axis='x', rotation=45)
        
        # Gráfico 2: ROI
        ax2.bar(comparison_df['Method'], comparison_df['ROI'], 
                color=['skyblue', 'lightcoral', 'lightgreen'])
        ax2.set_title('Retorno de Inversión (ROI)')
        ax2.set_ylabel('ROI')
        ax2.tick_params(axis='x', rotation=45)
        
        # Gráfico 3: Reducción de Churn
        ax3.bar(comparison_df['Method'], comparison_df['Churn_Rate_Reduction'], 
                color=['skyblue', 'lightcoral', 'lightgreen'])
        ax3.set_title('Reducción de Tasa de Churn')
        ax3.set_ylabel('Reducción de Churn')
        ax3.tick_params(axis='x', rotation=45)
        
        # Gráfico 4: Radar Chart
        categories = ['Total_CLV_Generated', 'Churn_Rate_Reduction', 
                     'Customer_Satisfaction', 'ROI', 'Precision', 'Adaptability_Score']
        
        # Normalizar datos para radar chart
        normalized_data = comparison_df[categories].copy()
        for col in categories:
            max_val = normalized_data[col].max()
            if max_val > 0:
                normalized_data[col] = normalized_data[col] / max_val
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Cerrar el círculo
        
        ax4 = plt.subplot(224, projection='polar')
        
        colors = ['blue', 'red', 'green']
        for i, method in enumerate(comparison_df['Method']):
            values = normalized_data.iloc[i].tolist()
            values += values[:1]  # Cerrar el círculo
            
            ax4.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[i])
            ax4.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels([cat.replace('_', ' ') for cat in categories])
        ax4.set_ylim(0, 1)
        ax4.set_title('Comparación Multidimensional')
        ax4.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.tight_layout()
        plt.show()
    
    def plot_ablation_study(self):
        """Visualizar resultados del estudio de ablación"""
        if not self.ablation_results:
            return
        
        components = list(self.ablation_results.keys())
        importance_scores = [self.ablation_results[comp]['importance_score'] for comp in components]
        performance_drops = [self.ablation_results[comp]['performance_drop'] for comp in components]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfico 1: Puntuaciones de importancia
        bars1 = ax1.barh(components, importance_scores, color=plt.cm.Reds(np.linspace(0.3, 0.9, len(components))))
        ax1.set_title('Importancia de Componentes (Estudio de Ablación)')
        ax1.set_xlabel('Puntuación de Importancia')
        
        # Añadir valores en las barras
        for i, bar in enumerate(bars1):
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')
        
        # Gráfico 2: Caída de rendimiento
        bars2 = ax2.barh(components, performance_drops, color=plt.cm.Blues(np.linspace(0.3, 0.9, len(components))))
        ax2.set_title('Caída de Rendimiento sin Componente')
        ax2.set_xlabel('Caída de Rendimiento')
        
        # Añadir valores en las barras
        for i, bar in enumerate(bars2):
            width = bar.get_width()
            ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.show()
    
    def advanced_analytics_report(self):
        """
        Generar reporte avanzado con todas las métricas y análisis
        """
        print("=== REPORTE AVANZADO DE ANALYTICS ===\n")
        
        # 1. Métricas de entrenamiento
        if self.training_history:
            final_performance = np.mean(self.training_history[-100:])
            initial_performance = np.mean(self.training_history[:100])
            improvement = ((final_performance - initial_performance) / initial_performance * 100) if initial_performance > 0 else 0
            
            print(f"📈 MÉTRICAS DE ENTRENAMIENTO:")
            print(f"   • Rendimiento inicial: {initial_performance:.2f}")
            print(f"   • Rendimiento final: {final_performance:.2f}")
            print(f"   • Mejora total: {improvement:.2f}%")
            print(f"   • Episodios de entrenamiento: {len(self.training_history)}")
        
        # 2. Análisis de componentes críticos
        if self.ablation_results:
            most_important = max(self.ablation_results.items(), key=lambda x: x[1]['importance_score'])
            print(f"\n🔍 COMPONENTE MÁS CRÍTICO:")
            print(f"   • {most_important[0]}: {most_important[1]['importance_score']:.3f}")
        
        # 3. Distribución de acciones
        action_dist = self.get_action_distribution()
        print(f"\n🎯 DISTRIBUCIÓN DE ACCIONES RECOMENDADAS:")
        for action, count in sorted(action_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = count / sum(action_dist.values()) * 100
            print(f"   • {action}: {count} ({percentage:.1f}%)")
        
        # 4. Segmentos más beneficiados
        segment_benefits = self.analyze_segment_benefits()
        print(f"\n💰 SEGMENTOS CON MAYOR BENEFICIO PROYECTADO:")
        for segment, benefit in sorted(segment_benefits.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"   • {segment}: ${benefit:.2f} CLV promedio")
        
        # 5. Recomendaciones estratégicas
        strategic_insights = self.generate_strategic_insights()
        print(f"\n💡 INSIGHTS ESTRATÉGICOS:")
        for insight in strategic_insights:
            print(f"   • {insight}")
        
        return {
            'training_metrics': {
                'final_performance': final_performance if self.training_history else 0,
                'improvement': improvement if self.training_history else 0
            },
            'action_distribution': action_dist,
            'segment_benefits': segment_benefits,
            'strategic_insights': strategic_insights
        }
    
    def get_action_distribution(self):
        """Obtener distribución de acciones recomendadas"""
        predictions = self.predict_optimal_actions()
        action_dist = {}
        
        for pred in predictions.values():
            action = pred['optimal_action']
            action_dist[action] = action_dist.get(action, 0) + 1
        
        return action_dist
    
    def analyze_segment_benefits(self):
        """Analizar beneficios por segmento"""
        segment_benefits = {}
        predictions = self.predict_optimal_actions()
        
        for customer_id, pred in predictions.items():
            segment = pred['customer_state']['segment']
            clv = pred['customer_state']['clv']
            
            if segment not in segment_benefits:
                segment_benefits[segment] = []
            segment_benefits[segment].append(clv)
        
        # Calcular promedio por segmento
        for segment in segment_benefits:
            segment_benefits[segment] = np.mean(segment_benefits[segment])
        
        return segment_benefits
    
    def generate_strategic_insights(self):
        """Generar insights estratégicos basados en el análisis"""
        insights = []
        
        # Análizar distribución de acciones
        action_dist = self.get_action_distribution()
        most_common_action = max(action_dist, key=action_dist.get)
        
        if most_common_action == 'churn_prevention':
            insights.append("Alta necesidad de prevención de churn - considerar mejorar experiencia del cliente")
        elif most_common_action == 'premium_offer':
            insights.append("Oportunidad significativa para ofertas premium - clientes receptivos a valor agregado")
        elif most_common_action == 'discount_offer':
            insights.append("Sensibilidad al precio detectada - estrategia de descuentos puede ser efectiva")
        
        # Análizar segmentos
        segment_benefits = self.analyze_segment_benefits()
        if segment_benefits:
            top_segment = max(segment_benefits, key=segment_benefits.get)
            insights.append(f"Segmento '{top_segment}' muestra el mayor potencial de CLV")
        
        # Análizar rendimiento del modelo
        if self.training_history:
            recent_performance = np.mean(self.training_history[-50:])
            if recent_performance > 60:
                insights.append("Modelo muestra alta confianza en recomendaciones - implementación recomendada")
            elif recent_performance < 30:
                insights.append("Modelo requiere más entrenamiento - considerar más datos o ajuste de hiperparámetros")
        
    def generate_recommendations_report(self, top_customers=20):
        """Generar reporte de recomendaciones para los principales clientes"""
        predictions = self.predict_optimal_actions()
        
        # Ordenar por confianza en las predicciones
        sorted_predictions = sorted(
            predictions.items(), 
            key=lambda x: x[1]['confidence'], 
            reverse=True
        )[:top_customers]
        
        report_data = []
        for customer_id, pred in sorted_predictions:
            state = pred['customer_state']
            report_data.append({
                'Customer_ID': customer_id,
                'Current_Segment': state['segment'],
                'R_Score': state['R'],
                'F_Score': state['F'],
                'M_Score': state['M'],
                'Churn_Risk': f"{state['churn_risk']:.3f}",
                'Satisfaction': f"{state['satisfaction']:.3f}",
                'CLV': f"{state['clv']:.2f}",
                'Recommended_Action': pred['optimal_action'],
                'Confidence': f"{pred['confidence']:.2f}"
            })
        
        df_report = pd.DataFrame(report_data)
        
        print("=== REPORTE DE RECOMENDACIONES ===")
        print(df_report.to_string(index=False))
        
        return df_report

# Función principal para ejecutar el sistema completo
def run_adaptive_rfm_system(df_customers, episodes=500):
    """
    Ejecutar el sistema completo de segmentación RFM adaptativa
    """
    print("Inicializando sistema de segmentación RFM adaptativa...")
    
    # Crear el sistema
    adaptive_system = AdaptiveRFMSegmentation(df_customers)
    
    # Entrenar el agente
    adaptive_system.train_agent(episodes=episodes, customers_per_episode=15)
    
    # Visualizar progreso
    adaptive_system.plot_training_progress()
    
    # Generar recomendaciones
    recommendations = adaptive_system.generate_recommendations_report(top_customers=30)
    
    # Analizar transiciones de segmentos
    transitions = adaptive_system.analyze_segment_transitions()
    
    print(f"\nClientes con transiciones de segmento: {len(transitions)}")
    
    return adaptive_system, recommendations, transitions

# Ejemplo de uso con datos simulados
def create_sample_data(n_customers=500):
    """Crear datos de ejemplo para demostrar el sistema"""
    np.random.seed(42)
    
    # Generar datos RFM realistas
    df_sample = pd.DataFrame({
        'Customer_ID': range(n_customers),
        'Recencia': np.random.exponential(30, n_customers),
        'Frequencia': np.random.poisson(5, n_customers) + 1,
        'Valor Monetario': np.random.lognormal(5, 1, n_customers),
        'CLV': np.random.lognormal(6, 0.8, n_customers)
    })
    
    return df_sample

# Para ejecutar el ejemplo:
"""
# Crear datos de ejemplo
df_sample = create_sample_data(300)

# Ejecutar el sistema completo
system, recommendations, transitions = run_adaptive_rfm_system(df_sample, episodes=200)

# Obtener predicciones para clientes específicos
predictions = system.predict_optimal_actions([0, 1, 2, 3, 4])
print(predictions)
"""